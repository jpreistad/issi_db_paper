#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:33:22 2023

@author: jone
Helper functions used to investigate magnetic perturbations in GEMINI run

"""
import sys
import numpy as np
sys.path.append('/Users/jone/Dropbox (Personal)/uib/researcher/git/e3dsecs/')
from e3dsecs import simulation, grid, coordinates, data
from gemini3d.grid.convert import geomag2geog, geog2geomag
import secsy
from scipy.linalg import lstsq, solve
from scipy.interpolate import griddata
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import dipole # https://github.com/klaundal/dipole
import polplot # https://github.com/klaundal/polplot



RE = 6371.2 #Earth radius in km

def gemini_vec_2_enu_vec(gemini_vec, lon, lat, geographic=True):
    '''
    Convert a gemini vector (given along e1,e2,e3 directions) to local 
    ENU components at the input lon, lat locations. geographic keyword control whether
    the returned components are in geographic ENU (default) or geomagnetic ENU.
    
    This function can replace the 
    compute_enu_components() function, as this is more general and not locked 
    to the GEMINI grid. 
    I dont understand why altitude is not part of this calculation. Probably 
    because the e1 vector has the same angle to the main B field regardless
    of the distance r along the field line. The results
    obtained here reproduces the ones provided in the xg grid dict, so it should 
    be correct. Note that a factor 3 was added to the z-component in eq 124 in the
    gemini doc. This was a typo in their derivation.
    
    Parameters
    ----------
    gemini_vec : array_like
        (N,3) array of N vectors represented along e1, e2, e3 directions
    lon : array-like
        longitude of the vector to convert, in degrees.
    lat : array-like
        latitude of the vector to convert, in degrees.
    alt : array-like
        altitude in km of the vector to convert.

    Returns
    -------
    (N,3) shape array of ENU components of input vector.

    '''
    
    # Get cartesian geomagnetic ECEF components of local unit vector at (lon, lat)
    egmlon, egmlat, egmalt = coordinates.unitvecs_geographic_general(lon, lat, dipole=geographic)
    if geographic:
        phi, theta = geog2geomag(lon, lat) # degrees input, radians out
    else:
        phi = np.radians(lon)
        theta = np.pi/2 - np.radians(lat)
    
    # Get cartesian geomagnetic components of local (e1, e2, e3) unit vector
    # at (lat, lon)
    # Will use eqs 123-125 in the GEMINI document. 
    sf = np.sqrt(1+3*(np.cos(theta))**2)
    sfm = 1-3*(np.cos(theta))**2
    e1 = np.array([-3*np.cos(theta)*np.sin(theta)*np.cos(phi)/sf, 
                   -3*np.cos(theta)*np.sin(theta)*np.sin(phi)/sf, 
                   sfm/sf]).T
    e2 = np.array([np.cos(phi)*sfm/sf, np.sin(phi)*sfm/sf, 
                   3*np.cos(theta)*np.sin(theta)/sf]).T
    e3 = np.array([-np.sin(phi), np.cos(phi), np.zeros(phi.size)]).T
    
    # Project each GEMINI component (1,2,3) of gemini_vec onto the local ENU directions
    vgalt=( np.sum(e1*egmalt,1)*gemini_vec[:,0] + 
           np.sum(e2*egmalt,1)*gemini_vec[:,1] + 
           np.sum(e3*egmalt,1)*gemini_vec[:,2] )
    vglat=( np.sum(e1*egmlat,1)*gemini_vec[:,0] + 
           np.sum(e2*egmlat,1)*gemini_vec[:,1] +
           np.sum(e3*egmlat,1)*gemini_vec[:,2] )
    vglon=( np.sum(e1*egmlon,1)*gemini_vec[:,0] + 
           np.sum(e2*egmlon,1)*gemini_vec[:,1] + 
           np.sum(e3*egmlon,1)*gemini_vec[:,2] )   
    
    return vglon, vglat, vgalt


def height_integtated_current(sim, gr, ENU=False, maph=110, interpolate=True, 
                              conductance=False):
    """Calculate height integrated current from GEMINI data.

    Args:
        sim (instance of E3DSECS simulation class): Contain the GEMINI data and grid
        gr (instance of E3DSECS grid class): Contain the grids
        ENU (bool, optional): Specified how to do the height integration. If True,
            a vertical height integration is done. If False, a field-aligned integration
            is done. Its value will be associated with the geographic lat/lon of the where
            the field-line intersectedthe maph altitude. Defaults to False.
        maph (int, optional): Altitude in km of the height integrated value. 
            Defaults to 110km.
        interpolate (bool, optional): Wheter to interpolate the height integrated values 
            from GEMINI on the gr.grid.lat/lon_mesh grid to aviod the singularity issue
            with the SECS representation. Defaults to True.
        conductance (bool, optional): Wheter to also return Hall and Pedersen conductance

    Returns:
        The height integrated current and their respective locations:
        4 element tuple: Je, Jn, glon, glat
    """    
    xg = sim.xg
    dat = sim.dat
    grid = gr.grid
    
    shape = xg['alt'].shape
    if not ENU:
        # Calculate inclination
        # H = np.sqrt(dat.Be**2+dat.Bn**2)
        # I = np.degrees(np.arctan(H/np.abs(dat.Bu)))
        I = 11 # degrees inclination. 
        dx1 = -np.diff(xg['alt'], axis=0) * np.cos(np.deg2rad(I)) # in meters, first index at highest altitude
        j2 = dat.J2.values[0:shape[0]-1,:,:]
        J2 = np.sum(j2 * dx1, axis=0) # in A/m
        j3 = dat.J3.values[0:shape[0]-1,:,:]
        J3 = np.sum(j3 * dx1, axis=0) # in A/m
        sh = dat.sh.values[0:shape[0]-1,:,:]
        SH = np.sum(sh * dx1, axis=0) # in S
        sp = dat.sp.values[0:shape[0]-1,:,:]
        SP = np.sum(sp * dx1, axis=0) # in S
        
        # Convert the height integrated current J2 and J3 into ENU components, at maph height
        gemini_vec = np.vstack((np.zeros(J2.size),J2.flatten(), J3.flatten())).T
        k_s = np.argmin(np.abs(xg['alt']-maph*1000), axis=0)
        i_s = np.tile(np.arange(shape[1])[:,np.newaxis],shape[2])
        j_s = np.tile(np.arange(shape[2])[:,np.newaxis],shape[1]).T
        kij_s = np.ravel_multi_index((k_s,i_s,j_s), shape).flatten()
        lons = xg['glon'].flatten()[kij_s]
        large = lons > 180
        lons[large] = lons[large] - 360
        lats = xg['glat'].flatten()[kij_s]
        Je, Jn, Ju, = gemini_vec_2_enu_vec(gemini_vec, lons, lats)# geographic components
        J_enu_gg = np.vstack((Je,Jn,Ju)).T
        # J_enu_gm = coordinates.enugg2enugm(J_enu_gg, lons, lats) # Convert from geographic to geomag components
        Je = J_enu_gg[:,0]
        Jn = J_enu_gg[:,1]
        # mlons = np.degrees(xg['phi'].flatten()[kij_s])
        # mlats = 90-np.degrees(xg['theta'].flatten()[kij_s])
        glons = xg['glon'].flatten()[kij_s]
        small_lon = glons > 180
        glons[small_lon] = glons[small_lon] - 360
        glats = xg['glat'].flatten()[kij_s]
        
        if interpolate: # Interpolate onto grid locations to avoid possible singularity probmlems
            points = np.vstack((glons, glats)).T
            Je = griddata(points, Je, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
            Jn = griddata(points, Jn, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
            glons = grid.lon_mesh.flatten()
            glats = grid.lat_mesh.flatten()
            if conductance:
                SH = griddata(points, SH, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
                SP = griddata(points, SP, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
                return (Je, Jn, SH, SP, glons, glats)
            else:
                return (Je, Jn, glons, glats)
        else:
            if conductance:
                return (Je, Jn, SH, SP, glons, glats)
            else:
                return (Je, Jn, glons, glats)
    
    else: #Height integrate in vertical direction
        # Need to sample ENU components of j on a new grid that is convenient to integrate on
        minglon = -45
        maxglon = 45
        minglat = 55
        maxglat = 85
        Nlat = 300
        Nlon = 100
        _glats = np.linspace(minglat, maxglat, Nlat)
        _glons = np.linspace(minglon, maxglon, Nlon)
        _alts = np.concatenate((np.arange(90,140,2),np.arange(140,170,5),np.arange(170,230,10),np.arange(230,830,50)))
        alts, glons, glats = np.meshgrid(_alts, _glons, _glats, indexing='ij')
        _dat = data.data(gr, sim, beams=False, points=True, 
                  lat_ev=glats, lon_ev=glons, alt_ev=alts, 
                  e3doubt_=False)
        je = _dat.je.reshape(glons.shape)
        jn = _dat.jn.reshape(glons.shape)
        _altres = np.diff(_alts)
        _altres = np.abs(np.concatenate((np.array([_altres[0]]),_altres)))
        altres,_,__ = np.meshgrid(_altres, np.ones(Nlon), np.ones(Nlat), indexing='ij')
        Je = np.sum(je*altres*1e3, axis=0).flatten() #gg components
        Jn = np.sum(jn*altres*1e3, axis=0).flatten() #gg components
        sh = _dat.sh.reshape(glons.shape)
        SH = np.sum(sh*altres*1e3, axis=0).flatten()
        sp = _dat.sp.reshape(glons.shape)
        SP = np.sum(sp*altres*1e3, axis=0).flatten()
        _glats = glats[0,:,:].flatten()
        _glons = glons[0,:,:].flatten()
        if interpolate:
            points = np.vstack((_glons, _glats)).T
            Je = griddata(points, Je, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
            Jn = griddata(points, Jn, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
            _glons = grid.lon_mesh.flatten()
            _glats = grid.lat_mesh.flatten()
            if conductance:
                SH = griddata(points, SH, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
                SP = griddata(points, SP, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
                return (Je, Jn, SH, SP, _glons, _glats)
            else:
                return (Je, Jn, _glons, _glats)
        else:
            if conductance:
                return (Je, Jn, SH, SP, _glons, _glats)
            else:
                return (Je, Jn, _glons, _glats)
        
    
def fit_J(grid, Je, Jn, lons, lats, extend=1, maph=110, l1=1e-5, l2=0, singularity=1, 
          laplacian=False, lcurve=False):
    Je = Je.flatten()
    Jn = Jn.flatten()
    lons = lons.flatten()
    lats = lats.flatten()
    use = grid.ingrid(lons, lats, ext_factor=-extend) & np.isfinite(Je)
    Ge_cf, Gn_cf = secsy.get_SECS_J_G_matrices(lats[use], lons[use],
                grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + maph * 1e3, current_type = 'curl_free', 
                singularity_limit=grid.Lres*0.5*singularity)
    Gcf = np.vstack((Ge_cf, Gn_cf))
    Ge_df, Gn_df = secsy.get_SECS_J_G_matrices(lats[use], lons[use],
                grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + maph * 1e3, current_type = 'divergence_free', 
                singularity_limit=grid.Lres*0.5*singularity)
    Gdf = np.vstack((Ge_df, Gn_df))
    if laplacian:
        Le = np.zeros((Ge_cf.shape[0],2))
        Le[:,0] = 1
        Ln = np.zeros((Ge_cf.shape[0],2))
        Ln[:,1] = 1
        La = np.vstack((Le, Ln))
        G = np.hstack((Gcf,Gdf,La))
    else:
        G = np.hstack((Gcf,Gdf))
    d = np.hstack((Je[use], Jn[use]))
    GTG = G.T.dot(G)
    GTd = G.T.dot(d)
    gtg_mag = np.median(np.diagonal(GTG))

    if (l2 != 0) & laplacian:
        print('east-west regularization with laplacian=True not implemented')
        print(1/0)
    elif l2>0:
        De2, Dn2 = grid.get_Le_Ln()
        L = De2
        _LTL = L.T.dot(L)
        LTL = np.zeros(GTG.shape)
        _m = int(GTG.shape[0]/2)
        LTL[0:_m,0:_m] = _LTL
        LTL[_m:,_m:] = _LTL
        ltl_mag = np.median(LTL.diagonal())
        GG = GTG + l1*gtg_mag * np.eye(GTG.shape[0]) + l2 * gtg_mag / ltl_mag * LTL
    else:
        GG = GTG + l1*gtg_mag * np.eye(GTG.shape[0])
        
    if lcurve:
        resnorm, modelnorm, ls = Lcurve(G, d, steps=10)
        return(resnorm, modelnorm, ls)
    
    # m = lstsq(GG, GTd, cond=0.)[0]
    m = solve(GG, GTd)

    if laplacian:
        m_cf = m[0:(m.shape[0]-2)//2]
        m_df = m[(m.shape[0]-2)//2:-2]
        m_l = m[-2:]
        return (m_cf, m_df, m_l)
    else:
        m_cf = m[0:m.shape[0]//2]
        m_df = m[m.shape[0]//2:]        
        return (m_cf, m_df)


def Lcurve(G, d, steps=10):
    GTG = G.T.dot(G)
    GTd = G.T.dot(d)
    gtg_mag = np.median(np.diagonal(GTG))
    ls = np.linspace(-4,0,steps)
    resnorm = []
    modelnorm = []
    for l in ls:
        GG = GTG + 10**l * gtg_mag * np.eye(GTG.shape[0])
        m = solve(GG, GTd)
        # m = lstsq(GG, GTd, cond=0.)[0]
        res = (d - G.dot(m))
        resnorm.append(np.sqrt(np.sum(res**2)))
        modelnorm.append(np.sqrt(np.sum(m**2)))
        print(l)
    plt.figure()
    plt.plot(resnorm, modelnorm)
    plt.scatter(resnorm, modelnorm)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('log residual norm')
    plt.ylabel('log model norm')
    for i in range(steps):
        sss = '%4.1f' % ls[i]
        plt.text(resnorm[i], modelnorm[i], sss)

    return (resnorm, modelnorm, ls)


def evalJ(gr, m_cf, m_df, maph=110, singularity=1):
    # Evaluates the SECS model of he height integrated currents.
    # gr is an instance of the grid object, containging both the inner and extended grid
    # The use of the Laplacian part is not yet implemented here yet.
    # use = grid.ingrid(grid.lon_mesh.flatten(), grid.lat_mesh.flatten(), ext_factor=-extend)
    Ge_, Gn_ = secsy.get_SECS_J_G_matrices(gr.grid.lat_mesh.flatten(),
                gr.grid.lon_mesh.flatten(),  
                gr.grid_l.lat.flatten(), gr.grid_l.lon.flatten(), 
                constant = 1./(4.*np.pi), RI=RE * 1e3 + maph * 1e3, 
                current_type = 'curl_free', singularity_limit=gr.grid.Lres*0.5*singularity)
    G_pred = np.vstack((Ge_, Gn_))
    Jcf = G_pred.dot(m_cf)
    Ge_, Gn_ = secsy.get_SECS_J_G_matrices(gr.grid.lat_mesh.flatten(), 
                gr.grid.lon_mesh.flatten(),  
                gr.grid_l.lat.flatten(), gr.grid_l.lon.flatten(), 
                constant = 1./(4.*np.pi), RI=RE * 1e3 + maph * 1e3, 
                current_type = 'divergence_free', singularity_limit=gr.grid.Lres*0.5*singularity)
    G_pred = np.vstack((Ge_, Gn_))
    Jdf = G_pred.dot(m_df)
    N = Ge_.shape[0]
    Jdf_e = Jdf[0:N]
    Jdf_n = Jdf[N:2*N]
    Jcf_e = Jcf[0:N]
    Jcf_n = Jcf[N:2*N]

    return (Jcf_e, Jcf_n, Jdf_e, Jdf_n, gr.grid.lon_mesh.flatten(), gr.grid.lat_mesh.flatten())


def show_grids(xg, grid, pax, csgrid=True):
    # Draw GEMINI region
    shape = xg['alt'].shape
    maph = 110
    k_s = np.argmin(np.abs(xg['alt']-maph*1000), axis=0)
    i_s = np.tile(np.arange(shape[1])[:,np.newaxis],shape[2])
    j_s = np.tile(np.arange(shape[2])[:,np.newaxis],shape[1]).T
    kij_s = np.ravel_multi_index((k_s,i_s,j_s), shape)
    glons = xg['glon'].flatten()[kij_s]
    large = glons > 180
    glons[large] = glons[large] - 360
    glats = xg['glat'].flatten()[kij_s]
    pax.plot(glats[:,0], glons[:,0]/15, color='green')    
    pax.plot(glats[:,-1], glons[:,-1]/15, color='green') 
    pax.plot(glats[0,:], glons[0,:]/15, color='green')    
    # pax.plot(glats[0:-1], glons[0:-1]/15, color='red') 
    
    if csgrid:
        # Draw CS grid region
        glat = grid.lat_mesh
        glon = grid.lon_mesh
        pax.plot(glat[:,0], glon[:,0]/15, color='orange')
        pax.plot(glat[:,-1], glon[:,-1]/15, color='orange')
        pax.plot(glat[0,:], glon[0,:]/15, color='orange')
        pax.plot(glat[-1,:], glon[-1,:]/15, color='orange')


    # plt.tight_layout()
    # plt.show()


def make_pax(ax, dp):
    left_lt = 20
    right_lt = 4
    minlat = 55
    pax = polplot.Polarplot(ax, minlat = minlat, sector = str(left_lt) + '-' + str(right_lt))
    pax.coastlines(linewidth = .4, color = 'grey')    
    for lat in np.r_[50:81:10]:
        lon = np.linspace(0, 360, 360)
        glat, glon, _, _ = dp.mag2geo(lat, lon, lat, lat)
        pax.plot(glat, glon/15, color = 'C0', zorder = 1, linewidth = .4)
    # plot dipole meridians
    for lon in np.r_[0:351:30]:
        lat = np.linspace(0, 90, 190)
        glat, glon, _, _ = dp.mag2geo(lat, lon, lat, lat)
        pax.plot(glat, glon/15, color = 'C0', zorder = 1, linewidth = .4)
    # draw frame
    pax.plot([minlat, 90], [left_lt, left_lt], color = 'black')
    pax.plot([minlat, 90], [right_lt, right_lt], color = 'black')
    pax.plot( np.full(100, minlat), np.linspace(left_lt, 24 + right_lt, 100), 
             color = 'black')
    return pax