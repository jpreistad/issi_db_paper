#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:33:22 2023

@author: jone
Helper functions used to investigate magnetic perturbations in GEMINI run

"""
import numpy as np
import git.secs_3d as secs3d
from gemini3d.grid.convert import geomag2geog, geog2geomag
import secsy
from scipy.linalg import lstsq, solve
from scipy.interpolate import griddata
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import dipole # https://github.com/klaundal/dipole



RE = 6371.2 #Earth radius in km

def height_integtated_current(xg, dat, grid, ENU=False, maph=110, interpolate=True):
    shape = xg['alt'].shape
    if not ENU:
        # Calculate inclination
        # H = np.sqrt(dat.Be**2+dat.Bn**2)
        # I = np.degrees(np.arctan(H/np.abs(dat.Bu)))
        I = 10 # degrees
        dx1 = -np.diff(xg['alt'], axis=0) * np.cos(np.deg2rad(I)) # in meters, first index at highest altitude
        j2 = dat.J2.values[0:shape[0]-1,:,:]
        J2 = np.sum(j2 * dx1, axis=0) # in A/m
        j3 = dat.J3.values[0:shape[0]-1,:,:]
        J3 = np.sum(j3 * dx1, axis=0) # in A/m
        
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
        Je, Jn, Ju, = secs3d.gemini_tools.gemini_vec_2_enu_vec(gemini_vec, lons, lats)# geographic components
        J_enu_gg = np.vstack((Je,Jn,Ju)).T
        J_enu_gm = secs3d.gemini_tools.enugg2enugm(J_enu_gg, lons, lats) # Convert from geographic to geomag components
        Je = J_enu_gm[:,0]
        Jn = J_enu_gm[:,1]
        mlons = np.degrees(xg['phi'].flatten()[kij_s])
        mlats = 90-np.degrees(xg['theta'].flatten()[kij_s])
        
        if interpolate: # Interpolate onto grid locations to avoid singularity probmlems
            points = np.vstack((mlons, mlats)).T
            _Je = griddata(points, Je, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
            _Jn = griddata(points, Jn, (grid.lon_mesh, grid.lat_mesh), method='linear').flatten()
            use = np.isfinite(_Je)
    
            return (_Je[use], _Jn[use], grid.lon_mesh.flatten()[use], grid.lat_mesh.flatten()[use])
        else:
            return (Je, Jn, mlons, mlats)
    
    else: #Height integrate in vertical direction
        minmlon = 60
        maxmlon = 140
        minmlat = 60
        maxmlat = 85
        Nlat = 300
        Nlon = 100
        _mlats = np.linspace(minmlat, maxmlat, Nlat)
        _mlons = np.linspace(minmlon, maxmlon, Nlon)
        _alts = np.concatenate((np.arange(90,140,2),np.arange(140,170,5),np.arange(170,230,10),np.arange(230,830,50)))
        alts, mlons, mlats = np.meshgrid(_alts, _mlons, _mlats, indexing='ij')
        glons, glats = geomag2geog(np.radians(mlons), np.radians(90-mlats)) #returns in degrees
        datadict = secs3d.gemini_tools.sample_points(xg, dat, glats.flatten(), glons.flatten(), alts.flatten())
        je = datadict['je'].reshape(glons.shape)
        jn = datadict['jn'].reshape(glons.shape)
        # ju = datadict['ju'].reshape(glons.shape)
        _altres = np.diff(_alts)
        _altres = np.abs(np.concatenate((np.array([_altres[0]]),_altres)))
        altres,_,__ = np.meshgrid(_altres, np.ones(Nlon), np.ones(Nlat), indexing='ij')
        Je = np.sum(je*altres*1e3, axis=0).flatten() #gg components
        Jn = np.sum(jn*altres*1e3, axis=0).flatten() #gg components
        
        J_enu_gg = np.vstack((Je,Jn,np.zeros(Je.shape))).T
        J_enu_gm = secs3d.gemini_tools.enugg2enugm(J_enu_gg, glons[0,:,:].flatten(), glats[0,:,:].flatten()) # Convert from geographic to geomag components
        Je = J_enu_gm[:,0].reshape(Nlon,Nlat)
        Jn = J_enu_gm[:,1].reshape(Nlon,Nlat)

        return (Je, Jn, mlons[0,:,:], mlats[0,:,:])
    
def fit_J(grid, Je, Jn, mlons, mlats, extend=1, maph=110, l1=1e-5, l2=0, singularity=1, 
          laplacian=True, lcurve=False):
    Je = Je.flatten()
    Jn = Jn.flatten()
    mlons = mlons.flatten()
    mlats = mlats.flatten()
    use = grid.ingrid(mlons, mlats, ext_factor=-extend)
    Ge_cf, Gn_cf = secsy.get_SECS_J_G_matrices(mlats[use], mlons[use],
                grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + maph * 1e3, current_type = 'curl_free', 
                singularity_limit=grid.Lres*0.5*singularity)
    Gcf = np.vstack((Ge_cf, Gn_cf))
    Ge_df, Gn_df = secsy.get_SECS_J_G_matrices(mlats[use], mlons[use],
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
    ls = np.linspace(-25,2,steps)
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
    
    plt.plot(resnorm, modelnorm)
    plt.scatter(resnorm, modelnorm)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('log residual norm')
    plt.ylabel('log model norm')

    return (resnorm, modelnorm, ls)

def evalJ(grid, m_cf, m_df, maph=110, extend=1, singularity=1):
    use = grid.ingrid(grid.lon_mesh.flatten(), grid.lat_mesh.flatten(), ext_factor=-extend)
    Ge_, Gn_ = secsy.get_SECS_J_G_matrices(grid.lat_mesh.flatten()[use],
                grid.lon_mesh.flatten()[use],  
                grid.lat.flatten(), grid.lon.flatten(), 
                constant = 1./(4.*np.pi), RI=RE * 1e3 + maph * 1e3, 
                current_type = 'curl_free', singularity_limit=grid.Lres*0.5*singularity)
    G_pred = np.vstack((Ge_, Gn_))
    Jcf = G_pred.dot(m_cf)
    Ge_, Gn_ = secsy.get_SECS_J_G_matrices(grid.lat_mesh.flatten()[use], 
                grid.lon_mesh.flatten()[use],  
                grid.lat.flatten(), grid.lon.flatten(), 
                constant = 1./(4.*np.pi), RI=RE * 1e3 + maph * 1e3, 
                current_type = 'divergence_free', singularity_limit=grid.Lres*0.5*singularity)
    G_pred = np.vstack((Ge_, Gn_))
    Jdf = G_pred.dot(m_df)
    N = Ge_.shape[0]
    Jdf_e = Jdf[0:N]
    Jdf_n = Jdf[N:2*N]
    Jcf_e = Jcf[0:N]
    Jcf_n = Jcf[N:2*N]

    return (Jcf_e, Jcf_n, Jdf_e, Jdf_n, grid.lon_mesh.flatten()[use], grid.lat_mesh.flatten()[use])

def show_grids(xg, grid, pax, csgrid=True):
    dp = dipole.Dipole(dipole_pole=((90-11),289))

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
        glat, glon = dp.mag2geo(grid.lat_mesh, grid.lon_mesh)
        pax.plot(glat[:,0], glon[:,0]/15, color='orange')
        pax.plot(glat[:,-1], glon[:,-1]/15, color='orange')
        pax.plot(glat[0,:], glon[0,:]/15, color='orange')
        pax.plot(glat[-1,:], glon[-1,:]/15, color='orange')


    # plt.tight_layout()
    # plt.show()