#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:21:29 2023

@author: jone

Script that will process and analyze GEMINI run that is intended to use in
ISSI team review paper on magnetic disturbances on ground.

"""

import sys
sys.path.append('/Users/jone/BCSS-DAG Dropbox/Jone Reistad/git/DAG/src')
import numpy as np
sys.path.append('/Users/jone/Dropbox (Personal)/uib/researcher/git/e3dsecs/')
from e3dsecs import simulation, grid, data
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from gemini3d.grid.convert import geomag2geog, geog2geomag
import secsy
import helpers
from matplotlib import colors, colorbar
from scipy.interpolate import griddata
import fac_input_to_matt
import xarray as xr
import dipole # https://github.com/klaundal/dipole
import gemini3d.magtools


#Global variables
RE = 6371.2 #Earth radius in km
maph = 110 # Height in km of SECS representation of height integrated currents
ENU = True # Do heith integration in vertical direction, in contrast to field aligned
interpolate = True # interpolate height integrated current on secs mesh grid (avoid singularity)
laplacian = False # Only fit two parameters to describe the const. filed, to try to avoid the "frame". Should set extend=0 if attempting this. Does not seem to work as intended...
extend = 4 # SECS padding "frames"
singularity = 0.5 # how many grid cells to modulate
l1 = 10**(-0.2) # reg parameter
l2 = 0#1e-1       # reg parameter
lcurve = False  # Use to make plot to determine l1
profile_mlon = 90 # magnetic longitude of the lat. profile cut to show
###########################################

# According to the GEMINI docs, this is the centered dipole they use in GEMINI: 
# https://zenodo.org/record/3903830/files/GEMINI.pdf
M = 7.94e22 # magnetic moment in A/m^2
mu0 = 4*np.pi*10**(-7)
B0 = mu0*M/(4*np.pi*RE**3) # from e.g. https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/RG025i001p00001
dp = dipole.Dipole(dipole_pole=((90-11),289), B0=B0)

# Load GEMINI grid and data
# Update the path variable below. The simulation can be found on dropbox:
# 'your_bcss_dropbox_path/Data/issi_team_506_gemini_run/aurora_EISCAT3D/'
path = "/Users/jone/Dropbox (Personal)/uib/researcher/tmpfiles/aurora_EISCAT3D/"
# sim: Instance of simulation object, contain the GEMINI grid (sim.xg) and the output 
#      on this grid in sim.dat
sim = simulation.simulation(path, maph=maph, timeindex=61)
# gr:   Instance of the grid object. Contain two CS grids. gr.grid and gr.grid_l
#       grid_l is the extended one, that is padded with # of "frames" as specified
#       with the extend keyword. Besides that, the grids are identical in the interior.
gr = grid.grid(sim, extend=extend, dlat=1.5, dlon=-3, resolution_factor=0.3, 
               crop_factor=0.67, extend_ew=4.5, asymres=1.5, orientation=-28)
daysec = sim.dat.time.dt.hour.values * 3600 + sim.dat.time.dt.minute.values*60
ff = path+"magfields/20160303_%5i.000000.h5" % daysec
magdat = gemini3d.magtools.magframe(ff) # The Biot-Savart integration data from GEMINI

# Get height integrated currents
Je, Jn, glons, glats = helpers.height_integtated_current(sim, gr, ENU=ENU, maph=maph, 
                                                         interpolate=interpolate)

# Fit the height integrated current with CF + DF SECS
m = helpers.fit_J(gr.grid_l, Je, Jn, glons, glats, l1=l1, l2=l2, singularity=singularity, 
                       laplacian=laplacian, extend=extend, lcurve = lcurve, maph=maph)
m_cf = m[0]
m_df = m[1]
#################################################


##################################################
# PLOTTING
##################################################


##################################################
# Plot comparing height integrated currents from GEMINI with the fitted SECS current
# The eval function always evaluate at the mesh locations, and only on the inner grid.
if interpolate:
    fig,axs = plt.subplots(1,1,figsize=(10,7))
    pax = helpers.make_pax(axs, dp)
    pax.ax.set_title('SECS fit vs. height integrated $J_{hor}$ from GEMINI')
    helpers.show_grids(sim.xg, gr.grid_l, pax, csgrid=True)
    # Plot the height integrated currents
    kk = 5
    pax.quiver(gr.grid.lat_mesh.flatten()[::kk], gr.grid.lon_mesh.flatten()[::kk]/15, Jn[::kk], 
            Je[::kk], color='blue', label='GEMINI', alpha=0.5)
    Jcf_e, Jcf_n, Jdf_e, Jdf_n, lon, lat = helpers.evalJ(gr, m_cf, m_df, maph=maph)
    Je_fit = Jcf_e + Jdf_e
    Jn_fit = Jcf_n + Jdf_n
    pax.quiver(gr.grid.lat_mesh.flatten()[::kk], gr.grid.lon_mesh.flatten()[::kk]/15, Jn_fit[::kk], 
            Je_fit[::kk], color='red', label='SECS', alpha=0.5)
    fig.savefig('./plots/secs_fit_currents.pdf')

########################################
# Plot of SECS amplitudes vs GEMINI FAC
fig,axs = plt.subplots(1,2,figsize=(10,5))
#Colorbar
cax = fig.add_axes((0.1,0.15,0.8,0.02))
clim= 4e-6 #A/m2
norm = colors.Normalize(vmin=-clim, vmax=clim)
cb1 = colorbar.ColorbarBase(cax, cmap='bwr',
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[$A/m^2$]')
# SECS fit
fac = (m_cf.reshape(gr.grid_l.lon.shape)/gr.grid_l.A)#[extend:-extend,extend:-extend]
_mlat = gr.grid_l.lat#[extend:-extend,extend:-extend]
_mlon = gr.grid_l.lon#[extend:-extend,extend:-extend]
pax2 = helpers.make_pax(axs[0], dp)
pax1 = helpers.make_pax(axs[1], dp)
glat = gr.grid_l.lat
glon = gr.grid_l.lon
x_, y_ = pax1._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = False)
iii = ~np.isfinite(x_)
x_, y_ = pax1._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = True)
fac[iii] = np.nan # filter facs where coordinates are not defined
pax1.ax.pcolormesh(x_, y_, fac, cmap = plt.cm.bwr, vmin=-clim, vmax=clim)
if ENU:
    pax1.ax.set_title('SECS fit @ %3i km to height integrated $J_{hor}$' % maph)
else:
    pax1.ax.set_title('SECS amplitudes from fit of $J_{\perp}$ [A/m] @ %3i km' % maph)
# Plot line along the lat-cut to use later
glat, glon = dp.mag2geo(np.linspace(50,90,50), np.ones(50)*profile_mlon)
pax1.plot(glat, glon/15, color='black')
helpers.show_grids(sim.xg, gr.grid_l, pax1, csgrid=False)
# GEMINI output from specific height
_data = data.data(gr, sim, beams=False, points=True, lat_ev=gr.grid_l.lat.flatten(), 
                  lon_ev=gr.grid_l.lon.flatten(), alt_ev=np.ones(gr.grid_l.lon.size)*180, 
                  e3doubt_=False)
datadict = _data.__dict__
fac = _data.fac.reshape(gr.grid_l.shape)
glat = gr.grid_l.lat
glon = gr.grid_l.lon
x_, y_ = pax2._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = False)
iii = ~np.isfinite(x_)
x_, y_ = pax2._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = True)
fac[iii] = np.nan # filter facs where coordinates are not defined
pax2.ax.pcolormesh(x_, y_, fac, cmap = plt.cm.bwr, vmin=-clim, vmax=clim)
pax2.ax.set_title('FAC from GEMINI @ 180 km')
pot = datadict['Phitop'].reshape(gr.grid_l.shape)
helpers.show_grids(sim.xg, gr.grid_l, pax2)
plt.tight_layout()
fig.savefig('./plots/secs_fit_facs.pdf')


##################################################################
# Compare GEMINI FAC to the SECS estimates of FACs along meridian
plt.figure()
centerlon = 105 # the longitudinal cenrte (in degrees) of SCW structure
width = 90 # longitudinal width in degrees of SCW feature
scaling = 16 # increase the resulting FAC magnitudes, since the fitted values are too small (AMPERE does not capture small scale stuff)
duration = 20
sigmat = 5
_times = np.ones(1)*10 #temporal locations to evaluare for FAC [minutes]
# _times = np.arange(0,200,100) #temporal locations to evaluare for FAC
_mlats = np.linspace(50, 85, 2500) # mlats to evaluate
_mlons = np.linspace(centerlon-width*0.5, centerlon+width*0.5, 10) # mlons to evaluate
shape = (_times.size, _mlats.size, _mlons.size)
times, mlats, mlons = np.meshgrid(_times, _mlats, _mlons, indexing='ij') # make 3D grid of locations
fac = fac_input_to_matt.fac_input(times, mlons, mlats, centerlon=centerlon, width=width, scaling=scaling, 
                                  duration=duration, sigmat=sigmat)
_glon, _glat = geomag2geog(np.radians(mlons), np.radians(90-mlats)) #returns in degrees
_data = data.data(gr, sim, beams=False, points=True, lat_ev=_glat.flatten(), 
                  lon_ev=_glon.flatten(), alt_ev=np.ones(_glon.flatten().size)*180, 
                  e3doubt_=False)
datadict = _data.__dict__
lonindex = np.argmin(np.abs(_mlons - profile_mlon))
# Map FAC pattern down to maph before plotting. Map from observed location (2) 
# to the maph height (1) using dipole formula
r_2 = datadict['alt'] + RE
r_1 = np.ones(r_2.size)*(maph + RE)
colat_2 = np.radians(90 - mlats.flatten())
colat_1 = np.arcsin(np.sin(colat_2) * np.sqrt(r_1/r_2))
mlats_1 = np.degrees(np.pi/2 - colat_1).reshape(mlons.shape)
plt.plot(mlats_1[0,:,lonindex],datadict['fac'].reshape(shape)[0,:,lonindex], label='GEMINI')
# plt.plot(mlats[0,:,lonindex],fac.reshape(shape)[0,:,lonindex], label='Analytic expression')
# SECS facs
phi, theta = geog2geomag(gr.grid_l.lon.flatten(), gr.grid_l.lat.flatten())
points = np.vstack((np.degrees(phi), 90-np.degrees(theta))).T
_mcf = griddata(points, m_cf, (mlons[0,:,:], mlats[0,:,:]), method='linear').flatten()
_A = griddata(points, gr.grid_l.A.flatten(), (mlons[0,:,:], mlats[0,:,:]), method='linear').flatten()
_fac = _mcf/_A
plt.plot(mlats[0,:,lonindex],_fac.reshape(shape)[0,:,lonindex], label='SECS fit')
plt.legend()
plt.xlabel('mlat')
plt.ylabel('FAC [A/m2]')
plt.xlim(64,78)
plt.ylim(-4e-6,1e-6)
plt.title('FACs along mlon = %3i' % _mlons[lonindex])
plt.savefig('./plots/FAC_profile_fit_performance.pdf')

##################################################################
# Magnetic field on ground from SECS fit
# Get magnetic field estimate at SECS node locations. Can not use GEMINI since the SECS grid
# edges may extend beyond the simulation. Use Dipole module. 
cdlat, cdlon = dp.geo2mag(gr.grid_l.lat.flatten(), gr.grid_l.lon.flatten())
Bn_cd, Bu = dp.B(cdlat, np.ones(gr.grid_l.lon.size)*maph+RE)
Be_cd = np.zeros(gr.grid_l.lon.size)
gclat, gclon, Be, Bn = dp.mag2geo(cdlat, cdlon, Ae = Be_cd, An = Bn_cd)
#_data = data.data(gr, sim, beams=False, points=True, lat_ev=gr.grid_l.lat.flatten(), 
#                  lon_ev=gr.grid_l.lon.flatten(), alt_ev=np.ones(gr.grid_l.lon.size)*maph, 
#                  e3doubt_=False)
Ge_cf, Gn_cf, Gu_cf = secsy.get_CF_SECS_B_G_matrices_for_inclined_field(gr.grid.lat_mesh.flatten(), 
            gr.grid.lon_mesh.flatten(), np.ones(gr.grid.lon_mesh.flatten().size)*RE*1000, 
            gr.grid_l.lat.flatten(), gr.grid_l.lon.flatten(), Be, Bn, Bu, 
            RI = RE * 1e3 + maph * 1e3)
Ge_, Gn_, Gu_ = secsy.get_SECS_B_G_matrices(gr.grid.lat_mesh.flatten(),
            gr.grid.lon_mesh.flatten(),
            np.ones(gr.grid.lon_mesh.flatten().size)*RE*1000,  
            gr.grid_l.lat.flatten(), gr.grid_l.lon.flatten(), 
            constant = 1./(4.*np.pi), RI=RE * 1e3 + maph * 1e3, 
            current_type = 'divergence_free', singularity_limit=gr.grid.Lres*0.5)
sh = gr.grid.lat_mesh.shape
Be = Ge_.dot(m_df).reshape(sh)
Bn = Gn_.dot(m_df).reshape(sh)
Bu = Gu_.dot(m_df).reshape(sh)
Be_cf = Ge_cf.dot(m_cf).reshape(sh)
Bn_cf = Gn_cf.dot(m_cf).reshape(sh)
Bu_cf = Gu_cf.dot(m_cf).reshape(sh)
#Plotting
fig,axs = plt.subplots(1,3,figsize=(9,3))
#Colorbar
cax = fig.add_axes((0.1,0.15,0.8,0.02))
clim = 300
norm = colors.Normalize(vmin=-clim, vmax=clim)
cb1 = colorbar.ColorbarBase(cax, cmap='bwr',
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[nT]')
csax = secsy.CSplot(axs[0],gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, (Be+Be_cf)*1e9, cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Be ground, SECS')
csax = secsy.CSplot(axs[1],gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, (Bn+Bn_cf)*1e9, cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Bn ground, SECS')
csax = secsy.CSplot(axs[2],gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, (Bu+Bu_cf)*1e9, cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Bu ground, SECS')
kk = 4
csax.quiver((Be+Be_cf).flatten()[::kk]*1e9, (Bn+Bn_cf).flatten()[::kk]*1e9, 
            gr.grid.lon_mesh.flatten()[::kk], gr.grid.lat_mesh.flatten()[::kk], label='$B_{hor}$')
csax.ax.legend()
fig.tight_layout()
fig.savefig('./plots/dB_ground_from_SECS_currents.pdf')


##################################################################
# Magnetic field on ground from Biot-Savart
_mlat = magdat['mlat']#[:-4]
_mlon = magdat['mlon']
mlat, mlon = np.meshgrid(_mlat, _mlon, indexing='ij')
# Be_cd = magdat['Bphi'][0,:-4,:]*1e9
# Bn_cd = -magdat['Btheta'][0,:-4,:]*1e9
# Bu = magdat['Br'][0,:-4,:]*1e9
# I have no idea why the following works. Some lats are >90 deg,
# and why this 90 deg rotation? Flipping of phi/theta?
Be_cd = np.rot90(magdat['Bphi'][0,:,:]*1e9,1)
Bn_cd = np.rot90(-magdat['Btheta'][0,:,:]*1e9,1)
Bu = np.rot90(magdat['Br'][0,:,:]*1e9,1)
gclat, gclon, Be, Bn = dp.mag2geo(mlat.flatten(), mlon.flatten(), 
                                  Ae = Be_cd.flatten(), An = Bn_cd.flatten())
# Interpolate onto CS grid
gclon = gclon.flatten()
largelons = gclon>180
gclon[largelons] = gclon[largelons]-360
points = np.vstack((gclon, gclat.flatten())).T
_Be = griddata(points, Be.flatten(), (gr.grid.lon_mesh, gr.grid.lat_mesh), method='linear')
_Bn = griddata(points, Bn.flatten(), (gr.grid.lon_mesh, gr.grid.lat_mesh), method='linear')
_Bu = griddata(points, Bu.flatten(), (gr.grid.lon_mesh, gr.grid.lat_mesh), method='linear')
fig,axs = plt.subplots(1,3,figsize=(9,3))
#Colorbar
cax = fig.add_axes((0.1,0.15,0.8,0.02))
clim = 300
norm = colors.Normalize(vmin=-clim, vmax=clim)
cb1 = colorbar.ColorbarBase(cax, cmap='bwr',
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[nT]')
csax = secsy.CSplot(axs[0],gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, (_Be), cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Be ground, Biot-Savart')
csax = secsy.CSplot(axs[1],gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, (_Bn), cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Bn ground, Biot-Savart')
csax = secsy.CSplot(axs[2],gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, (_Bu), cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Bu ground, Biot-Savart')
kk = 4
csax.quiver((_Be).flatten()[::kk], (_Bn).flatten()[::kk], 
            gr.grid.lon_mesh.flatten()[::kk], gr.grid.lat_mesh.flatten()[::kk], label='$B_{hor}$')
csax.ax.legend()
fig.tight_layout()
fig.savefig('./plots/dB_ground_from_Biot-Savart.pdf')


########################################
# Conductance maps
# As a sanity check, the conductance can be estimated from the height intgrated currents and
# the E-field from GEMINI. This result can be compared with the height integrated conductance
# from GEMINI
Jcf_e, Jcf_n, Jdf_e, Jdf_n, lon, lat = helpers.evalJ(gr, m_cf, m_df, maph=maph)
#Now make a new data object that contain the E-field from GEMINI at the same locations
_data = data.data(gr, sim, beams=False, points=True, lat_ev=lat, 
                  lon_ev=lon, alt_ev=np.ones(lon.size)*maph, 
                  e3doubt_=False)
Emag = np.sqrt(_data.Ee**2 + _data.En**2)
Je = Jcf_e + Jdf_e
Jn = Jcf_n + Jdf_n
SigmaH = (Je*_data.En-Jn*_data.Ee)/Emag**2
SigmaP = (Je*_data.Ee + Jn*_data.En)/Emag**2
fig = plt.figure(figsize=(10,4))
cax = fig.add_axes((0.45,0.1,0.4,0.01))
clim = 25
norm = colors.Normalize(vmin=0, vmax=clim)
cmap='viridis'
cb1 = colorbar.ColorbarBase(cax, cmap=cmap,
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[$S$]')
ax = fig.add_subplot(231)
csax = secsy.CSplot(ax,gr.grid,gridtype='geo')
csax.contour(lon.reshape(gr.grid.lon_mesh.shape), 
             lat.reshape(gr.grid.lon_mesh.shape), 
             _data.Phitop.reshape(gr.grid.lon_mesh.shape), colors='black')
kk=5
csax.quiver(_data.Ee[::kk], _data.En[::kk], lon[::kk], lat[::kk], label='E')
csax.ax.set_title('E and $\Phi$')
csax.ax.legend()
ax = fig.add_subplot(232)
csax = secsy.CSplot(ax,gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, SigmaH.reshape(gr.grid.lon_mesh.shape), 
                norm=norm, cmap=cmap)
csax.ax.set_title('$\Sigma_H=\dfrac{\\hat{r} \cdot (\\vec{J_h} \\times \\vec{E_h})}{E_h^2}$', fontsize=10)
ax = fig.add_subplot(233)
csax = secsy.CSplot(ax,gr.grid,gridtype='geo')
csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, SigmaP.reshape(gr.grid.lon_mesh.shape), 
                norm=norm, cmap=cmap)
csax.ax.set_title('$\Sigma_P = \dfrac{\\vec{J_h} \cdot \\vec{E_h}}{E_h^2}$', fontsize=10)
if interpolate:
    Je, Jn, SH, SP, glons, glats = helpers.height_integtated_current(sim, gr, ENU=ENU, 
                                    maph=maph, interpolate=interpolate, conductance=True)
    ax = fig.add_subplot(235)
    csax = secsy.CSplot(ax,gr.grid,gridtype='geo')
    csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, SH.reshape(gr.grid.lon_mesh.shape), 
                    norm=norm, cmap=cmap)
    csax.ax.set_title('$\Sigma_H$ from GEMINI', fontsize=10)
    ax = fig.add_subplot(236)
    csax = secsy.CSplot(ax,gr.grid,gridtype='geo')
    csax.pcolormesh(gr.grid.lon_mesh, gr.grid.lat_mesh, SP.reshape(gr.grid.lon_mesh.shape), 
                    norm=norm, cmap=cmap)
    csax.ax.set_title('$\Sigma_P$ from GEMINI', fontsize=10)
fig.savefig('./plots/conductance.pdf')



###################################################################
'''
# Scatterplot of data fit
#Input data
d = np.hstack((Je,Jn))
dataN = d.size//2
data_e = d[0:dataN]
data_n = d[dataN:2*dataN]

#Model at input locations
# phi, theta = geog2geomag(lons, lats) # degrees input, radians out 
# mlons = np.degrees(phi)
# mlats = 90 - np.degrees(theta)
use = grid.ingrid(mlons, mlats, ext_factor=-extend)
Ge_cf, Gn_cf = secsy.get_SECS_J_G_matrices(mlats[use], mlons[use],
            grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
            RI=RE * 1e3 + maph * 1e3, current_type = 'curl_free', 
            singularity_limit=grid.Lres*0.5)
G_pred = np.vstack((Ge_cf, Gn_cf))
Jcf = G_pred.dot(m_cf)
Ge_df, Gn_df = secsy.get_SECS_J_G_matrices(mlats[use], mlons[use],
            grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
            RI=RE * 1e3 + maph * 1e3, current_type = 'divergence_free', 
            singularity_limit=grid.Lres*0.5)
G_pred = np.vstack((Ge_df, Gn_df))
Jdf = G_pred.dot(m_df)
N = Ge_cf.shape[0]
Jdf_e = Jdf[0:N]
Jdf_n = Jdf[N:2*N]
Jcf_e = Jcf[0:N]
Jcf_n = Jcf[N:2*N]
model_e = Jcf_e + Jdf_e
model_n = Jcf_n + Jdf_n
plt.scatter(data_e[use], model_e)
plt.scatter(data_n[use], model_n)
plt.hist(model_n-data_n[use])
glons, glats = geomag2geog(np.radians(mlons[use]), np.radians(90-mlats[use])) #returns in degrees
fig=plt.figure()
ax = fig.add_subplot(121)
ax.scatter(mlons[use], mlats[use], c=model_e, vmin=-0.1, vmax=0.1, cmap='bwr')
ax = fig.add_subplot(122)
ax.scatter(mlons[use], mlats[use], c=data_e[use], vmin=-0.1, vmax=0.1, cmap='bwr')
'''