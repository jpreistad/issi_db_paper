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
sys.path.append('/Users/jone/BCSS-DAG Dropbox/Jone Reistad')
import numpy as np
import git.secs_3d as secs3d
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import apexpy
import matplotlib
from secsy import cubedsphere
from gemini3d.grid.convert import geomag2geog, geog2geomag
import gemini3d.read as read
import xarray as xr
import time
import secsy
import helpers
from matplotlib import colors, colorbar
from scipy.interpolate import griddata
import fac_input_to_matt
import xarray as xr
import polplot # https://github.com/klaundal/polplot
import dipole # https://github.com/klaundal/dipole
import gemini3d.magtools


#Global variables
RE = 6371.2 #Earth radius in km
maph = 110 # Height in km of SECS representation of height integrated currents
ENU = False # Do heith integration in vertical direction, in contrast to field aligned
laplacian = False
extend = 4 # SECS padding "frames"

# Load GEMINI grid and data
# path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/aurora_EISCAT3D/"
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/tmpfiles/aurora_EISCAT3D/"
magdat=gemini3d.magtools.magframe(path+"magfields/20160303_15310.000000.h5")

# xg, dat = secs3d.gemini_tools.read_gemini(path, timeindex=-1, estimate_E_field=True,
                                    # maph=maph, dipolelib=True)
# dat.to_netcdf(path+'temp_dat.nc')
dat = xr.open_dataset(path+'temp_dat.nc')
cfg = read.config(path)
xg = read.grid(path)

#Define CS grid to work with for SECS representation
grid, _ = secs3d.gemini_tools.make_csgrid(xg, height=maph, crop_factor=0.67, #0.2
                        resolution_factor=0.3, extend=extend, extend_ew=3.4, #0.25
                        dlat = 1.5, dlon=-3, dipole_lompe=True, asymres=1.5)

# Get height integrated currents
Je, Jn, mlons, mlats = helpers.height_integtated_current(xg, dat, grid, ENU=ENU, maph=maph, interpolate=True)
# plt.scatter(mlons, mlats, c=Je, vmin=-0.1, vmax=0.1, cmap='bwr')
# plt.xlim(43,153)
# plt.ylim(60,85)


# Fit the height integrated current with CF + DF SECS
# Use magnetic dipole coordinates 
singularity = 0.5
l1 = 1e-1
lcurve = False
m = helpers.fit_J(grid, Je, Jn, mlons, mlats, l1=l1, singularity=singularity, 
                       laplacian=laplacian, extend=extend, lcurve = lcurve)

if lcurve:
    resnorm = m[0]
    modelnorm = m[1]
    ls = m[2]
else:
    m_cf = m[0]
    m_df = m[1]
if laplacian:
    m_l = m[2]

#Evaluate CF SECS representation.
Jcf_e, Jcf_n, Jdf_e, Jdf_n, mlon_eval, mlat_eval = helpers.evalJ(grid, m_cf, m_df, 
                    maph=maph, extend=extend, singularity=singularity)    
########################################

########################################
# Plotting of SECS representation
fig,axs = plt.subplots(1,2,figsize=(10,5))
left_lt = 20
right_lt = 4
minlat = 55

#Colorbar
cax = fig.add_axes((0.1,0.15,0.8,0.02))
clim= 3e-7 #A/m2
norm = colors.Normalize(vmin=-clim, vmax=clim)
cb1 = colorbar.ColorbarBase(cax, cmap='bwr',
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[$A/m^2$]')

# SECS fit
dp = dipole.Dipole(dipole_pole=((90-11),289))
fac = (m_cf.reshape(grid.lon.shape)/grid.A)#[extend:-extend,extend:-extend]
_mlat = grid.lat#[extend:-extend,extend:-extend]
_mlon = grid.lon#[extend:-extend,extend:-extend]
pax2 = polplot.Polarplot(axs[0], minlat = minlat, sector = str(left_lt) + '-' + str(right_lt))
pax1 = polplot.Polarplot(axs[1], minlat = minlat, sector = str(left_lt) + '-' + str(right_lt))
for pax in [pax1,pax2]:
    pax.coastlines(linewidth = .4, color = 'grey')
    # plot dipole latitude circles
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
    pax.plot( np.full(100, minlat), np.linspace(left_lt, 24 + right_lt, 100), color = 'black')
# csax = secsy.CSplot(axs[0],grid,gridtype='geo')
glat, glon = dp.mag2geo(grid.lat, grid.lon)
x_, y_ = pax1._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = False)
iii = ~np.isfinite(x_)
x_, y_ = pax1._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = True)
fac[iii] = np.nan # filter facs where coordinates are not defined
pax1.ax.pcolormesh(x_, y_, fac, cmap = plt.cm.bwr, vmin=-clim, vmax=clim)
if ENU:
    pax1.ax.set_title('SECS fit @ %3i km to height integrated $J_{hor}$' % maph)
else:
    pax1.ax.set_title('SECS fit @ %3i km to height integrated $J_{\perp}$' % maph)
# csax.contour(_mlon, _mlat, _mlat, colors='black')
kk = 8
glat, glon, Jdf_e, Jdf_n = dp.mag2geo(mlat_eval, mlon_eval, Jdf_e, Jdf_n)
glat, glon, Jcf_e, Jcf_n = dp.mag2geo(mlat_eval, mlon_eval, Jcf_e, Jcf_n)
# pax1.quiver(glat[::kk], glon[::kk]/15, Jdf_n[::kk], Jdf_e[::kk], color='blue', label='DF current', alpha=0.5)
# pax1.quiver(glat[::kk], glon[::kk]/15, Jcf_n[::kk], Jcf_e[::kk], color='red', label='CF current', alpha=0.5)
# pax1.ax.legend()
glat, glon = dp.mag2geo(np.linspace(50,90,50), np.ones(50)*90)
pax1.plot(glat, glon/15, color='black')
helpers.show_grids(xg, grid, pax1, csgrid=False)

# Look at GEMINI output from specific height
_glon, _glat = geomag2geog(np.radians(grid.lon), np.radians(90-grid.lat)) #returns in degrees
datadict = secs3d.gemini_tools.sample_points(xg, dat, _glat.flatten(), _glon.flatten(), np.ones(_glon.size)*180)
br, btheta, bphi = secs3d.secs3d.make_b_unitvectors(datadict['Bu'], 
                -datadict['Bn'], datadict['Be'])
fac = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                np.array([br, btheta, bphi]), axis=0).reshape(grid.shape)
glat, glon = dp.mag2geo(grid.lat, grid.lon)
x_, y_ = pax2._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = False)
iii = ~np.isfinite(x_)
x_, y_ = pax2._latlt2xy(glat.reshape(fac.shape), glon.reshape(fac.shape)/15, ignore_plot_limits = True)
fac[iii] = np.nan # filter facs where coordinates are not defined
pax2.ax.pcolormesh(x_, y_, fac, cmap = plt.cm.bwr, vmin=-clim, vmax=clim)
pax2.ax.set_title('FAC from GEMINI at 180 km')
pot = datadict['Phitop'].reshape(grid.shape)
helpers.show_grids(xg, grid, pax2)
plt.tight_layout()
fig.savefig('secs_fit_currents_new.pdf')
# csax.contour(_mlon, _mlat, pot, colors='black')
# csax.quiver(Jdf_e[::kk]+Jcf_e[::kk], Jdf_n[::kk]+Jcf_n[::kk], mlon_eval[::kk], mlat_eval[::kk], label='Total current (SECS)')
# csax.ax.legend()

############
# Plot comparing height integrated currents from GEMINI with the fitted SECS current

########################################
# Magnetic field on ground
theta = np.radians(90 - grid.lat)
Bn, Bu = secs3d.gemini_tools.dipole_B(theta, height = maph)
Be = np.zeros((Bn.shape))
Ge_cf, Gn_cf, Gu_cf = secsy.get_CF_SECS_B_G_matrices_for_inclined_field(grid.lat_mesh.flatten(), 
            grid.lon_mesh.flatten(), np.ones(grid.lon_mesh.flatten().size)*RE*1000, 
            grid.lat.flatten(), grid.lon.flatten(), Be.flatten(), Bn.flatten(), 
            Bu.flatten(), RI = RE * 1e3 + maph * 1e3)
Ge_, Gn_, Gu_ = secsy.get_SECS_B_G_matrices(grid.lat_mesh.flatten(),
            grid.lon_mesh.flatten(),
            np.ones(grid.lon_mesh.flatten().size)*RE*1000,  
            grid.lat.flatten(), grid.lon.flatten(), 
            constant = 1./(4.*np.pi), RI=RE * 1e3 + maph * 1e3, 
            current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)
sh = grid.lat_mesh.shape
Be = Ge_.dot(m_df).reshape(sh)
Bn = Gn_.dot(m_df).reshape(sh)
Bu = Gu_.dot(m_df).reshape(sh)
Be_cf = Ge_cf.dot(m_cf).reshape(sh)
Bn_cf = Gn_cf.dot(m_cf).reshape(sh)
Bu_cf = Gu_cf.dot(m_cf).reshape(sh)
# datadict = {'mlat':grid.lat_mesh, 'mlon':grid.lon_mesh, 'Be_df':Be, 'Bn_df':Bn, 'Bu_df':Bu}
# np.save('dB_paper_plotting.npy', datadict)
# dd = np.load('dB_paper_plotting.npy',allow_pickle=True)
#Plttotting
fig,axs = plt.subplots(1,3,figsize=(8,3.5))
#Colorbar
cax = fig.add_axes((0.1,0.13,0.8,0.03))
clim = 50
norm = colors.Normalize(vmin=-clim, vmax=clim)
cb1 = colorbar.ColorbarBase(cax, cmap='bwr',
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[nT]')
csax = secsy.CSplot(axs[0],grid,gridtype='geo')
csax.pcolormesh(grid.lon_mesh, grid.lat_mesh, (Be+Be_cf)*1e9, cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Be_df+cf on ground')
csax = secsy.CSplot(axs[1],grid,gridtype='geo')
csax.pcolormesh(grid.lon_mesh, grid.lat_mesh, (Bn+Bn_cf)*1e9, cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Bn_df+cf on ground')
csax = secsy.CSplot(axs[2],grid,gridtype='geo')
csax.pcolormesh(grid.lon_mesh, grid.lat_mesh, (Bu+Bu_cf)*1e9, cmap='bwr', vmin=-clim, vmax=clim)
csax.ax.set_title('Bu_df+cf on ground')
kk = 4
csax.quiver((Be+Be_cf).flatten()[::kk]*1e9, (Bn+Bn_cf).flatten()[::kk]*1e9, 
            grid.lon_mesh.flatten()[::kk], grid.lat_mesh.flatten()[::kk], label='$B_{hor}$')
csax.ax.legend()
fig.tight_layout()
# fig.savefig('df.png')

##################################################################
################
# Compare GEMINI FAC to the synthetic prescribed FACs
plt.figure()
centerlon = 105 # the longitudinal cenrte (in degrees) of SCW structure
width = 90 # longitudinal width in degrees of SCW feature
scaling = 10 # increase the resulting FAC magnitudes, since the fitted values are too small (AMPERE does not capture small scale stuff)
_times = np.ones(1)*66 #temporal locations to evaluare for FAC [minutes]
# _times = np.arange(0,200,100) #temporal locations to evaluare for FAC
_mlats = np.linspace(50, 85, 2500) # mlats to evaluate
_mlons = np.linspace(centerlon-width*0.5, centerlon+width*0.5, 10) # mlons to evaluate
shape = (_times.size, _mlats.size, _mlons.size)
times, mlats, mlons = np.meshgrid(_times, _mlats, _mlons, indexing='ij') # make 3D grid of locations
fac = fac_input_to_matt.fac_input(times, mlons, mlats, centerlon=centerlon, width=width, scaling=10)
_glon, _glat = geomag2geog(np.radians(mlons), np.radians(90-mlats)) #returns in degrees
datadict = secs3d.gemini_tools.sample_points(xg, dat, _glat.flatten(), _glon.flatten(), np.ones(_glon.size)*180)
br, btheta, bphi = secs3d.secs3d.make_b_unitvectors(datadict['Bu'], 
                -datadict['Bn'], datadict['Be'])
datadict['fac'] = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                np.array([br, btheta, bphi]), axis=0)
lonindex = 3
# Map FAC pattern down to maph before plotting. Map from observed location (2) 
# to the maph height (1)
r_2 = datadict['alt'] + RE
r_1 = np.ones(r_2.size)*(maph + RE)
colat_2 = np.radians(90 - mlats.flatten())
colat_1 = np.arcsin(np.sin(colat_2) * np.sqrt(r_1/r_2))
mlats_1 = np.degrees(np.pi/2 - colat_1).reshape(mlons.shape)
plt.plot(mlats_1[0,:,lonindex],datadict['J1'].reshape(shape)[0,:,lonindex], label='GEMINI')
# plt.plot(mlats[0,:,lonindex],fac.reshape(shape)[0,:,lonindex], label='Analytic expression')
# SECS facs
points = np.vstack((grid.lon.flatten(), grid.lat.flatten())).T
_mcf = griddata(points, m_cf, (mlons[0,:,:], mlats[0,:,:]), method='linear').flatten()
_A = griddata(points, grid.A.flatten(), (mlons[0,:,:], mlats[0,:,:]), method='linear').flatten()
_fac = _mcf/_A
plt.plot(mlats[0,:,lonindex],_fac.reshape(shape)[0,:,lonindex], label='SECS fit')
plt.legend()
plt.xlabel('mlat')
plt.ylabel('FAC [A/m2]')
plt.xlim(64,78)
plt.ylim(-6e-7,2e-7)
plt.title('FACs along mlon = %3i' % _mlons[lonindex])
###################################

########################################
# Conductance maps
_glon, _glat = geomag2geog(np.radians(mlon_eval), np.radians(90-mlat_eval)) #returns in degrees
datadict = secs3d.gemini_tools.sample_points(xg, dat, _glat.flatten(), _glon.flatten(), np.ones(_glon.size)*maph)
pot = datadict['Phitop']
#interpolate nan values using scipy griddata
real = np.isfinite(pot)
phi, theta = geog2geomag(datadict['lon'], datadict['lat']) # degrees input, radians out 
mlons = np.degrees(phi)
mlats = 90 - np.degrees(theta)
points = np.vstack((mlons[real], mlats[real])).T
pot2d = griddata(points, pot[real], (grid.lon, grid.lat), method='nearest')
De2, Dn2 = grid.get_Le_Ln()
Ee = -De2.dot(pot2d.flatten())
En = -Dn2.dot(pot2d.flatten())
Emag2 = Ee**2 + En**2
use = grid.ingrid(grid.lon_mesh.flatten(), grid.lat_mesh.flatten(), ext_factor=-extend)
points = np.vstack((grid.lon_mesh.flatten()[use],grid.lat_mesh.flatten()[use])).T
points = np.vstack((mlon_eval[real], mlat_eval[real])).T
Je = griddata(points, Jcf_e[real]+Jdf_e[real], (grid.lon, grid.lat), method='nearest').flatten()
Jn = griddata(points, Jcf_n[real]+Jdf_n[real], (grid.lon, grid.lat), method='nearest').flatten()
SigmaH = (Je*En-Jn*Ee)/Emag2
SigmaP = (Je*Ee + Jn*En)/Emag2
fig = plt.figure(figsize=(10,4))
cax = fig.add_axes((0.45,0.2,0.4,0.02))
clim = 10
norm = colors.Normalize(vmin=-clim, vmax=clim)
cb1 = colorbar.ColorbarBase(cax, cmap='bwr',
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('[$S$]')
ax = fig.add_subplot(131)
csax = secsy.CSplot(ax,grid,gridtype='geo')
csax.contour(_mlon, _mlat, pot2d, colors='black')
csax.quiver(Ee, En, grid.lon.flatten(), grid.lat.flatten(), label='E')
csax.ax.set_title('E and $\Phi$')
csax.ax.legend()
ax = fig.add_subplot(132)
csax = secsy.CSplot(ax,grid,gridtype='geo')
csax.pcolormesh(grid.lon_mesh, grid.lat_mesh, SigmaH.reshape(grid.shape), vmin=-10, vmax=10, cmap='bwr')
csax.ax.set_title('$\Sigma_H=\dfrac{\\hat{r} \cdot (\\vec{J_h} \\times \\vec{E_h})}{E_h^2}$', fontsize=10)
ax = fig.add_subplot(133)
csax = secsy.CSplot(ax,grid,gridtype='geo')
csax.pcolormesh(grid.lon_mesh, grid.lat_mesh, SigmaP.reshape(grid.shape), vmin=-10, vmax=10, cmap='bwr')
csax.ax.set_title('$\Sigma_P = \dfrac{\\vec{J_h} \cdot \\vec{E_h})}{E_h^2}$', fontsize=10)

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
