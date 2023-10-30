#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 07:42:30 2023

@author: jone

This script uses the analytically defined input FAC functions to see what kind of 
resolution is needed from GEMINI to reconstruct the analytical FAC spectrum.

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
from pysymmetry.utils.spherical import sph_to_car, car_to_sph
from secsy import cubedsphere
from gemini3d.grid.convert import geomag2geog, geog2geomag
import gemini3d.read as read
import xarray as xr
import time
import secsy
import helpers
from matplotlib import colors, colorbar
from scipy.interpolate import griddata
import fac_input_analytic
import scipy.signal

#Global variables
RE = 6371.2 #Earth radius in km
maph = 110 # Height in km of SECS representation of height integrated currents


# Load GEMINI grid and data
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/aurora_EISCAT3D/"
xg = read.grid(path)


#Define CS grid to work with for SECS representation
extend = 1 # SECS padding "frames"
grid, grid_ev = secs3d.gemini_tools.make_csgrid(xg, height=maph, crop_factor=0.9, #0.2
                        resolution_factor=0.3, extend=extend, extend_ew=2, #0.25
                        dlat = 2., dlon=7, dipole_lompe=True, asymres=1.5)

# Evaluate the analytic FAC function at the grid
centerlon = 105 # the longitudinal cenrte (in degrees) of SCW structure
width = 90 # longitudinal width in degrees of SCW feature
scaling = 10 # increase the resulting FAC magnitudes, since the fitted values are too small (AMPERE does not capture small scale stuff)
duration = 200 # duration of time to model, in minutes
sigmat = 20 # Sigma of the Gaussian temporal modulation of the pattern [minutes]
# Make evaluation locations
_times = np.ones(grid.shape)*100 #temporal locations to evaluare for FAC [minuted]
_mlats = grid.lat # mlats to evaluate [degrees]
_mlons = grid.lon # mlons to evaluate [degrees]
fac = fac_input_analytic.fac_input(_times, _mlons, _mlats, centerlon=centerlon, width=width, scaling=10) # [A/m2]

# Make evaluation locations
_times = np.arange(0,200,10) #temporal locations to evaluare for FAC
_mlats = np.linspace(50, 85, 1000) # mlats to evaluate
_mlons = np.linspace(centerlon-width*0.5, centerlon+width*0.5, 150) # mlons to evaluate
shape = (_times.size, _mlats.size, _mlons.size)
times, mlats, mlons = np.meshgrid(_times, _mlats, _mlons, indexing='ij') # make 3D grid of locations
fac = fac_input_analytic.fac_input(times, mlons, mlats, centerlon=centerlon, width=width, scaling=10)




def freq_calc(nlat):
    _times = np.arange(0,200,100) #temporal locations to evaluare for FAC
    _mlats = np.linspace(50, 85, nlat) # mlats to evaluate
    _mlons = np.linspace(centerlon-width*0.5, centerlon+width*0.5, 10) # mlons to evaluate
    shape = (_times.size, _mlats.size, _mlons.size)
    times, mlats, mlons = np.meshgrid(_times, _mlats, _mlons, indexing='ij') # make 3D grid of locations
    fac = fac_input_analytic.fac_input(times, mlons, mlats, centerlon=centerlon, width=width, scaling=10)
    ff = plt.figure()
    psd, _f = plt.psd(fac[1,:,3],scale_by_freq=False, NFFT=512)
    L = 7628.888 #km
        
    latres = np.diff(_mlats)[0]*111
    Lnyq = 2*latres #wave length of nyquist frequency
    Knyq = 2*np.pi/Lnyq # wave number of nyquist frequency
    real_f = _f*Knyq
    real_L = 2*np.pi/real_f
    
    return (real_L, psd, latres)

K_ns = 2*np.pi/L_ns # wave numbers of the wave components
L_ns = L/np.arange(0,N+1) # wave lengths of the wave components
psd_ns = amp_coef**2

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L_ns, np.log10(psd_ns), label='Fourier series')
nlat = 100
real_L, psd, latres = freq_calc(nlat)
ax.plot(real_L, np.log10(psd), label='latres = %4.2f km,' % latres)
nlat = 500
real_L, psd, latres = freq_calc(nlat)
ax.plot(real_L, np.log10(psd), label='latres = %4.2f km,' % latres)
nlat = 1000
real_L, psd, latres = freq_calc(nlat)
ax.plot(real_L, np.log10(psd), label='latres = %4.2f km,' % latres)
nlat = 2000
real_L, psd, latres = freq_calc(nlat)
ax.plot(real_L, np.log10(psd), label='latres = %4.2f km,' % latres)
ax.set_xlim(0,2000)
ax.set_xlabel('Scale size [km]')
ax.set_ylabel('log power')
ax.legend()

(f, S) = scipy.signal.periodogram(fac[1,:,3], latres, scaling='density')
plt.semilogy(f, S)
# Fit the height integrated current with CF + DF SECS
# Use magnetic dipole coordinates 
m_cf, m_df = helpers.fit_J(grid, Je, Jn, mlons, mlats, l1=1e-3)


#Evaluate CF SECS representation.
Jcf_e, Jcf_n, Jdf_e, Jdf_n, mlon_eval, mlat_eval = helpers.evalJ(grid, m_cf, m_df, maph=maph, extend=extend)

########################################