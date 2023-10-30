#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:42:30 2023

@author: jone
"""

import fac_input_to_matt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Global variables
RE = 6371.2 #Earth radius in km
maph = 110 # Height in km of SECS representation of height integrated currents


# Evaluate the analytic FAC function at the grid
centerlon = 105 # the longitudinal cenrte (in degrees) of SCW structure
width = 90 # longitudinal width in degrees of SCW feature
scaling = 10 # increase the resulting FAC magnitudes, since the fitted values are too small (AMPERE does not capture small scale stuff)
duration = 200 # duration of time to model, in minutes
sigmat = 20 # Sigma of the Gaussian temporal modulation of the pattern [minutes]
L = 7628.888
startmlon = centerlon - 0.7*width
stopmlon = centerlon + 0.7*width
startmlat = 60
stopmlat = 85
Nres = 2000

# mlon part
mlon = np.linspace(startmlon, stopmlon, Nres)
lonfac = fac_input_to_matt.lon_fac(mlon, centerlon = centerlon, width = width)
plt.plot(mlon, lonfac)

# mlat part
mlat = np.linspace(startmlat, stopmlat, Nres)
latfac = fac_input_to_matt.lat_fac(mlat, L = L, scaling = scaling)
plt.plot(mlat, latfac)

# time part
time = np.linspace(0, duration, Nres)
timefac = fac_input_to_matt.temp_fac(time, duration=duration, sigmat=sigmat)
plt.plot(time, timefac)

# Full FAC pattern
shape = (Nres,Nres)
_t = 100
_times = np.ones(shape)*_t #temporal locations to evaluare for FAC [minuted]
_mlats, _mlons = np.meshgrid(mlat, mlon, indexing='ij')
fac = fac_input_to_matt.fac_input(_times, _mlons, _mlats, centerlon=centerlon, width=width, scaling=scaling) # [A/m2]
plt.pcolormesh(_mlons, _mlats, fac)

# datadict = {'mlat':_mlats, 'mlon':_mlons, 'fac':fac, '_mlat':mlat, 'mlatfactor':latfac, '_mlon':mlon, 'mlonfactor':lonfac}
# np.save('dB_paper_plotting.npy', datadict)
# dd = np.load('dB_paper_plotting.npy',allow_pickle=True)


# Create a figure with a gridspec layout
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.4])  # 2 columns, 3 rows with different column widths

# Create line plots in the left column
ax = fig.add_subplot(gs[0, 0])  # Left column
ax.plot(mlat, latfac*1e6)
df = pd.DataFrame(columns= ['fac'])
df.fac = latfac*1e6
ax.plot(mlat, df.rolling(300, center=True).fac.mean(), '--')
ax.set_title('Latitude pattern of FAC')
ax.set_ylabel('[$\mu A/m^2$]')
ax.set_xlabel('mlat [$^\circ$]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.85,0.25,'A', transform=ax.transAxes, fontsize=14)
ax = fig.add_subplot(gs[1, 0])  # Left column
ax.plot(mlon, lonfac)
ax.set_title('Longitude modulation of FAC')
ax.set_xlabel('mlon [$^\circ$]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.85,0.25,'B', transform=ax.transAxes, fontsize=14)
ax = fig.add_subplot(gs[2, 0])  # Left column
ax.plot(time, timefac)
ax.set_title('Temporal modulation of FAC')
ax.set_xlabel('Duration [min]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.85,0.25,'C', transform=ax.transAxes, fontsize=14)
# Create a contour plot in the right column spanning all 3 rows
clim = 3
ax = fig.add_subplot(gs[:, 1])  # Right column
ax.set_title('2D FAC pattern at t = %3i min [$\mu A/m^2$]' % _t)
ax.set_xlabel('mlon [$^\circ$]')
ax.set_ylabel('mlat [$^\circ$]')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ccc = ax.pcolormesh(_mlons, _mlats, fac*1e6, cmap='bwr')
cbar = fig.colorbar(ccc, ax=ax, shrink=0.6, aspect=10)  # Add colorbar
# cbar.set_label('[$\mu A/m^2$]', rotation=90, labelpad=20)  # Adjust rotation and padding
ax.text(0.,0.85,'D', transform=ax.transAxes, fontsize=14)

# Adjust the aspect ratio to be slightly wider than it is tall
# ax.set_aspect('equal', adjustable='box')

# Adjust the layout
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Adjust the horizontal space between subplots
fig.savefig('fac_input_figure.png')
# Show the plot
plt.show()
