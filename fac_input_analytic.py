#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:35:03 2023

@author: jone
Provide a spatial and temporal pattern of FAC to be input to GEMINI run for 
review paper on magnetic disturbances, as part off Kalles ISSI group. 

Output are analytical expressions that given (mlat, mlon, time) will output the
FAC density to drive the GEMINI model with.

"""
import numpy as np
import matplotlib.pyplot as plt
import polplot
import xarray as xr
import pandas as pd
from lmfit import Minimizer, Parameters, create_params, report_fit
from scipy import interpolate
import pandas as pd

def fcn2min(params, x, data):
    '''
    Cost function used for fitting the paramerters in the fourier series, to be
    passed to lmfit.

    '''
    model = np.zeros(x.size) + params['AB0'] # n = 0 term
    for n in np.arange(1,N+1):
        # print(n)
        key = list(params.keys())[n+1]
        model =+ model + params['a']*S[n-1]*(np.cos(params[key])*np.cos(2*np.pi*n*x/L) + 
                                    np.sin(params[key]) * np.sin(2*np.pi*n*x/L))
        # model =+ model + params['a']*S[n-1]*(np.cos(2*np.pi*n*x/L) + 
        #                             np.sin(2*np.pi*n*x/L))                                    
    return model - data

def fit_latitudinal_profile(N=120):
    '''
    Function that will do a fit to Simons SCW event as envelope, and impose structure
    mimicing the Gjerloev et al 2011 paper, figure 10, nightside disturbed:
    https://angeo.copernicus.org/articles/29/1713/2011/angeo-29-1713-2011.pdf

    N : int
        number of waves in the fourier expansion

    Returns
    -------
    None.

    '''
    
    # Open the SCW data from Simon
    scw = xr.open_dataset('Amplitudes.ncdf', engine='netcdf4')
    dat = scw.isel(time=15)
    amp = dat.Amplitude_cf.values
    mlat = dat.mlat.values
    mlt = dat.mlt.values
    area = dat.Area.values*1e6 # in m2
    fac = amp/area
    plt.pcolormesh(fac, vmin=-1e-6, vmax=1e-6, cmap='seismic')


    # Latitudinal profile
    use = (mlt>20.75) & (mlt<21.25)
    _mlt = mlt.copy()
    _mlt[use] = 0
    mlats = mlat[use]
    facs = fac[use]
    sortind = np.argsort(mlats)
    mlats = mlats[sortind]
    facs = facs[sortind]
    df = pd.DataFrame({'mlat':mlats, 'fac':facs})
    bins = np.arange(50,86,1)
    gglat = df.groupby(pd.cut(df.mlat, bins=bins)).mean()
    plt.plot(gglat.mlat,gglat.fac)
    # Use a gaussian window to make it go to 0 at the edges
    _g =np.exp(-(gglat.mlat.values-gglat.mlat.median())**2/(2*7**2))
    gglat.fac = gglat.fac*_g

    ###################################################################
    # Add realistic spatial structure with the latitudinal profile
    ##################################################################
    
    # Define a power spectrum based on the Gjerloev 2011 paper, their Figure 10, nightside disturbed
    # we modify their amplitudes by multiplying by their scale size, since their power spectrum
    # is so flat it that it will not be able to fit the form we want (the SCW above). Note that
    # this spectrum reflect the magnetic perturbations from FACs, and not the FACs themselves.
    scale_size = np.flip(np.array([1e-3, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]))[:-1] # km
    amplitude  = np.flip(np.array([2, 12, 19, 24, 29, 33, 37, 41, 45, 48, 51, 54, 57, 60, 62, 64, 67, 70, 73]))[:-1] * scale_size # nT
    
    # amplitude vs scale_size is quite linear on log-log scale so
    # we fit a line to get a general relationship: 
    # plt.plot(np.log(scale_size), np.log(amplitude))
    p = np.polyfit(np.log(scale_size), np.log(amplitude), 1)  
    
    # amplitude as function of arbitrary scale size:
    amplitude_of_scalesize = lambda scalesize: np.exp(p[1])*scalesize**p[0]    
    
    # The scale sizes used in the Fourier series representation
    L = (gglat.mlat.max()-gglat.mlat.min())*111*2 # approx distance in km of the lat range, *2, since this will make the n=1 wave make a half wavelength in the chosen latitude range (mlat.max()-mlat.min()), giving more flexibility to the Fourier expansion. 
    scalesizes = L/np.arange(1,N+1)
    
    # The amplitudes of the scale sizes in the Fourier representation    
    S = amplitude_of_scalesize(scalesizes)
    
    # define objective function: returns the array to be minimized
    # The DeltaB profile y(x) will be expressed as follows, where x is distance
    # along the profile, L is the wavelength corresponding to the n=1 frequency, choosen 
    # such that a half wave will exactly fit in the latitudinal range choosen.
    
    # The power spectral shape is enforced by only fitting the phase and a common scaling
    #  amplitude of the
    # fourier components, keeping the relative amplitude between components the same as
    # from the Gjerloev paper (with modification). The amplitude of each component, 
    # c_n, is related to the
    # prescribed spectra through c_n = a * S_n, where a the is the constant to fit (independent of n)
    # and S_n are the Gjerloev amplitudes* interpolated to the scale size in question (n). 
    # The curve to fit (the SCW) is then represented as
    # y(x) = AB0 + a * sum(cos(d_n)*cos(2*pi*n*x/L) + sin(d_n)*sin(2*pi*n*x/L))
    # where d_n are the phase coeficients to be determined, and AB0 is the constant
    # from n=0, also to be fitted.
    
    
    # create a set of Parameters
    params = Parameters()
    params.add('AB0', value=0) # the constant from n=0
    params.add('a', value=0) # the scaling of the spectra
    for n in np.arange(1,N+1):
        # params.add('A'+str(n), value=m[n])
        # params.add('B'+str(n), value=m[n+N])
        params.add('d'+str(n), value=0, min=-np.pi, max=np.pi) 
    
    
    # Define the curve to fit on the x-space that the Fourier representation is formulated on
    xres = 1000 # resolution in the latitudinal direction. 
    x = np.linspace(-L/4, L/4, xres)
    _lats = (x/111) + gglat.mlat.median()
    f = interpolate.interp1d(gglat.mlat.values, -gglat.fac.values, fill_value='extrapolate')
    mu0 = 4*np.pi*10**(-7)
    dx = 1000*L*0.5/xres # in meters
    data = mu0*np.cumsum(f(_lats)*dx) # Cumsum is how we go from FAC to deltaB, which is what we will fit
    
    # do fit, here with the default leastsq algorithm
    minner = Minimizer(fcn2min, params, fcn_args=(x, data))
    result_p = minner.minimize()
    
    # calculate final result
    final = data + result_p.residual
    
    # Compare to implementation of analytic formula
    amp_coef = np.zeros(N+1)
    phase_coef = np.zeros(N+1)
    for n in np.arange(0,N+1):
        key = list(minner.result.params.keys())[n+1]
        if n == 0:
            amp_coef[n] = minner.result.params['AB0'].value
            phase_coef[n] = 0
        else:
            amp_coef[n] = minner.result.params['a'].value*S[n-1]
            phase_coef[n] = minner.result.params[key].value
    
    # Save coefficients
    df = pd.DataFrame(index=np.arange(start=0, stop=N+1))
    df.loc[:,'amplitude'] = amp_coef 
    df.loc[:,'phase'] = phase_coef
    df.to_hdf('fac_input_coefs.h5', 'data')
    
    
##########################################################3
# The following functions are used to read the fit coefficients and estimate
# the FAC using the analytic expressions at any input time, mlat, mlon location.


def mlat_to_latdist(mlat, mlat0 = 67.5163):
    '''
    Function that return the spatial x-argument to be used in Fourier representation
    of FAC from the fitted coefficients. 

    Parameters
    ----------
    mlat : int/float or array-like
        magnetic latitude in degreed (use centered dipole in GEMINI) to convert
    mlat0 : float, optional
        the mlat location corresponding to x=0. Found using gglat.mlat.median()
        from Simons SCW event.

    Returns
    -------
    x argument corresponding to mlat, in units of km.

    '''
    return (mlat-mlat0)*111  


def lat_fac(mlat, L = 7628.888, scaling = 10):
    '''
    Return the FAC at input mlat location, only from the latitudinal contribution

    Parameters
    ----------
    mlat : int/float or array-like
        input locations in degrees, centered dipole coordinates (magnetic)
    L : float, optional
        The width of the domain used in fitting Fourier series. Must be the same
        as the L used to estimate the coefficients.
    scaling : int/float
        The amplitude coefficients are multiplied by this number to scale the results. 
        The output from the AMPERE inversion from Simon results in very weak currents,
        typically ~0.1 muA/m2, which is unrealistic when going to finer scales. 
        This keyword modifies this.

    Returns
    -------
    FAC [A/m2] from latitude profile only, at input locations.

    '''
    
    if type(mlat) == int or type(mlat) == float:
        mlat = np.array([mlat])
    x = mlat_to_latdist(mlat)
    B_analytical = np.zeros(x.shape)
    current_analytical = np.zeros(x.shape)
    mu0 = 4*np.pi*10**(-7)
    
    # Read coeficient file:
    df = pd.read_hdf('fac_input_coefs.h5')
    N = df.shape[0] - 1
    amp_coef = df.amplitude.values * scaling # 
    phase_coef = df.phase.values
    
    for n in np.arange(0,N+1):
        B_analytical =+ B_analytical + amp_coef[n]*(np.cos(phase_coef[n])* 
                                np.cos(2*np.pi*n*x/L) + np.sin(phase_coef[n]) * 
                                np.sin(2*np.pi*n*x/L))
        current_analytical =+ current_analytical + 2*np.pi*n/(L*1000*mu0) * \
                                amp_coef[n]*(np.sin(phase_coef[n])*np.cos(2*np.pi*n*x/L) - \
                                np.cos(phase_coef[n])*np.sin(2*np.pi*n*x/L))
    g = np.exp(-(mlat-70)**4/(2*45**2)) # Function that will make the ends go to 0 for the current
    current_analytical = current_analytical * g
    
    return current_analytical


def lon_fac(mlon, centerlon = 105, width = 90):
    '''
    Function that return the longitude modulation of the FAC, based on a sine wave
    with a period of width degrees, centered around centerlon.

    Parameters
    ----------
    mlon : int/float or array-like
        magnetic longitude in degrees (use centered dipole in GEMINI) to convert
    centerlon : float, optional
        the mlon location in degrees corresponding to x=0.
    width : the width (in degrees) of a full period of the sine wave in longitude
        modulation

    Returns
    -------
    mlon modulation factor for FAC, in [-1,1].

    '''
    k = 360/width
    sine_part = np.sin(k*np.radians(mlon-centerlon))
    
    #We make the function decay less rapid toward zero by introducing this exp function
    # Note that this make the max amplitude reduce by typically 25%
    exp_part = np.exp(-(mlon-centerlon)**4/(2*690**2))
    
    combined = sine_part * exp_part
    return combined

    # mlon = np.arange(centerlon-45,centerlon+45,1)
# plt.plot(mlon,np.sin(k*np.radians(mlon-centerlon)))    
    # np.sin(k*np.radians(mlon-centerlon))

def temp_fac(t, duration=200, sigmat=20):
    '''
    Compute the temporal part of the FAC value, based on a Gaussian.

    Parameters
    ----------
    t : int/float or array-like
        Time in minutes.
    duration : int/float, optional
        Duration of modulation in units of minutes. The default is 200 min.
    sigmat : int/float, optional
        The sigma of Gaussian modulation, in minutes. The default is 20 min.

    Returns
    -------
    The Gaussian modulation factor for the input time.
    '''
    
    return np.exp(-(t-duration/2)**2/(2*sigmat**2))

def fac_input(t, mlon, mlat, duration=200, sigmat=20, centerlon=105, width=90, 
              L = 7628.888, scaling = 10):
    '''
    Return the FAC input in A/m2 at given time [minutes], mlon, mlat [in degrees]

    Parameters
    ----------
    t : int/float
        time in minutes.
    mlon : int/float or array-like
        input magnetic longitude in degrees.
    mlat : int/float or array-like
        input magnetic latitude in degrees.
    duration : int/float, optional
        Duration of modulation in units of minutes. The default is 200 min.
    sigmat : int/float, optional
        The sigma of Gaussian modulation, in minutes. The default is 20 min.    
    centerlon : float, optional
        the mlon location in degrees corresponding to x=0.
    width : the width (in degrees) of a full period of the sine wave in longitude
        modulation
    L : float, optional
        The width of the domain used in fitting Fourier series. Must be the same
        as the L used to estimate the coefficients.
    scaling : int/float
        The amplitude coefficients are multiplied by this number to scale the results. 
        The output from the AMPERE inversion from Simon results in very weak currents,
        typically ~0.1 muA/m2, which is unrealistic when going to finer scales. 
        This keyword modifies this.
        
    Returns
    -------
    FAC at input time/location in A/m2.
    
    '''

    t_part = temp_fac(t, duration=duration, sigmat=sigmat)
    lon_part = lon_fac(mlon, centerlon=centerlon, width=width)
    lat_part = lat_fac(mlat, L=L, scaling=scaling)
    
    fac = t_part * lon_part * lat_part
    
    return fac



# ##################################
# # Example use
# ##################################

# # Set some parameters
# centerlon = 105 # the longitudinal cenrte (in degrees) of SCW structure
# width = 90 # longitudinal width in degrees of SCW feature
# scaling = 10 # increase the resulting FAC magnitudes, since the fitted values are too small (AMPERE does not capture small scale stuff)

# # Make evaluation locations
# _times = np.arange(0,200,100) #temporal locations to evaluare for FAC
# _mlats = np.linspace(50, 85, 2500) # mlats to evaluate
# _mlons = np.linspace(centerlon-width*0.5, centerlon+width*0.5, 10) # mlons to evaluate
# shape = (_times.size, _mlats.size, _mlons.size)
# times, mlats, mlons = np.meshgrid(_times, _mlats, _mlons, indexing='ij') # make 3D grid of locations
# fac = fac_input(times, mlons, mlats, centerlon=centerlon, width=width, scaling=10)

# # Some plotting
# clim=4e-6 #A/m2
# tind = 10
# plt.figure()
# plt.pcolormesh(mlons[tind,:,:],mlats[tind,:,:],fac[tind,:,:], cmap='bwr', vmin=-clim, vmax=clim)
# plt.xlabel('mlon [deg]')
# plt.ylabel('mlat [deg]')















# ##############################
# # Other diagnostic plotting

# # plot results, how fitted dB curve matches input
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(_lats, data*1e9, '+', label='data')
# plt.plot(_lats, final*1e9, label='lmfit')
# plt.plot(_lats, B_analytical*1e9, label='analytical formula')
# plt.legend()
# plt.xlabel('MLAT')
# plt.ylabel('$\Delta$B (what is fitted) [nT]')

# plt.figure()
# plt.plot(_lats, final-B_analytical)

# # Plot how reconstructed FAC input
# plt.figure()
# cropstart = 50
# cropstop = -35
# _fitcurrent = np.diff(final,append=np.nan)[cropstart:cropstop] / (mu0*dx)
# # _fitcurrent_analytical = np.diff(final_analytical,append=np.nan)[cropstart:cropstop]
# uselats = _lats[cropstart:cropstop]
# # Use a gaussian window to make it go to 0 at the edges
# _g =np.exp(-(uselats-70)**4/(2*45**2))
# fitcurrent = _fitcurrent*_g
# fitcurrent_analytical = current_analytical[cropstart:cropstop]#*_g
# # plt.plot(uselats, _fitcurrent, label='FAC from Prescribed spectrum, no gauss')
# plt.plot(uselats, fitcurrent, label='FAC from Prescribed spectrum')
# plt.plot(uselats, fitcurrent_analytical, label='FAC from Prescribed spectrum_analytical')
# plt.plot(uselats, f(_lats)[cropstart:cropstop], label='FAC from SCW envelope')
# plt.xlabel('MLAT')
# plt.ylabel('FAC [A/m$^2$]')
# plt.legend()