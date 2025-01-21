import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import harmonica as hm
from sklearn.preprocessing import QuantileTransformer
import gstatsim
import skgstat as skg
import xarray as xr
import xrft
import verde as vd

import warnings
warnings.filterwarnings("ignore")

from prisms import make_prisms
from utilities import xy_into_grid, lowpass_filter_invpad

def boug_interpolation_sgs(ds, grav, density):
    """
    Stochastically interpolate gridded Bouguer disturbance using SGS

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame of gravity data
        density : float of rock or background density
    Outputs:
        Terrain effect for use as target of inversion.
    """
    density_dict = {
        'ice' : 917,
        'water' : 1027,
        'rock' : density
    }
    
    prisms, densities = make_prisms(ds, ds.bed.values, density_dict)
    pred_coords = (grav.x, grav.y, grav.height)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')
    
    residual = grav.faa-g_z
    
    X = np.stack([grav.x[grav.inv_msk==False], grav.y[grav.inv_msk==False]]).T
    y = residual[grav.inv_msk==False]
    XX = np.stack([grav.x, grav.y]).T
    
    df_grid = pd.DataFrame({'X' : X[:,0], 'Y' : X[:,1], 'residual' : y})
    
    # normal score transformation
    data = df_grid['residual'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    df_grid['NormZ'] = nst_trans.transform(data) 

    # compute experimental (isotropic) variogram
    coords = df_grid[['X','Y']].values
    values = df_grid['NormZ']

    maxlag = 100_000             # maximum range distance
    n_lags = 70                # num of bins
    
    V3 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)
    
    V3.model = 'spherical'
    
    # set variogram parameters
    azimuth = 0
    nugget = V3.parameters[2]

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = V3.parameters[0]
    minor_range = V3.parameters[0]
    sill = V3.parameters[1]
    vtype = 'spherical'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 64         # number of neighboring data points used to estimate a given point 
    rad = 100_000    # 100 km search radius

    sim = gstatsim.Interpolation.okrige_sgs(XX, df_grid, 'X', 'Y', 'NormZ', k, vario, rad, quiet=True)
    sim_trans = nst_trans.inverse_transform(sim.reshape(-1,1))

    return grav.faa.values-sim_trans.squeeze()

def filter_boug(ds, grav, target, cutoff=10e3, pad=0):
    """
    Filter Bouguer disturbance with lowpass Gaussian filter
    given a terrain effect simulation.

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame gravity data
        target : terrain effect resulting from Bouguer SGS interpolation
        cutoff : pass frequencies below this
        pad : amount to pad inversion domain for filtering
    Outputs:
        Filtered Bouguer disturbance
    """
    xx, yy = np.meshgrid(ds.x, ds.y)

    target_grid = xy_into_grid(ds, (grav.x.values, grav.y.values), target)
    faa_grid = xy_into_grid(ds, (grav.x, grav.y), grav.faa)
    boug_grid = faa_grid - target_grid
    
    grav_msk = ~np.isnan(boug_grid)

    nearest = vd.KNeighbors(k=10)
    nearest.fit(
        coordinates=(grav.x.values, grav.y.values),
        data = grav.faa-target
    )
    boug_fill = nearest.predict((xx.flatten(), yy.flatten()))
    boug_fill = np.where(grav_msk==True, boug_grid, boug_fill.reshape(xx.shape))
    
    boug_filt = lowpass_filter_invpad(ds, boug_fill, cutoff, pad)
    boug_filt = boug_filt[grav_msk]
    
    return boug_filt

def sgs_filt(ds, grav, density, cutoff, pad=0):
    """
    Performs SGS Bouguer interpolation, filters Bouguer,
    returns new target terrain effect

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame gravity data
        target : terrain effect resulting from Bouguer SGS interpolation
        cutoff : pass frequencies below this
        pad : amount to pad inversion domain for filtering
    Outputs:
        Target terrain effect from filtered Bouguer SGS interpolation
    """
    target = boug_interpolation_sgs(ds, grav, density)
    boug_filt = filter_boug(ds, grav, target, cutoff, pad)
    new_target = grav.faa.values - boug_filt
    
    return new_target