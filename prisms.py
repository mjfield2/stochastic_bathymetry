import numpy as np
import xarray as xr
import boule as bl

def make_prisms(ds, bed, density_dict, msk=None, ice=True):

    ice_dens = density_dict['ice']
    water_dens = density_dict['water']
    rock_dens = density_dict['rock']

    res = ds.res
    half_res = res/2
    
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    
    surf = ds.surface.values
    thick = ds.thickness.values

    # make prisms for entire x/y domain
    if msk is None:
        msk = np.ones_like(xx)

    # calculate water column thickness
    water_thickness = surf - thick - bed

    # water prisms
    water_msk = ((ds.mask==0) ^ (ds.mask==3)) & (msk==True)
    prisms_water = layer_prisms(xx, yy, bed, bed+water_thickness, water_msk, half_res)
    prisms_water, idx_water_pos = split_prisms(prisms_water)
    d_water = np.where(idx_water_pos, water_dens, water_dens-rock_dens)
    
    # positive rock
    rock_msk = (bed > 0) & (msk==True)
    prisms_rock = layer_prisms(xx, yy, np.zeros_like(xx), bed, rock_msk, half_res)
    d_rock = np.full(prisms_rock.shape[0], rock_dens)
    
    # negative rock
    negrock_msk = (surf < 0) & (msk==True)
    prisms_negrock = layer_prisms(xx, yy, surf, np.zeros_like(xx), negrock_msk, half_res)
    d_negrock = np.full(prisms_negrock.shape[0], -rock_dens)
    
    # ice prisms
    if ice==True:
        ice_msk = ((ds.mask==3) ^ (ds.mask==2)) & (msk==True)
        prisms_ice = layer_prisms(xx, yy, surf-thick, surf, ice_msk, half_res)
        prisms_ice, idx_ice_pos = split_prisms(prisms_ice)
        d_ice = np.where(idx_ice_pos, ice_dens, ice_dens-rock_dens)

    if ice==True:
        prisms = np.vstack([prisms_ice, prisms_water, prisms_rock, prisms_negrock])
        densities = np.concatenate([d_ice, d_water, d_rock, d_negrock])
    else:
        prisms = np.vstack([prisms_water, prisms_rock, prisms_negrock])
        densities = np.concatenate([d_water, d_rock, d_negrock])

    # remove bad prisms where bottom is above top
    bad_idx = np.nonzero(prisms[:,4] > prisms[:,5])[0]
    prisms = np.delete(prisms, bad_idx, axis=0)
    densities = np.delete(densities, bad_idx, axis=0)

    return prisms, densities

def layer_prisms(xx, yy, bottom, top, msk, half_res):
    """
    Make right-rectangular prism coordinates for a layer
    Args:
        xx : 2D array of x-coordinates
        yy : 2D array of y-coordinates
        bottom : 2D array of bottom surface
        top : 2D array of top surface
        msk : 2D binary array. Make prisms where True
        half_res : half of the grid spacing
    Outputs:
        Numpy array of prisms coordinates with shape (n_prisms, 6).
        The coordinates are in order W, E, S, N, bottom, top
    """
    prisms = np.array([
        xx[msk]-half_res,
        xx[msk]+half_res,
        yy[msk]-half_res,
        yy[msk]+half_res,
        bottom[msk],
        top[msk]
    ]).T
    return prisms

def split_prisms(prisms):
    '''
    Function to split prisms above and below the ellipsoid.
    Args:
        prisms : Numpy array of prism coordinates
    Outputs:
        Combined split prisms and an index of which ones are above the ellipsoid.
    '''
    prisms_pos = prisms[prisms[:,5] >= 0, :]
    prisms_neg = prisms[prisms[:,4] < 0, :]
    prisms_pos[prisms_pos[:,4] < 0, 4] = 0.0
    prisms_neg[prisms_neg[:,5] > 0, 5] = 0.0
    prisms = np.vstack([prisms_pos, prisms_neg])
    idx_pos = np.full(prisms.shape[0], False)
    idx_pos[:prisms_pos.shape[0]] = True
    return prisms, idx_pos