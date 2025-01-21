import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm.auto import tqdm
from pathlib import Path
import verde as vd
import os
from utilities import lowpass_filter_invpad

"""
Run this script to load inversions, upscale them, and save upscaled beds.
"""

def load_data(ds, res_path, geoid=False, filter=False):
    """
    Load inversions results from directory

    Args:
        ds : Xarray.Dataset preprocessed BedMachine data
        res_path : path to directory with inversion results
        geoid : reference beds to geoid from ellipsoid
        filter : apply Gaussian lowpass filter to remove edges
    Outputs:
        Beds, mean of beds, stdev of beds, densities, losses
    """
    count = 0
    for entry in os.scandir(res_path):
        if 'result' in entry.name and '._' not in entry.name:
            count += 1
    print(f' {count} inversions')
    
    densities = np.zeros(count)
    last_iter = np.zeros((count, *ds.bed.shape))
    losses = np.zeros((count, 47_000))
    
    i = 0
    for entry in os.scandir(res_path):
        if 'result' in entry.name and '._' not in entry.name:
            result = np.load(entry.path, allow_pickle=True).item()
            densities[i] = result['density'][0]
            bed = result['bed_cache']
            if filter==True:
                bed_filt = lowpass_filter_invpad(ds, bed, cutoff=5e3)
                last_iter[i] = bed_filt.reshape(bed.shape)
            else:
                last_iter[i] = bed
            losses[i,:result['loss_cache'].size] = result['loss_cache']
            i += 1
            
            
    mean = np.mean(last_iter, axis=0)
    std = np.std(last_iter, axis=0)

    if geoid==True:
        last_iter -= ds.geoid.values
        mean -= ds.geoid.values
        std -= ds.geoid.values

    return last_iter, mean, std, densities, losses

def upscale_data(ds, grid, data, grid_vals, preds_msk, outside=True):
    """
    Upscale data to BedMachine v3 500 m resolution. Data and grid_vals are
    not bathymetry specific so that other fields like standard deviation can
    be upscaled as well.
    Args:
        ds : trimmed and coarsened BedMachine xarray.Dataset used for the inversions
        grid : trimmed BedMachine xarray.Dataset with original resolution
        data : array to upscale
        grid_vals : conditioning data at higher resolution
        preds_msk : inversion domain at higher resolution
        outside : if True, interpolate between the grid_vals outside inversion domain.
            Use True if interpolating coarse bathymetry to higher resolution bathymetry.
    Outputs:
        Data at 500m BedMachine resolution.
    """
    xx_i, yy_i = np.meshgrid(grid.x.values, grid.y.values)
    pred_coords = np.stack([xx_i.flatten(), yy_i.flatten()]).T
    xx_int = xx_i[~preds_msk]
    yy_int = yy_i[~preds_msk]
    interp_coords = np.stack([xx_int, yy_int]).T
    interp_vals = grid_vals[~preds_msk]
    
    xx_g, yy_g = np.meshgrid(ds.x.values, ds.y.values)
    xx_g = xx_g[ds.inv_no_muto.values]
    yy_g = yy_g[ds.inv_no_muto.values]
    interp_coords_grav = np.stack([xx_g.flatten(), yy_g.flatten()]).T
    interp_vals_grav = data[ds.inv_no_muto.values]
    if outside==True:
        interp_vals_i = np.concatenate([interp_vals, interp_vals_grav])
        interp_coords_i = np.concatenate([interp_coords, interp_coords_grav], axis=0)
        
        upscale = griddata(interp_coords_i, interp_vals_i, pred_coords, method='cubic').reshape(grid.bed.shape)
    else:
        upscale = griddata(interp_coords_grav, interp_vals_grav, pred_coords, method='cubic').reshape(grid.bed.shape)
    upscale = np.where(preds_msk, upscale, grid.bed.values)
    return upscale

def save_upscale(ds, grid, pred_msk, data_path, out_path, out_path_up):
    """
    Load inversions, upscale beds, save upscaled beds

    Args:
        ds : trimmed and coarsened BedMachine xarray.Dataset used for the inversions
        grid : trimmed BedMachine xarray.Dataset at original resolution
        pred_msk : inversion domain mask at higher resolution
        data_path : path to directory with inversions
        out_path : path to save upscaled beds
    Outputs:
        None
    """
    beds, _, _, _, _ = load_data(ds, data_path, geoid=True, filter=True)

    beds_up = np.zeros((beds.shape[0], *grid.bed.shape))
    
    for i in tqdm(range(beds.shape[0])):
        beds_up_i = upscale_data(ds, grid, beds[i], grid.bed.values, preds_msk)
        beds_up[i,...] = np.where(beds_up_i > grid.surface-grid.thickness, grid.surface-grid.thickness, beds_up_i)

    ii = np.arange(beds.shape[0])
    ds_beds = xr.DataArray(beds, coords = {'i' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
    ds_beds_up = xr.DataArray(beds_up, coords = {'i' : ii, 'y' : grid.y.values, 'x' : grid.x.values})

    # save as netcdf
    ds_beds.to_netcdf(out_path)
    ds_beds_up.to_netcdf(out_path_up)

if __name__ == '__main__':
    # load preprocessed and original BedMachine
    ds = xr.open_dataset(Path('processed_data/xr_2000.nc'))
    grid = xr.open_dataset(Path('raw_data/bedmachine/BedMachineAntarctica-v3.nc'))

    xx, yy = np.meshgrid(ds.x, ds.y)

    # trim original BedMachine, get coordinates
    x_trim = (grid.x >= np.min(xx)) & (grid.x <= np.max(xx))
    y_trim = (grid.y >= np.min(yy)) & (grid.y <= np.max(yy))
    grid = grid.sel(x=x_trim, y=y_trim)
    xx_bm, yy_bm = np.meshgrid(grid.x.values, grid.y.values)

    # interpolate inversion mask to original resolution
    kn = vd.KNeighbors(1)
    kn.fit((xx.flatten(), yy.flatten()), ds.inv_no_muto.values.flatten())
    preds_msk = kn.predict((xx_bm, yy_bm))
    preds_msk = preds_msk.reshape(xx_bm.shape) > 0.5

    # path to where inversion directories are
    base_path = Path('results')

    # save ensemble with conditioning and density
    print('upscaling beds cd')
    save_upscale(ds, grid, preds_msk,
                 base_path/'cond_dens',
                 base_path/'cond_dens_geoid_2000.nc',
                 base_path/'cond_dens_geoid_500.nc')

    # save ensemble with conditioning and no density
    print('upscaling beds cnd')
    save_upscale(ds, grid, preds_msk,
                 base_path/'cond_nodens',
                 base_path/'cond_nodens_geoid_2000.nc',
                 base_path/'cond_nodens_geoid_500.nc')

    # save ensemble with conditioning and no deteministic bouger
    print('upscaling beds c determ')
    save_upscale(ds, grid, preds_msk,
                 base_path/'cond_deterministic',
                 base_path/'cond_determ_geoid_2000.nc',
                 base_path/'cond_determ_geoid_500.nc')

    # # save ensemble with no conditioning and density
    # print('upscaling beds ucd')
    # save_upscale(grid, preds_msk,
    #              base_path/'uncond_dens',
    #              base_path/'uncond_dens_geoid_2000.nc',
    #              base_path/'uncond_dens_geoid_500.nc')

    # # save ensemble with no conditioning and no density
    # print('upscaling beds ucd')
    # save_upscale(grid, preds_msk,
    #              base_path/'uncond_nodens',
    #              base_path/'uncond_nodens_geoid_2000.nc',
    #              base_path/'uncond_nodens_geoid_500.nc')

    # # save ensemble with no conditioning and no deteministic bouger
    # print('upscaling beds uc determ')
    # save_upscale(grid, preds_msk,
    #              base_path/'uncond_deterministic',
    #              base_path/'uncond_determ_geoid_2000.nc',
    #              base_path/'uncond_determ_geoid_500.nc')