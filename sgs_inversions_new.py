import numpy as np
from numpy.random import PCG64, SeedSequence
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import xarray as xr
import os
import sys
import multiprocessing as mp
import time
from datetime import datetime
import pickle
import argparse

import warnings
warnings.filterwarnings('ignore')

from prisms import *
from rfgen import *
from bouguer import *
from block_update import *

os.environ['KMP_WARNINGS'] = 'FALSE'

parser = argparse.ArgumentParser(description='Run bathymetry inversions with SGS interpolation')
parser.add_argument('-n', '--ninvs', default=100, type=int, help='number of inversions')
parser.add_argument('-f', '--filt', action='store_true', default=False, help='filter SGS')

args = parser.parse_args()

if __name__ == '__main__':

    # print(f'\nrunning {args.ninvs} SGS inversions')
    # print(f'stop: {args.stop}')
    # print(f'condition: {args.condition}')
    # print(f'density: {args.density}')
    print(f'filt: {args.filt}')

    start_time = datetime.now()
    print('start: ', start_time)

    # results_path = Path(args.path)
    # if os.path.exists(results_path) == False:
    #     os.makedirs(results_path)

    results_path_nodens = Path('results/test/nodens')
    results_path_dens = Path('results/test/dens')
    results_path_krige = Path('results/test/krige')

    if os.path.exists(results_path_nodens) == False:
        os.makedirs(results_path_nodens)
    if os.path.exists(results_path_dens) == False:
        os.makedirs(results_path_dens)
    if os.path.exists(results_path_krige) == False:
        os.makedirs(results_path_krige)

    # load data
    ds = xr.load_dataset(Path('processed_data/xr_2000.nc'))
    grav = pd.read_csv(Path('processed_data/grav_leveled_2000.csv'))

    # trim gravity data
    grav_mskd = grav[grav.inv_pad==True]

    # gravity prediction locations
    pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)

    # make arrays for random field generation
    range_max = [50e3, 50e3]
    range_min = [30e3, 30e3]
    high_step = 300
    nug_max = 0.0
    eps = 3e-4
    
    # block size, range, amplitude, iterations
    sequence = [
        [21, 10, 60, 1000],
        [15, 8, 40, 1000],
        [9, 6, 40, 5000],
        [5, 5, 40, 40000]
    ]

    density_dict = {
            'ice' : 917,
            'water' : 1027,
            'rock' : 2670
    }

    # gravity uncertainty
    sigma = 1.6

    # stopping condition
    stop = 1.2

    # make base PRNG
    root_seed = 328613813390984468677358742156199349641
    base_seq = SeedSequence()
    rng = np.random.default_rng(base_seq)

    n_invs = args.ninvs

    target_cache_nodens = np.zeros((n_invs, grav.shape[0]))
    target_cache_dens = np.zeros((n_invs, grav.shape[0]))

    print(f'running {n_invs} inversions with fixed densities')
    pbar = tqdm(range(n_invs))
    for i in pbar:
        rng_i = np.random.default_rng([i, root_seed])

        # bouguer interpolation
        target = boug_interpolation_sgs(ds, grav, density=2670, covmodel='matern', k=24, rng=rng_i)

        if args.filt == True:
            boug_filt = filter_boug(ds, grav, target, cutoff=12e3, pad=0)
            target = grav.faa.values - boug_filt

        # save target
        target_cache_nodens[i,:] = target
        
        # trim to mask
        target = target[grav.inv_pad==True]

        # initial pertubation away from BedMachine
        rfgen = RFGen(ds, range_max, range_min, high_step, nug_max, eps, 'Gaussian', rng=rng_i)
        x0 = ds.bed.data + rfgen.generate_field(condition=True, seed=rng_i.integers(10_000, 20_000, 1))
        x0 = np.where(x0>ds.surface-ds.thickness, ds.surface-ds.thickness, x0)
        
        path = Path(results_path_nodens/f'result_{i}.npy')
        
        result = chain_sequence(sequence, ds, x0, pred_coords, target, sigma, density_dict, rng_i, 
                                weights=None, stop=stop, save=path, full_cache=False, quiet=True)

        pbar.update(1)
    pbar.close()

    print('finished ensemble with fixed density')
    print(f'running {n_invs} ensemble with variable densities')
    pbar = tqdm(range(n_invs))
    for i in pbar:
        rng_i = np.random.default_rng([i+n_invs, root_seed])

        rock_density = rng_i.normal(loc=2700, scale=50, size=1)
        density_dict['rock'] = rock_density

        # bouguer interpolation
        target = boug_interpolation_sgs(ds, grav, density=rock_density, covmodel='matern', k=24, rng=rng_i)

        if args.filt == True:
            boug_filt = filter_boug(ds, grav, target, cutoff=12e3, pad=0)
            target = grav.faa.values - boug_filt

        # save target
        target_cache_dens[i,:] = target

        # trim to mask
        target = target[grav.inv_pad==True]

        # initial pertubation away from BedMachine
        rfgen = RFGen(ds, range_max, range_min, high_step, nug_max, eps, 'Gaussian', rng=rng_i)
        x0 = ds.bed.data + rfgen.generate_field(condition=True, seed=rng_i.integers(10_000, 20_000, 1))
        x0 = np.where(x0>ds.surface-ds.thickness, ds.surface-ds.thickness, x0)
        
        path = Path(results_path_dens/f'result_{i}.npy')
        
        result = chain_sequence(sequence, ds, x0, pred_coords, target, sigma, density_dict, rng_i, 
                                weights=None, stop=stop, save=path, full_cache=False, quiet=True)

        pbar.update(1)
    pbar.close()

    print('finished ensemble with variable densities')

    # get kriging mean
    krige_mean = np.load(Path('processed_data/krige_mean.npy'))
    inv_pad = xy_into_grid(ds, (grav.x.values, grav.y.values), grav.inv_pad.values)
    inv_pad = np.where(inv_pad==1, 1, 0)
    boug_mean = krige_mean[inv_pad==1]
    target_mean = grav_mskd.faa - boug_mean

    # remake density dictionary
    density_dict = {
            'ice' : 917,
            'water' : 1027,
            'rock' : 2670
    }
    
    print('running ensemble with fixed Bouguer disturbance')
    pbar = tqdm(range(n_invs))
    for i in pbar:
        rng_i = np.random.default_rng([i+n_invs*2, root_seed])

        # initial pertubation away from BedMachine
        rfgen = RFGen(ds, range_max, range_min, high_step, nug_max, eps, 'Gaussian', rng=rng_i)
        x0 = ds.bed.data + rfgen.generate_field(condition=True, seed=rng_i.integers(10_000, 20_000, 1))
        x0 = np.where(x0>ds.surface-ds.thickness, ds.surface-ds.thickness, x0)
        
        path = Path(results_path_krige/f'result_{i}.npy')
        
        result = chain_sequence(sequence, ds, x0, pred_coords, target_mean, sigma, density_dict, rng_i, 
                                weights=None, stop=stop, save=path, full_cache=False, quiet=True)

        pbar.update(1)
    pbar.close()

    # save Bouguer simulations
    np.save(Path('results/simulation_nodens.npy'), target_cache_nodens)
    np.save(Path('results/simulation_dens.npy'), target_cache_dens)

    end_time = datetime.now()
    print('end: ', end_time)
    print('time elapsed: ', end_time-start_time)
    sys.exit()
