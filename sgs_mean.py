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

parser = argparse.ArgumentParser(description='Run bathymetry inversions with mean Bouguer')
parser.add_argument('-p', '--path', required=True, help='path to results')
parser.add_argument('-c', '--condition', action='store_true', default=False, help='condition on edges')
parser.add_argument('-n', '--ninvs', default=100, type=int, help='number of inversions')
parser.add_argument('-s', '--stop', default=2.5, type=float, help='RMSE stopping criterion')

args = parser.parse_args()

if __name__ == '__main__':

    print(f'\nrunning {args.ninvs} inversions with same Bouguer')
    print(f'stop: {args.stop}')
    print(f'condition: {args.condition}')

    start_time = datetime.now()
    print('start: ', start_time)

    results_path = Path(args.path)
    if os.path.exists(results_path) == False:
        os.mkdir(results_path)

    # resolution
    res = 2000

    # load data
    ds = xr.load_dataset(Path(f'processed_data/xr_{res}.nc'))
    grav = pd.read_csv(Path(f'processed_data/grav_leveled_{res}.csv'))

    # trim gravity data
    grav_mskd = grav[grav.inv_pad==True]

    # make arrays for random field generation
    range_max = [50e3, 50e3]
    range_min = [30e3, 30e3]
    high_step = 300
    nug_max = 0.0
    eps = 3e-4
    
    rfgen = RFGen(ds, range_max, range_min, high_step, nug_max, eps, 'Gaussian')
    
    # block size, range, amplitude, iterations
    sequence = [
        [21, 10, 60, 1000],
        [15, 8, 40, 1000],
        [9, 6, 40, 5000],
        [5, 5, 40, 40000]
    ]

    # gravity uncertainty
    sigma = 1.6

    if args.condition == True:
        # conditioning weights
        min_dist_l2 = min_dist(ds)
        dist_scale = rescale(min_dist_l2)
        dist_logi = logistic(dist_scale, 1, 0.05, 18)
    else:
        dist_logi = None
    
    base_seq = SeedSequence()
    rng = np.random.default_rng(base_seq)

    # number of inversions to run with sampled densities
    n_invs = int(args.ninvs)

    entropies = []
    input_params = []
    # create input parameters to chain_sequence
    for i in range(n_invs):
        # initial pertubation away from BedMachine
        x0 = ds.bed.data + rfgen.generate_field(condition=True)
        x0 = np.where(x0>ds.surface-ds.thickness, ds.surface-ds.thickness, x0)
        
        density_dict = {
            'ice' : 917,
            'water' : 1027,
            'rock' : 2670
        }
        # spawn new random number generator
        new_seq = SeedSequence()
        entropies.append(new_seq.entropy)
        rng_i = np.random.default_rng(PCG64(new_seq))
        
        # gravity prediction locations
        pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)
        te_dist_i = grav_mskd.te_mean.values.copy()
        
        path = Path(results_path/f'result_{i}.npy')
        
        # RMSE stopping condition
        stop = args.stop
        
        input_params.append([sequence, ds, x0, pred_coords, te_dist_i, sigma, density_dict, rng_i, dist_logi, stop, path, False, True, i])

    print('running inversions')
    # run inversions in parallel
    nproc = min(4, args.ninvs)
    with mp.Pool(processes=4) as pool:
        for _ in pool.starmap(chain_sequence, input_params):
            pass

    with open(results_path/'rng_entropies.txt', 'w') as f:
        f.writelines([str(i)+'\n' for i in entropies])

    end_time = datetime.now()
    print('end: ', end_time)
    print('time elapsed: ', end_time-start_time)
