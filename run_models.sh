#!/usr/bin/env bash

# SGS interpolation with filtering at 2 km resolution
python -u sgs_inversions.py -p results/cond_dens -c -d -n 100 -f -s 1.2
python -u sgs_inversions.py -p results/cond_nodens -c -n 100 -f -s 1.2
python -u sgs_mean.py -p results/cond_deterministic -c -n 100 -s 1.2

# no conditioning
# python -u sgs_inversions.py -p results/uncond_dens -d -n 100 -f -s 1.2
# python -u sgs_inversions.py -p results/uncond_nodens -n 100 -f -s 1.2
# python -u sgs_mean.py -p results/uncond_deterministic -n 100 -s 1.2

# test
# python -u sgs_inversions.py -p results/test/cond_dens -c -d -n 2 -f -s 1.2
# python -u sgs_inversions.py -p results/test/cond_nodens -c -n 2 -f -s 1.2
# python -u sgs_mean.py -p results/test/cond_deterministic -c -n 2 -s 1.2