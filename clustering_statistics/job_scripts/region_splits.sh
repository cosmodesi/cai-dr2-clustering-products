#!/bin/bash

# !!! ask allocation first !!!
# salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
# source /global/homes/s/shengyu/env.sh 2pt_env

# !!! Process one catalog bin and region per job to avoid cross-bin interaction issues
# indx=6
for indx in {0..6}; do
    for region in NGC SGC N SNGC SGCnoDES GCcomb Scomb noDEScomb SnoDEScomb; do
        echo ">>> Processing indx=$indx region=$region"
        # srun -n 4 python blinded_data_pip.py --version dr2-v2 --indx $indx --regions $region --todo blinded_mesh2_spectrum
        # srun -n 4 python blinded_data_pip.py --version dr2-v2 --indx $indx --regions $region --todo window_mesh2_spectrum
        srun -n 4 python blinded_data_pip.py --version dr2-v2 --indx $indx --regions $region --todo covariance_mesh2_spectrum
    done
done