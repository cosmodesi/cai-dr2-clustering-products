# !/bin/bash

# Activate environments
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# source /global/homes/s/shengyu/env.sh fit_env

# -----------------------------------------------
# Cosmological fitting
# -----------------------------------------------
# TRACERS=("BGS") #"BGS" "LRG" "ELG" "QSO"
# REGION = GCcomb NGC SGC N S noDES SnoDES
# splits on region and compute the power poles
# for indx in {0..5}; do
indx=6
echo ">>> Processing indx=$indx"
srun -N 1 -n 1 -C cpu -c 128 -t 04:00:00 --qos interactive --account desi \
python fit_blinded_data.py --indx $indx --regions GCcomb NGC SGC N --approaches SF --option _wq_prior
# python fit_blinded_data.py --indx $indx --regions SNGC Scomb noDEScomb SnoDEScomb --approaches SF --option _wq_prior

# srun -N 1 -n 1 -C cpu -c 128 -t 04:00:00 --qos interactive --account desi python fit_blinded_data.py --indx $indx --regions GCcomb--approaches FM