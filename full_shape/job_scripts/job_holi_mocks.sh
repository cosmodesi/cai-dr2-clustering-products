#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 0:20:00
#SBATCH -q regular
#SBATCH -J holi_mocks
#SBATCH -L SCRATCH
#SBATCH -o slurm_outputs/holi_mocks_%A/mock%a.log
#SBATCH --array=451-500,601-650

set -e
# Timer initialisation:
SECONDS=0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

imock=$SLURM_ARRAY_TASK_ID
#CAT_DIR=/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/altmtl$imock/loa-v1/mock$imock/LSScats/
#MEAS_DIR=/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/holi_v1/altmtl/mock$imock/spectrum/
VERSION='holi-v1-altmtl'
MEAS_DIR=$SCRATCH/holi_v1/altmtl/spectrum/

CODE=../compute_fiducial_stats.py
echo $MEAS_DIR

JOB_FLAGS="-N 1 -n 4"
COMMON_FLAGS="--stats mesh2_spectrum --region NGC SGC --imock $imock --version $VERSION --meas_dir $MEAS_DIR --combine"

#LRG_FLAGS="--tracer LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --weight_type default_FKP"
LRG_FLAGS="--tracer LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --weight_type default_FKP"
ELG_FLAGS="--tracer ELG_LOPnotqso --zrange 0.8 1.1 1.1 1.6 --weight_type default_FKP"
QSO_FLAGS="--tracer QSO --zrange 0.8 2.1 --weight_type default_FKP"

srun $JOB_FLAGS python $CODE $LRG_FLAGS $COMMON_FLAGS
#srun $JOB_FLAGS python $CODE $ELG_FLAGS $COMMON_FLAGS
#srun $JOB_FLAGS python $CODE $QSO_FLAGS $COMMON_FLAGS

echo " "
if (( $SECONDS > 3600 )); then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )); then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi
echo
