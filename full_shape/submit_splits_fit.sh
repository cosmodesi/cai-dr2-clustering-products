#!/bin/bash
#SBATCH -J fit
#SBATCH -A desi_g
#SBATCH -q regular
#SBATCH -t 21:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus 4
#SBATCH -n 4
#SBATCH -o fit-%x-%A_%a.out
#SBATCH --array=0-7     # 8 regions

# Load environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# source /global/homes/s/shengyu/env.sh fit_env

# Tracers
TRACERS=("BGS" "LRG" "ELG" "QSO")

# regions controlled by the job array
REGIONS=("GCcomb" "NGC" "SGC" "N" "SNGC" "Scomb" "noDEScomb" "SnoDEScomb")
region="${REGIONS[$SLURM_ARRAY_TASK_ID]}"

for tracer in "${TRACERS[@]}"; do
  echo ">>> Running tracer=$tracer region=$region approach=(FM)"
  srun -n 4 python fit_blinded_data.py --tracers "$tracer" --regions "$region" --approaches FM

  echo ">>> Running tracer=$tracer region=$region, approach=(SF + _wq_prior)"
  srun -n 4 python fit_blinded_data.py --tracers "$tracer" --regions "$region" --approaches SF --option _wq_prior
done

echo ">>> All tracers completed for region: $region"