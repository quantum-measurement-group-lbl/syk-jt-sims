#!/bin/bash

#SBATCH --job-name=syk_GLL_cpu_mpi  # Job name
#SBATCH --qos=regular           # qos
#SBATCH --account=m5258
#SBATCH --time=03:00:00            # wallclock time
#SBATCH --constraint=cpu
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/syk_test9-7-26_N50_1.%j.out
#SBATCH --error=logs/syk_test9-7-26_N50_1.%j.err
#SBATCH --mail-user=bpknepper@lbl.gov
#SBATCH --mail-type=ALL

module load conda
conda activate /global/common/software/m5258/conda/syk_dynamite_env

# Prevent NumPy or other libraries from starting hidden thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=== SYK CPU MPI NERSC Perlmutter ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

srun --cpu-bind=cores python -u run_syk_wormhole_NERSC_cpuMpi.py \
    -N 50 \
    -b '0.0, 20.0, 40.0' \
    -t '0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0' \
    --H-iters 5 \
    --state-iters 50 \
    --output-dir results
