#!/bin/bash
#SBATCH -J ATM_CACHE
#SBATCH -q preempt
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A mp107a

# Python environment
ulimit -c unlimited
export PYTHONSTARTUP=""
export PYTHONNOUSERSITE=1
export HOME=$SCRATCH
export HDF5_USE_FILE_LOCKING=FALSE

# TOAST variables
export TOAST_FUNCTIME=1
export TOAST_LOGLEVEL=DEBUG

# Parallelization
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export OMP_NUM_THREADS=4

echo "$(date) : Start"
echo "Running with"
echo "            nnode = ${SLURM_JOB_NUM_NODES}"
echo "  OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "            ntask = ${SLURM_NTASKS}"
echo "    cpus-per-task = ${SLURM_CPUS_PER_TASK}"

# go to submission directory
cd ${SLURM_SUBMIT_DIR}

# schedule file
schedule="schedules/schedule.01jan.txt"

# output
outdir=$SCRATCH/runs/pairdiff/atm_cache
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --thinfp 64 \
    --schedule $schedule \
    --sim_atmosphere.cache_only \
    --sim_atmosphere_coarse.cache_only \
    --out $outdir \
    >>$logfile 2>&1

echo "$(date) : Done!"
