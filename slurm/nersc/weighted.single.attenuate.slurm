#!/bin/bash
#SBATCH -J WEIGHTED_HWP_SINGLE_ATTENUATE
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 4
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
export TOAST_LOGLEVEL=INFO

# Parallelization
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export OMP_NUM_THREADS=2

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
outdir=$SCRATCH/runs/pairdiff/hwp/weighted_single_attenuate
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"

srun ./so_mappraiser.py \
    --single-det \
    $(< sat.par) \
    $(< atm.par) \
    --scan_map.file "nside0512_attenuated_100.fits" \
    --sim_atmosphere_coarse.gain 6e-06 \
    --sim_atmosphere.gain 4e-07 \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    --mappraiser.estimate_psd \
    --mappraiser.save_noise_model \
    --schedule $schedule \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
