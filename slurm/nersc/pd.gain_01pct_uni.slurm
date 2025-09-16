#!/bin/bash
#SBATCH -J PD_GAIN01PCT_UNI
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
outdir=$SCRATCH/runs/pairdiff/hwp/pd_gain_01pct_uni
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"

srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    --gainscrambler.enable \
    --gainscrambler.scale 0.001 \
    --gainscrambler.constant \
    --mappraiser.pair_diff \
    --mappraiser.estimate_psd \
    --schedule $schedule \
    --out $outdir \
    >$logfile 2>&1

# output
outdir=$SCRATCH/runs/pairdiff/nohwp/pd_gain_01pct_uni
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"

srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    $(< no_hwp.par) \
    --gainscrambler.enable \
    --gainscrambler.scale 0.001 \
    --gainscrambler.constant \
    --mappraiser.pair_diff \
    --mappraiser.estimate_psd \
    --schedule $schedule \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
