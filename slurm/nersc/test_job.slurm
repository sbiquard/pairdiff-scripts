#!/bin/bash
#SBATCH -J PD_TEST
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
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
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=close

let nnode=$SLURM_JOB_NUM_NODES
# 128 cores, 258 hardware threads
let ntask_node=$SLURM_NTASKS_PER_NODE
let ntask=$nnode*$ntask_node

echo "$(date) : Start"
echo "Running with"
echo "            nnode = ${nnode}"
echo "  OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "       ntask_node = ${ntask_node}"
echo "            ntask = ${ntask}"

# Just 1 CES
schedule="schedules/schedule.opti.txt"

outdir=$SCRATCH/runs/pairdiff/test
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_flags.par) \
    --schedule $schedule \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
