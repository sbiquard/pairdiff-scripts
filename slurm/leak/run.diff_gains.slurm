#!/bin/bash
#SBATCH --job-name=leak-diff-gains
#SBATCH --account=nih@cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

# go to submission directory
cd ${SLURM_SUBMIT_DIR}

# OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# OpenMP binding
export OMP_PLACES=cores

let nnode=$SLURM_JOB_NUM_NODES
let ntask_node=$SLURM_NTASKS_PER_NODE
let ntask=nnode*ntask_node

# print some meta information
date
echo "Running with"
echo "            nnode = ${nnode}"
echo "  OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "       ntask_node = ${ntask_node}"
echo "            ntask = ${ntask}"

schedule="schedules/schedule.small.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=$SCRATCH/pairdiff_runs/leak/diff_gains
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --gainscrambler.enable \
    --mappraiser.estimate_psd \
    --mappraiser.maxiter 2000 \
    --variable_model.disable \
    --sim_noise.disable \
    --thinfp 4 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --gainscrambler.enable \
    --mappraiser.estimate_psd \
    --mappraiser.maxiter 2000 \
    --variable_model.disable \
    --sim_noise.disable \
    --thinfp 4 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.pair_diff \
    --out $outdir/pd \
    >$logfile 2>&1

    echo "$(date) : Done!"
