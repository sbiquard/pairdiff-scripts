#!/bin/bash
#SBATCH --job-name=opti-one-over-f
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00

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

schedule="schedule.01.south.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=out/opti/one_over_f
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 4 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 512 \
    --elevation_model.disable \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 4 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 512 \
    --mappraiser.pair_diff \
    --elevation_model.disable \
    --out $outdir/pd \
    >$logfile 2>&1

echo "$(date) : Done!"
