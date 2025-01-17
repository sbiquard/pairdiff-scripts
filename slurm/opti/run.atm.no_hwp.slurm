#!/bin/bash
#SBATCH --job-name=check-atm-simu-no-hwp
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

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

outdir=out/checks/atm_no_hwp
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 4 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 10000 \
    --sim_atmosphere.enable \
    --sim_ground.hwp_rpm 0 \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 4 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 10000 \
    --mappraiser.pair_diff \
    --sim_atmosphere.enable \
    --sim_ground.hwp_rpm 0 \
    --out $outdir/pd \
    >$logfile 2>&1

echo "$(date) : Done!"
