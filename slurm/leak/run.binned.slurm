#!/bin/bash
#SBATCH --job-name=leak-atm-binned
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=2
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

schedule="schedules/schedule.small.txt"

outdir=$SCRATCH/pairdiff_runs/leak/binned
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --sim_noise.disable \
    --mappraiser.binned \
    --schedule $schedule \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --sim_noise.disable \
    --mappraiser.binned \
    --schedule $schedule \
    --mappraiser.pair_diff \
    --out $outdir/pd \
    >$logfile 2>&1

outdir=$SCRATCH/pairdiff_runs/leak/binned_no_hwp
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    $(< no_hwp.par) \
    --sim_noise.disable \
    --mappraiser.binned \
    --schedule $schedule \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    $(< no_hwp.par) \
    --sim_noise.disable \
    --mappraiser.binned \
    --schedule $schedule \
    --mappraiser.pair_diff \
    --out $outdir/pd \
    >$logfile 2>&1

echo "$(date) : Done!"
