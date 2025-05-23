#!/bin/bash
#SBATCH --job-name=opti-white-noise
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

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

schedule="schedules/schedule.01.south.txt"

rootdir=$SCRATCH/pairdiff_runs/opti/white

# -----------#
# NO SCATTER #
# -----------#

outdir=$rootdir/no_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --variable_model.white \
    $(<no_scatter.par) \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --mappraiser.pair_diff \
    --variable_model.white \
    $(<no_scatter.par) \
    --out $outdir/pd \
    >$logfile 2>&1

# -------------#
# SAME SCATTER #
# -------------#

outdir=$rootdir/same_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --variable_model.white \
    $(<same_scatter.par) \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --mappraiser.pair_diff \
    --variable_model.white \
    $(<same_scatter.par) \
    --out $outdir/pd \
    >$logfile 2>&1

# -----------------#
# OPPOSITE SCATTER #
# -----------------#

outdir=$rootdir/opposite_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --variable_model.white \
    $(<opposite_scatter.par) \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --mappraiser.pair_diff \
    --variable_model.white \
    $(<opposite_scatter.par) \
    --out $outdir/pd \
    >$logfile 2>&1

# ---------------#
# RANDOM SCATTER #
# ---------------#

outdir=$rootdir/random_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --variable_model.white \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(<sat.par) \
    --schedule $schedule \
    --mappraiser.lagmax 1 \
    --mappraiser.pair_diff \
    --variable_model.white \
    --out $outdir/pd \
    >$logfile 2>&1

echo "$(date) : Done!"
