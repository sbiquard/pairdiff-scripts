#!/bin/bash
#SBATCH --job-name=leak-atm-p(f)-nohwp
#SBATCH --account=nih@cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00

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
sky="ffp10_polar_0512.fits"

outdir=$SCRATCH/pairdiff_runs/validation/incl/correlated-regularized-nohwp-noiseless-polar
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    $(< no_flags.par) \
    $(< no_hwp.par) \
    --mappraiser.estimate_psd \
    --mappraiser.zero_noise \
    --mappraiser.psd_regularization 6e-8 \
    --variable_model.disable \
    --sim_noise.disable \
    --schedule $schedule \
    --out $outdir \
    --scan_map.file $sky \
    >$logfile 2>&1

echo "$(date) : Done!"
