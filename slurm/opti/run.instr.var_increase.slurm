#!/bin/bash
#SBATCH --job-name=opti-var-increase-instr
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=50
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

schedule="schedules/schedule.01.south.txt"

opti="${SCRATCH}/pairdiff_runs/opti"

rootdir_hwp="${opti}/var_increase_instr"
rootdir_no_hwp="${opti}/var_increase_instr_no_hwp"

# looping over scatter values

for scatter in 0.001 0.01 0.1 0.2; do

    echo "Running scatter = $scatter"

    #--------#
    # HWP ON #
    # -------#

    outdir=$rootdir_hwp/scatter_${scatter}
    mkdir -p $outdir/ml
    mkdir -p $outdir/pd

    # ML run
    logfile=$outdir/ml/run.log
    echo "Writing $logfile"
    srun ./so_mappraiser.py \
        $(<sat.par) \
        --schedule $schedule \
        --mappraiser.lagmax 512 \
        --variable_model.scatter $scatter \
        --out $outdir/ml \
        >$logfile 2>&1

    # PD run
    logfile=$outdir/pd/run.log
    echo "Writing $logfile"
    srun ./so_mappraiser.py \
        $(<sat.par) \
        --schedule $schedule \
        --mappraiser.lagmax 512 \
        --mappraiser.pair_diff \
        --variable_model.scatter $scatter \
        --out $outdir/pd \
        >$logfile 2>&1

    #---------#
    # HWP OFF #
    # --------#

    outdir=$rootdir_no_hwp/scatter_${scatter}
    mkdir -p $outdir/ml
    mkdir -p $outdir/pd

    # ML run
    logfile=$outdir/ml/run.log
    echo "Writing $logfile"
    srun ./so_mappraiser.py \
        $(<sat.par) \
        $(< no_hwp.par) \
        --schedule $schedule \
        --mappraiser.lagmax 512 \
        --variable_model.scatter $scatter \
        --out $outdir/ml \
        >$logfile 2>&1

    # PD run
    logfile=$outdir/pd/run.log
    echo "Writing $logfile"
    srun ./so_mappraiser.py \
        $(<sat.par) \
        $(< no_hwp.par) \
        --schedule $schedule \
        --mappraiser.lagmax 512 \
        --mappraiser.pair_diff \
        --variable_model.scatter $scatter \
        --out $outdir/pd \
        >$logfile 2>&1

done

echo "$(date) : Done!"
