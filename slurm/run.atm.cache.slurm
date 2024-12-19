#!/bin/bash
#SBATCH --job-name=pdiff-atm-cache
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --open-mode=append  # append to the output file

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

# schedule file
schedule="schedule.01.south.txt"

# output
outdir=out/atm_cache
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

# in case there is a problem, we'll have more information
export TOAST_LOGLEVEL=DEBUG

# Time out the program before Slurm does
timeout 115m \
    srun ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --config sat.toml \
    --schedule $schedule \
    --scan_map.disable \
    --det_pointing_azel.shared_flag_mask 0 \
    --det_pointing_radec.shared_flag_mask 0 \
    --sim_atmosphere.enable \
    --sim_atmosphere.shared_flag_mask 0 \
    --sim_atmosphere.det_flag_mask 0 \
    --sim_atmosphere.cache_only \
    --sim_atmosphere.field_of_view "Quantity('40 deg')" \
    --out $outdir \
    >>$logfile 2>&1

# and requeue it
if [[ $? == 124 ]]; then
    scontrol requeue $SLURM_JOB_ID
fi

echo "$(date) : Done!"
