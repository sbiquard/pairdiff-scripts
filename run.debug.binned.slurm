#!/bin/bash
#SBATCH --job-name=pdiff-debug-binned
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=35
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

# simulation parameters
freq=90
nside=512
telescope=SAT1
band=$(printf "SAT_f%03i" $freq)

schedule="schedule.01.south.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

sigma=0.1
outdir=out/gains/constant/e-01-binned
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"

srun so_sim_mappraiser.py \
    --thinfp 4 \
    --config sat.toml \
    --schedule $schedule \
    --bands $band \
    --telescope $telescope \
    --sample_rate 37 \
    --scan_map.file $cmb_input \
    --det_pointing_azel.shared_flag_mask 0 \
    --det_pointing_radec.shared_flag_mask 0 \
    --sim_atmosphere.shared_flag_mask 0 \
    --sim_atmosphere.det_flag_mask 0 \
    --sim_atmosphere_coarse.shared_flag_mask 0 \
    --sim_atmosphere_coarse.det_flag_mask 0 \
    --mappraiser.det_flag_mask 0 \
    --mappraiser.shared_flag_mask 0 \
    --mappraiser.enable \
    --mappraiser.mem_report \
    --mappraiser.pair_diff \
    --mappraiser.bandwidth 1 \
    --my_gainscrambler.enable \
    --my_gainscrambler.process_pairs \
    --my_gainscrambler.constant \
    --my_gainscrambler.sigma $sigma \
    --sim_atmosphere_coarse.disable \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
