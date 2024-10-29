#!/bin/bash
#SBATCH --job-name=pdiff-baseline  # job name
#SBATCH --account=nih@cpu          # use CPU allocation
#SBATCH --qos=qos_cpu-dev          # dev qos (10 jobs, 2h max.)
#SBATCH --nodes=35                 # number of nodes
#SBATCH --ntasks-per-node=10       # number of MPI tasks per node
#SBATCH --cpus-per-task=4          # number of cores per task
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --time=00:30:00            # default: 10 minutes on cpu_p1

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

outdir=out/baseline
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

srun ./toast_so_mappraiser.py \
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
    --sim_atmosphere_coarse.disable \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
