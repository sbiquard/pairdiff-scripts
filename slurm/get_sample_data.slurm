#!/bin/bash
#SBATCH --job-name=get-sample-obs-data
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00

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

outdir="out/sample_obs_data"
mkdir -p $outdir
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

# observation number to simulate, from argv
obs_id=$1
full_schedule="schedules/schedule.01.south.txt"
header_lines=3
header=$(head -n "$header_lines" "$full_schedule")
obs_line=$(sed -n "$((obs_id + header_lines))p" "$full_schedule")

# write the subschedule
subschedule="$outdir/schedule.$obs_id.txt"
echo "$header" >$subschedule
printf "%s" "$obs_line" >>$subschedule

srun ./so_mappraiser.py \
    $(<sat.par) \
    $(<atm.par) \
    --simulate-only \
    --full-pointing \
    --sim_atmosphere_coarse.det_data "atm_coarse" \
    --save_hdf5.enable \
    --save_hdf5.detdata "['signal', 'atm', 'atm_coarse', 'flags', 'noise', 'pixels', 'weights']" \
    --save_hdf5.shared "['flags', 'times']" \
    --thinfp 64 \
    --schedule $subschedule \
    --out $outdir \
    >$outdir/run.log 2>&1

echo "$(date) : Done!"
