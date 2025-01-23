#!/bin/bash
#SBATCH --job-name=opti-atm-atomic
#SBATCH --account=nih@cpu
#SBATCH --array=1-166        # 166 scans in the full schedule
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2  # 20% of the node
#SBATCH --cpus-per-task=4    # should be enough
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00      # each ML run takes several hours
#SBATCH --output=slurm-output/slurm-%A_%a.out
#SBATCH --error=slurm-output/slurm-%A_%a.out

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
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

# Output (sub)directory
outdir="out/opti/atm-ml-atomic"
scan_counter_padded=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
subdir="$outdir/sub_${scan_counter_padded}"
mkdir -p $subdir

# Pick the scan line corresponding to the job index
header_lines=3
header=$(head -n "$header_lines" "$schedule")
scan_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + header_lines))p" "$schedule")

# Write the subschedule
subschedule="$subdir/schedule.txt"
echo "$header" > $subschedule
printf "%s" "$scan_line" >> $subschedule

# ML run
srun ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --mappraiser.estimate_psd \
    --thinfp 4 \
    --schedule $subschedule \
    --scan_map.file $cmb_input \
    --out $subdir \
    >$subdir/run.log 2>&1

echo "$(date) : Done!"
