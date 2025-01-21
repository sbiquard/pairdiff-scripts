#!/bin/bash
#SBATCH --job-name=opti-atm-atomic
#SBATCH --account=nih@cpu
#SBATCH --nodes=1
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

schedule="schedules/schedule.01.south.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=out/opti/atm-ml-atomic
mkdir -p $outdir

# Loop through the scan lines
header_lines=3
header=$(head -n "$header_lines" "$schedule")
scan_lines=$(tail -n +$((header_lines + 1)) "$schedule")
line_count=$(echo "$scan_lines" | wc -l)
scan_counter=1

while IFS= read -r scan_line; do
    # Print progress
    echo "Processing scan $scan_counter of $line_count"

    # Output subdirectory for the current scan
    scan_counter_padded=$(printf "%04d" "$scan_counter")
    subdir="$outdir/sub_${scan_counter_padded}"
    mkdir -p $subdir

    # Write the subschedule
    subschedule="$subdir/schedule.txt"
    echo "$header" > "$subschedule"
    printf "%s" "$scan_line" >> "$subschedule"

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
        </dev/null

    # Increment the scan counter
    scan_counter=$((scan_counter + 1))
done <<< "$scan_lines"

echo "$(date) : Done!"
