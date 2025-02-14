#!/bin/bash

export OMP_NUM_THREADS=4
ntask=2

schedule="schedules/schedule.small.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outroot="out/atomic"

# Loop through each scan line
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
    subdir="$outroot/sub_${scan_counter_padded}"
    mkdir -p "$subdir"

    # Write the subschedule
    subschedule="$subdir/schedule.txt"
    echo "$header" > "$subschedule"
    printf "%s" "$scan_line" >> "$subschedule"

    # ML run
    logfile=$subdir/run.log
    echo "Writing $logfile"
    mpirun -np $ntask ./so_mappraiser.py \
        $(< sat.par) \
        --thinfp 64 \
        --schedule $schedule \
        --scan_map.file $cmb_input \
        --mappraiser.lagmax 1 \
        --mappraiser.downscale 3000 \
        --variable_model.use_white \
        --variable_model.uniform \
        --out $subdir \
        >$logfile 2>&1 \
        </dev/null  # mpirun reads from stdin, but we don't want to read the whole schedule

    # Increment the scan counter
    scan_counter=$((scan_counter + 1))
done <<< "$scan_lines"
