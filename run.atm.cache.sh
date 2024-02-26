#!/bin/bash
# Light version that can run on a laptop

export OMP_NUM_THREADS=4
ntask=2

freq=90
nside=512
telescope=SAT1
band=$(printf "SAT_f%03i" $freq)

schedule="schedule.small.txt"

outdir=out/atm
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

TOAST_LOGLEVEL="DEBUG" \
mpirun -np $ntask so_sim_mappraiser.py \
    --thinfp 64 \
    --config sat.toml \
    --schedule $schedule \
    --bands $band \
    --telescope $telescope \
    --sample_rate 37 \
    --scan_map.disable \
    --det_pointing_azel.shared_flag_mask 0 \
    --det_pointing_radec.shared_flag_mask 0 \
    --sim_atmosphere.shared_flag_mask 0 \
    --sim_atmosphere.det_flag_mask 0 \
    --sim_atmosphere_coarse.shared_flag_mask 0 \
    --sim_atmosphere_coarse.det_flag_mask 0 \
    --sim_atmosphere.cache_only \
    --out $outdir \
    --sim_atmosphere_coarse.disable \
    >$logfile 2>&1
