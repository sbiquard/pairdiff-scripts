#!/bin/bash
# Light version that can run on a laptop

ulimit -n 4096
export OMP_NUM_THREADS=1
ntask=4

freq=90
nside=512
telescope=SAT1
band=$(printf "SAT_f%03i" $freq)

schedule="schedule.small.txt"
# cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=out/atm
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

mpirun -np $ntask so_sim_mappraiser.py \
    --thinfp 64 \
    --config sat.toml \
    --schedule $schedule \
    --bands $band \
    --telescope $telescope \
    --sample_rate 37 \
    --scan_map.disable \
    --pixels_healpix_radec.nside 512 \
    --det_pointing_azel.shared_flag_mask 0 \
    --det_pointing_radec.shared_flag_mask 0 \
    --sim_atmosphere.shared_flag_mask 0 \
    --sim_atmosphere.det_flag_mask 0 \
    --sim_atmosphere_coarse.shared_flag_mask 0 \
    --sim_atmosphere_coarse.det_flag_mask 0 \
    --sim_atmosphere.cache_only \
    --out $outdir \
    >&$logfile
