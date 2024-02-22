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
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=out/test
mkdir -p $outdir
ref=run0
logfile=$outdir/$ref.log
echo "Writing $logfile"

mpirun -np $ntask so_sim_mappraiser.py \
    --ref \
    --thinfp 64 \
    --config sat.toml \
    --schedule $schedule \
    --bands $band \
    --telescope $telescope \
    --sample_rate 37 \
    --scan_map.file $cmb_input \
    --pixels_healpix_radec.nside 512 \
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
    --mappraiser.downscale 30 \
    --mappraiser.pair_diff \
    --my_gainscrambler.enable \
    --my_gainscrambler.center 1.01 \
    --my_gainscrambler.sigma 0.0 \
    --my_gainscrambler.pattern ".*B" \
    --out $outdir \
    --sim_atmosphere.disable \
    --sim_atmosphere_coarse.disable \
    >&$logfile
