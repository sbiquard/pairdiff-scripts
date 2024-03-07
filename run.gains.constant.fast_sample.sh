#!/bin/bash
# Light version that can run on a laptop

export OMP_NUM_THREADS=4
ntask=2

freq=90
nside=512
telescope=SAT1
band=$(printf "SAT_f%03i" $freq)

schedule="schedule.small.2.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=out/gains/constant/slow_sample
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

mpirun -np $ntask ./toast_mappraiser_workflow.py \
    --thinfp 64 \
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
    --mappraiser.downscale 3000 \
    --mappraiser.pair_diff \
    --mappraiser.nperseg_frac 0.1 \
    --my_gainscrambler.enable \
    --my_gainscrambler.process_pairs \
    --my_gainscrambler.constant \
    --my_gainscrambler.sigma 0.1 \
    --sim_atmosphere_coarse.disable \
    --out $outdir \
    >$logfile 2>&1

outdir=out/gains/constant/fast_sample
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"

mpirun -np $ntask ./toast_mappraiser_workflow.py \
    --thinfp 64 \
    --config sat.toml \
    --schedule $schedule \
    --bands $band \
    --telescope $telescope \
    --sample_rate 74 \
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
    --mappraiser.downscale 3000 \
    --mappraiser.pair_diff \
    --mappraiser.nperseg_frac 0.1 \
    --my_gainscrambler.enable \
    --my_gainscrambler.process_pairs \
    --my_gainscrambler.constant \
    --my_gainscrambler.sigma 0.1 \
    --sim_atmosphere_coarse.disable \
    --out $outdir \
    >$logfile 2>&1
