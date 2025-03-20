#!/bin/bash

export OMP_NUM_THREADS=4
ntask=2

schedule="schedules/schedule.opti.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

outdir=out/leak/gains
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --sim_atmosphere_coarse.disable \
    --gainscrambler.enable \
    --mappraiser.estimate_psd \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< atm.par) \
    --sim_atmosphere_coarse.disable \
    --gainscrambler.enable \
    --mappraiser.estimate_psd \
    --mappraiser.pair_diff \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/ml \
    >$logfile 2>&1

echo "$(date) : Done!"
