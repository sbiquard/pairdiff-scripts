#!/bin/bash

export OMP_NUM_THREADS=4
ntask=2

schedule="schedules/schedule.opti.txt"
cmb_input="0064.fits"

# HWP run
outdir=out/leak/toast_noiseless
mkdir -p $outdir
logfile=$outdir/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --pixels_healpix_radec.nside 64 \
    --variable_model.scatter 0 \
    --mappraiser.disable \
    --binner.enable \
    --mapmaker.enable \
    --thinfp 64 \
    --schedule $schedule \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
