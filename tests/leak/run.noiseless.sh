#!/bin/bash

export OMP_NUM_THREADS=8
ntask=2

schedule="schedules/schedule.opti.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"
# cmb_input="qu_nside0512.fits"
# cmb_input="0064.fits"

outdir=out/leak/noiseless
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --variable_model.scatter 0 \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/ml \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --variable_model.scatter 0 \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --mappraiser.pair_diff \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/pd \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \

outdir=out/leak/noiseless_nohwp
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_hwp.par) \
    --variable_model.scatter 0 \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/ml \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_hwp.par) \
    --variable_model.scatter 0 \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --mappraiser.pair_diff \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --out $outdir/pd \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \

echo "$(date) : Done!"
