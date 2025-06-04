#!/bin/bash

export OMP_NUM_THREADS=8
ntask=2

schedule="schedules/schedule.opti.txt"
sky="ffp10_lensed_scl_100_nside0512.fits"
# sky="qu_nside0512.fits"
# sky="0064.fits"

outdir=out/leak/noiseless
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_scatter.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --out $outdir/ml \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \
    # --full_pointing \
    # --save_hdf5.enable \
    # --save_hdf5.detdata "['signal', 'flags', 'pixels', 'weights']" \
    # --save_hdf5.shared "['flags', 'times']" \

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_scatter.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --mappraiser.pair_diff \
    --thinfp 64 \
    --out $outdir/pd \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \

outdir=out/leak/noiseless_noflags
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --out $outdir/ml \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \
    # --full_pointing \
    # --save_hdf5.enable \
    # --save_hdf5.detdata "['signal', 'flags', 'pixels', 'weights']" \
    # --save_hdf5.shared "['flags', 'times']" \

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --mappraiser.pair_diff \
    --thinfp 64 \
    --out $outdir/pd \
    >$logfile 2>&1
    # --pixels_healpix_radec.nside 64 \
    # --mappraiser.nside 64 \

echo "$(date) : Done!"
echo "Plotting..."

python post/plot_maps_all.py -v -r out/leak --sky $sky
