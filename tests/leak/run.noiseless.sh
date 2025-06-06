#!/bin/bash

export OMP_NUM_THREADS=8
ntask=2

schedule="schedules/schedule.opti.txt"
sky="ffp10_lensed_scl_100_nside0512.fits"
# sky="nside0128.fits"
nside=512

outdir=out/leak/noiseless/incl
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --pixels_healpix_radec.nside $nside \
    --mappraiser.nside $nside \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --out $outdir \
    >$logfile 2>&1

outdir=out/leak/noiseless/excl
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_scatter.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --pixels_healpix_radec.nside $nside \
    --mappraiser.nside $nside \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --out $outdir \
    >$logfile 2>&1

outdir=out/leak/noiseless/incl_nohwp
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_flags.par) \
    $(< no_scatter.par) \
    $(< no_hwp.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --pixels_healpix_radec.nside $nside \
    --mappraiser.nside $nside \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --out $outdir \
    >$logfile 2>&1

outdir=out/leak/noiseless/excl_nohwp
mkdir -p $outdir

logfile=$outdir/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    $(< no_scatter.par) \
    $(< no_hwp.par) \
    --schedule $schedule \
    --scan_map.file $sky \
    --pixels_healpix_radec.nside $nside \
    --mappraiser.nside $nside \
    --mappraiser.binned \
    --mappraiser.zero_noise \
    --thinfp 64 \
    --out $outdir \
    >$logfile 2>&1

echo "$(date) : Done!"
echo "Plotting..."

python post/plot_maps_all.py -v -r out/leak/noiseless --sky $sky --hits-percentile 0
