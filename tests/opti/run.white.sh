#!/bin/bash

export OMP_NUM_THREADS=4
ntask=2

schedule="schedules/schedule.opti.txt"
cmb_input="ffp10_lensed_scl_100_nside0512.fits"

# -----------#
# NO SCATTER #
# -----------#

outdir=out/opti/white/no_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --variable_model.white \
    --variable_model.scatter 0 \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --mappraiser.pair_diff \
    --variable_model.white \
    --variable_model.scatter 0 \
    --out $outdir/pd \
    >$logfile 2>&1

# -------------#
# SAME SCATTER #
# -------------#

outdir=out/opti/white/same_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --variable_model.white \
    --variable_model.pairs \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --mappraiser.pair_diff \
    --variable_model.white \
    --variable_model.pairs \
    --out $outdir/pd \
    >$logfile 2>&1


# -----------------#
# OPPOSITE SCATTER #
# -----------------#

outdir=out/opti/white/opposite_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --variable_model.white \
    --variable_model.biased \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --mappraiser.pair_diff \
    --variable_model.white \
    --variable_model.biased \
    --out $outdir/pd \
    >$logfile 2>&1


# ---------------#
# RANDOM SCATTER #
# ---------------#

outdir=out/opti/white/random_scatter
mkdir -p $outdir/ml
mkdir -p $outdir/pd

# ML run
logfile=$outdir/ml/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --variable_model.white \
    --out $outdir/ml \
    >$logfile 2>&1

# PD run
logfile=$outdir/pd/run.log
echo "Writing $logfile"
mpirun -np $ntask ./so_mappraiser.py \
    $(< sat.par) \
    --thinfp 64 \
    --schedule $schedule \
    --scan_map.file $cmb_input \
    --mappraiser.lagmax 1 \
    --mappraiser.downscale 3000 \
    --mappraiser.pair_diff \
    --variable_model.white \
    --out $outdir/pd \
    >$logfile 2>&1
