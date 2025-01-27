# Pull plots and spectra from Jean-Zay
rsync -ravuLPh \
    --include 'opti/atm/ml/Hits_*.fits' \
    --exclude '*.fits' \
    jean-zay:/lustre/fswork/projects/rech/nih/usl22vm/repos/pairdiff-scripts/out/ \
    jz_out
