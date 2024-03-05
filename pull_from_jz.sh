# Pull plots and spectra from Jean-Zay
rsync -ravuLPh \
    --include 'baseline/Hits_*.fits' \
    --include 'ml/Hits_*.fits' \
    --exclude '*.fits' \
    jean-zay:/gpfsscratch/rech/nih/usl22vm/pairdiff/out/ \
    jz_out
