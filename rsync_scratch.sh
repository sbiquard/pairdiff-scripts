# Copy stuff from scratch except .h5 (noise models), .png and .npz (PSD) files
rsync -ravuLPh \
    --exclude '*.npz' \
    --exclude '*.png' \
    --exclude '*.h5' \
    /lustre/fsn1/projects/rech/nih/usl22vm/pairdiff_runs/ \
    /lustre/fswork/projects/rech/nih/usl22vm/out
