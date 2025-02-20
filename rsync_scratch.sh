#!/bin/bash

# Copy stuff from scratch except noise PSD fits, noise model data and old stuff
rsync -ravuLPh \
    --exclude 'noise_psd_fit' \
    --exclude 'var_noise_model_data' \
    --exclude 'before_psd_fix' \
    /lustre/fsn1/projects/rech/nih/usl22vm/pairdiff_runs/ \
    /lustre/fswork/projects/rech/nih/usl22vm/repos/pairdiff-scripts/out
