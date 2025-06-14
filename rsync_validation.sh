#!/bin/bash

# Pull plots and spectra from Jean-Zay
rsync -ravuLPh \
    --exclude 'var_noise_model_data' \
    --exclude 'noise_psd_fit' \
    --exclude 'plots' \
    jean-zay:/lustre/fsn1/projects/rech/nih/usl22vm/pairdiff_runs/validation/ \
    jz_validation
