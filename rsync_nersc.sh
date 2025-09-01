#!/bin/bash

# Pull plots and spectra from NERSC
rsync -ravuLPh \
    --exclude '*.fits' \
    --exclude 'var_noise_model_data' \
    --exclude 'noise_psd_fit' \
    perlmutter:/global/homes/s/sbiquard/pairdiff-scripts/out \
    nersc_out
