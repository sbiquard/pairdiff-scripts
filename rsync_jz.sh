#!/bin/bash

# Pull plots and spectra from Jean-Zay
rsync -ravuLPh \
    --include 'opti/atm/*/Hits_*.fits' \
    --include 'mask_apo*.fits' \
    --include 'leak/**/*.fits' \
    --exclude '*.fits' \
    jean-zay:/lustre/fswork/projects/rech/nih/usl22vm/repos/pairdiff-scripts/out/ \
    jz_out
