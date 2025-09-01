# Installation

## Perlmutter

Installation prefix:
```bash
export SOFTDIR="/global/common/software/mp107a/sbiquard/pairdiff"
```

### TOAST

Using `$HOME/explicit_pairdiff_env.txt`, recreate the conda environment:
```bash
micromamba env create -n pairdiff --file explicit_pairdiff_env.txt
```

Then navigate to the `TOAST` root.
```bash
mkdir build; cd build
../platforms/conda_dev.sh -DCMAKE_INSTALL_PREFIX=$SOFTDIR
make -j 8; make install
```

#### mpi4py

This one must be executed **without conda environment loaded** but referencing the location of the Python binary:
```bash
MPICC="cc -shared" uv pip install --python $SOFTDIR/../micromamba/envs/pairdiff/bin/python --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

### sotodlib

Navigate to `sotodlib` root and simply run:
```bash
uv pip install .
```

### MIDAPACK/MAPPRAISER

Load required modules:
```bash
module load cray-fftw cfitsio
```

Then as usual:
```bash
cmake -S . -B build --install-prefix=$SOFTDIR -DPYTHON_MAJORMINOR=3.10
cmake --build build; cmake --install build
```
## Jean-Zay

TODO.
