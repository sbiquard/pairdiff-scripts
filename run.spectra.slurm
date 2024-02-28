#!/bin/bash
#SBATCH --job-name=pdiff-spectra
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

cd ${SLURM_SUBMIT_DIR}

module purge
module load python
conda deactivate
conda activate namaster

date
srun python compute_spectra.py -v -n 40
echo "$(date) : Done!"
