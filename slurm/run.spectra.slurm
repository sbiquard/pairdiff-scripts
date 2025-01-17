#!/bin/bash
#SBATCH --job-name=pdiff-spectra
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00

cd ${SLURM_SUBMIT_DIR}

module purge
module load python; conda deactivate
conda activate $SCRATCH/envs/namaster

date
ncpu=$SLURM_CPUS_PER_TASK
srun python compute_spectra.py --verbose -n $ncpu --plot --mask mask_apo_1000 --ref hits_1000
# srun python compute_spectra.py --verbose -n $ncpu --plot --mask mask_apo_20deg_1000 --ref hits_1000_apo_20
# srun python compute_spectra.py --verbose -n $ncpu --plot --mask mask_apo_10000 --ref hits_10000
# srun python compute_spectra.py --verbose -n $ncpu --plot --mask mask_apo_20deg_10000 --ref hits_10000_apo_20
echo "$(date) : Done!"
