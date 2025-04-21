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

# namaster environment
module purge
module load python
conda deactivate
conda activate $SCRATCH/micromamba/envs/namaster

date
ncpu=$SLURM_CPUS_PER_TASK
root=$1
echo "Running with $ncpu threads in $root"

for hits in 1000 10000; do
    for apo in 10deg 20deg; do
        if [[ "$apo" == "10deg" ]]; then
            mask="mask_apo_${hits}"
            ref="hits_${hits}"
        else
            mask="mask_apo_${apo}_${hits}"
            ref="hits_${hits}_apo_20"
        fi
        srun python post/compute_spectra.py --verbose -n $ncpu --plot --mask $mask --ref $ref --root $root
    done
done

echo "$(date) : Done!"
