#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=16gb
#SBATCH --job-name="sample"
#SBATCH --output=../outfiles/sample

module purge
module load cuda/11.6.2/gcc
. "/home/lherron/scratch.tiwary-prj/miniconda/etc/profile.d/conda.sh"
conda activate ML

for GENTEMP in {290..420..10}; do
    python -u 'sample.py' \
        --pdbid $1 \
        --expid $2 \
        --pred_type $3 \
        --epoch $4 \
        --gen_temp $GENTEMP \
        --num_samples $5 \
        --self_condition $6 \
        --sys_config_path $7 \
        --exp_config_path $8
done
