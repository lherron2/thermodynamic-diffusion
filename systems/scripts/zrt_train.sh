#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=16gb
#SBATCH --job-name="train"
#SBATCH --output=../outfiles/5udz_x0_mid_attn_sc.out

module purge
module load cuda/11.6.2/gcc
. "/home/lherron/scratch.tiwary-prj/miniconda/etc/profile.d/conda.sh"
conda activate ML

python -u 'train.py' \
    --pdbid $1 \
    --expid $2 \
    --pred_type $3 \
    --self_condition $4 \
    --sys_config_path $5 \
    --exp_config_path $6


