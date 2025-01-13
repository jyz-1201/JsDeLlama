#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2                # number of GPUs per node
#SBATCH -p compute
#SBATCH --time 18:00:00
#SBATCH -o ./outputs/inference_codellama_onetask_0724.out
#SBATCH --mail-user=jiang.2896@osu.edu
#SBATCH --mail-type=ALL

# conda init bash
source ./codes/bashes/activate_environment.sh

python codes/deobfuscators/infer_llama_old.py configs/inference/infer_codellama_sft_onetask_old.json
