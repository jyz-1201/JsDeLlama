#!/bin/sh

#SBATCH -p a100
#SBATCH -o ./outputs/mul_inference_codellama_onetask_2e4_beamsampling_1218.out
#SBATCH --nodes=7                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:a100:2                # number of GPUs per node
#SBATCH --mem=250000
#SBATCH --time=6:00:00             # maximum execution time (HH:MM:SS)


# Set enviroment
# shellcheck disable=SC2039
source ./codes/bashes/activate_environment.sh

LOG_PATH="./outputs/mul_inference_codellama_onetask_2e4_beamsampling_1218.out"

# Set distributed launcher
export LAUNCHER="accelerate launch \
    --config_file /home/jiang.2896/config/deepspeed_zero2_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $HEAD_NODE_IP \
    --main_process_port 29500 \
    --machine_rank \$SLURM_PROCID \
    "

# Configure python scripts and script args
export SCRIPT="./codes/deobfuscators/mul_inference.py"
export SCRIPT_ARGS="configs/inference/infer_codellama_sft_onetask_2e4_beamsampling_16.json"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

# Launch python code on each node
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"

