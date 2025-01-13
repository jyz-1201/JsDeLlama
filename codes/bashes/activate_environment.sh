# shellcheck disable=SC2039

source activate Sec_Research # enter conda environment
echo $CONDA_PREFIX
echo $SLURM_JOB_ID

######################
### Show logs ###
######################
GPUS=$(srun hostname | tr '\n' ' ')
GPUS=${GPUS//".cluster"/""}
echo $GPUS
echo "START TIME: $(date)"
echo "SLURM_PROCID: $SLURM_PROCID"

module load cuda/12.3
module list

nvidia-smi
which nvidia-smi


######################
### Set enviroment ###
######################
export NCCL_SOCKET_NTHREADS=8
#export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=50
export GPUS_PER_NODE=2


######################
#### Set network #####
######################
#scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1
export HEAD_NODE_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo $HEAD_NODE_IP
######################

huggingface-cli login --token your_huggingface_token_here --add-to-git-credential
