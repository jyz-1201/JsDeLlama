#!/bin/sh
#SBATCH -p a100
#SBATCH -o ./outputs/evaluate_codellama_0112.out
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:a100:1                # number of GPUs per node
#SBATCH --mem=250000
#SBATCH --time=2:00:00             # maximum execution time (HH:MM:SS)

# conda init bash
# shellcheck disable=SC2039
source ./codes/bashes/activate_environment.sh

models_dirs=("./outputs/codellama/")
infer_exp_name="beam1"

obfuscation_type=("name-obfuscation" "code-compact" "self-defending" \
"control-flow-flattening" "string-obfuscation" "deadcode-injection" "debug-protection")

for ((i=0; i<${#models_dirs[@]}; i++))
do
  for ((j=0; j<${#obfuscation_type[@]}; j++))
  do
    python ./codes/evaluators/CNmain.py --results "${models_dirs[i]}/inference_${infer_exp_name}_${obfuscation_type[j]}.jsonl"
  done
done
