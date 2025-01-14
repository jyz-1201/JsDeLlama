# JsDeLlama-replication-package
This repository contains source code that we used to perform experiment in paper titled "JsDeLlama: Learning Executability, Simplicity, and Human-readability for Versatile JavaScript Deobfuscation".

Please follow the steps below to reproduce the result.

## Machine Requirement
A GPU cluster with Slurm workload manager is recommended to smoothly reproduce our experiment. In our paper, the experiments are conducted on a linux computer cluster using 7 nodes with 2 NVIDIA A100 GPUs on each node with CUDA 12.3.

## Environment Dependence

### Pypi Deps
Python and Anaconda installations are suggested to set up the environment. The required packages are stored in `./requirements.yml`, which can be automatically installed using anaconda. 

Run the following command in terminal to prepare virtual environment. The required version of python and packages will be installed and activated.
```bash
conda env create --file requirements.yml
conda activate Sec_Research
```

### Tool Deps
```
git
nodejs
npm
apptainer
```

### Obfuscator Deps (Needed for Obfuscating Dataset)

`npm install --global javascript-obfuscator`

### Deobfuscators (Baselines) Deps

In directory `deobfuscators`: 

Install javascript-deobfuscator (JS-deobfuscator):

`git clone https://github.com/ben-sb/javascript-deobfuscator.git`

`ln -s <current_path>/deobfuscators/javascript-deobfuscator/dist/run.js ~/.local/bin/js-deobfuscator` 

also, make sure `~/.local/bin/js-deobfuscator` is in your PATH.

Install Synchrony:

`npm install --global deobfuscator`


### Escomplex (Needed for Evaluation)

In directory `evaluators`: 

`git clone https://github.com/escomplex/escomplex.git`

`ln -s <current_path>/evaluators/escomplex/src/cli.js ~/.local/bin/escomplex` 

## Prepare Obfuscated Dataset

Unzip the `dataset.7z` stored in `build_dataset` directory, you will have the test dataset obfuscated with 7 individual transformations. 

`7z x dataset.7z`

## Supervised Fine-tuning (SFT)
Before running any scripts to train and inference with your model, first put your huggingface token to variable `HUGGINGFACE_TOKEN` in `codes/Config.py`, and also the last line of file `codes\bashes\activate_environment.sh` for access of llama series model.

Run the following command to submit a batch job to Slurm for SFT:
`sbatch codes/bashes/multinode_SFT/multinode_SFT_codellama_onetask_2e4_16epoch.sh`

This will invoke python script `./codes/trainers/instruction_tuning.py` with arguments pre-specified in `configs/instruction_tuning/sft_codellama_onetask_2e4_16epoch.json` and save the trained model into `./outputs/instruction_tuning_codellama_onetask_2e4_16epoch`.

## RL Alignment
Run the following command to submit a batch job to Slurm for RL alignment:
`sbatch codes/bashes/multinode_RL/multinode_RL_codellama_onetask_16epoch.sh`

This will invoke python script `codes/trainers/ppo.py` with arguments pre-specified in `configs/ppo/ppo_codellama_onetask_16epoch.json` and save the trained model into `./outputs/multinode_RL_codellama_onetask_2e4_1003`.

After completion, run `tensorboard --logdir ./outputs` and select the best model based on your observed highest training reward and copy its checkpoint into `./outputs/multinode_RL_codellama_onetask_2e4_1003/best_checkpoint/`.

For experiment with ablation studies without RL, this step should be skipped.

## Deobfuscation with Best-of-N Sampling

Run the following command to submit a batch job to Slurm for deobfuscation with best-of-N sampling:
`codes/bashes/multinode_inference/mul_infer_codellama_onetask_2e4_rl_beamsampling_16.sh`

This will invoke python script `./codes/deobfuscators/mul_inference.py` with arguments pre-specified in `configs/inference/infer_codellama_sft_onetask_2e4_rl_beamsampling_16.json` and save the deobfuscation results into `./outputs/multinode_RL_codellama_onetask_2e4_1003/best_checkpoint/trained_model/`.

For experiment with ablation studies without best-of-N sampling, this step should be skipped.

## Evaluation

Run the following command to submit batch jobs to Slurm for evaluation:
`codes/bashes/evaluate_codellama_onetask.sh`

This will invoke python script `./codes/evaluators/CNmain.py` for certain obfuscation techniques and specified models and experiments names in the .sh file and print the results into its output logs. The detailed metrics file will be stored in the same directory, adding the suffix ".metric".

## Baselines

Implementations of baselines can be found in `codes/deobfuscators/deobfuscate_codenet.py` and `codes/deobfuscators/infer_oneshot.py`.


## Acks

We thanks to the following projects:

https://github.com/javascript-obfuscator/javascript-obfuscator

https://github.com/escomplex/escomplex

https://github.com/ben-sb/obfuscator-io-deobfuscator

https://github.com/ben-sb/javascript-deobfuscator

https://github.com/relative/synchrony
