import sys
from os import path
from dataclasses import dataclass, field
from typing import Optional

basedir = path.abspath(path.dirname(__file__))
codes_dir = basedir
project_dir = path.join(basedir, "../")
sys.path.append(project_dir)  # Change to root directory of your project
sys.path.append(codes_dir)  # Change to root directory of your project


from codes.deobfuscators.oneshot_prompting import generate_oneshot_prompt, auto_judge_oneshot_example


class Config:
    OUTPUT_DIR = "./outputs"

    HUGGINGFACE_TOKEN = "your_huggingface_token_here"
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DATA_PATH = "./datasets/cached_DatasetDict"

    def __init__(self):
        basedir = path.abspath(path.dirname(__file__))
        self.codes_dir = basedir
        self.project_dir = path.join(basedir, "../")

    def append_path(self):
        sys.path.append(self.project_dir)  # Change to root directory of your project
        sys.path.append(self.codes_dir)  # Change to root directory of your project


cfg = Config()
cfg.append_path()


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default=cfg.MODEL_NAME, metadata={"help": "the model name"})
    infer_exp_name: Optional[str] = field(default="", metadata={"help": "the name of the inference experiment"})
    data_path: Optional[str] = field(default=None, metadata={"help": "Path to Huggingface pre-processed dataset"})
    # log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})

    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    max_seq_length: Optional[int] = field(default=1024*8, metadata={"help": "the maximum sequence length"})
    # max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})

    use_peft: Optional[bool] = field(default=True, metadata={"help": "Whether to use PEFT or not to train adapters"})
    use_flash_attention: Optional[bool] = field(default=False, metadata={"help": "Weather to use flash attention"})
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the dropout parameter of the LoRA adapters"})
    peft_lora_r: Optional[int] = field(default=16, metadata={"help": "the r parameter of the LoRA adapters"})

    SFT_completion_only: Optional[bool] = field(default=False, metadata={"help": "Whether to train on completion only for SFT"})

    ppo_train_epochs: Optional[int] = field(default=3, metadata={"help": "epochs to train PPO"})
    ppo_num_warmup_steps: Optional[int] = field(default=300, metadata={"help": "number of warmup steps"})
    ppo_per_device_infer_batch_size: Optional[int] = field(default=4, metadata={"help": "Number of samples per rollout"})
    ppo_save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    ppo_resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "Weather to resume PPO training from checkpoint"})
    ppo_first_epoch: Optional[int] = field(default=0, metadata={"help": "first training epoch to resume from PPO checkpoint"})
    ppo_first_step: Optional[int] = field(default=0, metadata={"help": "first training step to resume from PPO checkpoint"})

    decoding_strategy: Optional[str] = field(default="beam", metadata={"help": "decoding strategy for inference"})
    bestofN_num_beams: Optional[int] = field(default=6, metadata={"help": "number of beams in best-of-N sampling"})
    oneshot_prompting: Optional[bool] = field(default=False, metadata={"help": "Whether to prompt one-shot example for inference"})
    untrained: Optional[bool] = field(default=False, metadata={"help": "Weather to this model has been trained locally or to be loaded with huggingface-provided repo"})

def format_prompt_notrain(sample, oneshot_prompting=False, path=None):
    instruction = sample["instruction"]
    input_text = sample["input"]

    if oneshot_prompting:
        text = generate_oneshot_prompt(input_text, example=auto_judge_oneshot_example(path))
    else:
        text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
'''
    return text


def format_prompt(sample, with_groundtruth=True):
    instruction = sample["instruction"]
    input_text = sample["input"]
    response = sample["output"]

    if with_groundtruth:
        text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}
'''

    else:
        text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
'''

    return text


