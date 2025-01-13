import datetime
import os
import pickle
import sys
import typing
from contextlib import contextmanager, nullcontext
from random import randrange
import warnings
from typing import Optional

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='You have modified the pretrained model configuration to control generation')
warnings.filterwarnings('ignore', category=UserWarning, message='Positional args are being deprecated')

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_from_disk, Dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline, HfArgumentParser, BitsAndBytesConfig, PreTrainedTokenizerBase, \
    get_scheduler
import bitsandbytes as bnb

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOTrainer, set_seed, PPOConfig, \
    PreTrainedModelWrapper
from trl.import_utils import is_npu_available, is_xpu_available

from codes.build_dataset.data_io import save_solution
from codes.Config import cfg, ScriptArguments, format_prompt
from codes.models.AugmentLLM import RewardEvaluator
from codes.utils import find_newest_checkpoint

tqdm.pandas()

NUM_GPUs = 8
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*2))

def prepare_RL_dataset(tokenizer, original_dataset):
    """
    Build dataset for training.
    """
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 1024*6}

    def tokenize(sample):
        query = format_prompt(sample, False)
        sample["input_ids"] = tokenizer.encode(query, **tokenizer_kwargs)
        return sample

    tokenized_dataset = original_dataset.map(
        tokenize,
        batched=False,
        load_from_cache_file=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )
    tokenized_dataset.set_format(type="torch")
    return tokenized_dataset


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


class MyPPOTrainer(PPOTrainer):
    def __init__(self, config: Optional[PPOConfig] = None, model: Optional[PreTrainedModelWrapper] = None,
                 ref_model: Optional[PreTrainedModelWrapper] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 dataset: Optional[typing.Union[torch.utils.data.Dataset, Dataset]] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None, data_collator: Optional[typing.Callable] = None,
                 num_shared_layers: Optional[int] = None,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 training_data_collator: Optional[typing.Callable] = None):

        super().__init__(config, model, ref_model, tokenizer, dataset, optimizer, data_collator, num_shared_layers,
                         lr_scheduler, training_data_collator)

        self.optional_peft_ctx = (
            self.use_ref_adapter
            if self.is_peft_model
            else nullcontext
        )

    @contextmanager
    def use_ref_adapter(self):
        try:
            self.accelerator.unwrap_model(self.model).pretrained_model.set_adapter("ref_adapter")
            yield
        finally:
            self.accelerator.unwrap_model(self.model).pretrained_model.set_adapter("default")


def train_PPO(script_args, ppo_config):
    # print(f"Enter train: {Accelerator().local_process_index} {Accelerator().process_index}")

    trl_model_class = AutoModelForCausalLMWithValueHead

    # Load the tokenized dataset from Hugging Face datasets
    original_dataset = load_from_disk(script_args.data_path)["train"]

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if script_args.ppo_resume_from_checkpoint:
        script_args.ppo_first_epoch, script_args.ppo_first_step = find_newest_checkpoint(ppo_config.project_kwargs['logging_dir'], "ppo_checkpoint_")

    if script_args.ppo_first_epoch == -1:
        print(f"Checkpoint {ppo_config.project_kwargs['logging_dir']} not found, training from scratch")
        script_args.ppo_resume_from_checkpoint = False

    checkpoint_dir = f"{ppo_config.project_kwargs['logging_dir']}/ppo_checkpoint_{script_args.ppo_first_epoch}_{script_args.ppo_first_step}"
    # load_model_path = f"{checkpoint_dir}/trained_model" if script_args.ppo_resume_from_checkpoint else \
    #     f"{script_args.model_name_or_path}"

    load_model_path = script_args.model_name_or_path

    # Now let's build the model, the reference model, and the tokenizer.
    if not script_args.use_peft:
        # Load model
        ref_model = trl_model_class.from_pretrained(
            load_model_path,
            quantization_config=bnb_config,
            use_cache=False,
            use_flash_attention_2=script_args.use_flash_attention,
            torch_dtype=torch.float16
        )
        device_map = None
        peft_config = None
    else:
        peft_config = LoraConfig(
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=script_args.peft_lora_dropout,
            r=script_args.peft_lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}

    model = trl_model_class.from_pretrained(
        load_model_path,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=script_args.use_flash_attention,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        peft_config=peft_config,
        is_trainable=True,
    )

    if script_args.use_peft:
        model.pretrained_model.load_adapter(script_args.model_name_or_path, adapter_name="ref_adapter")
        model.pretrained_model.set_adapter("default")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    RL_dataset = prepare_RL_dataset(tokenizer, original_dataset)

    # Create optimizer
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=ppo_config.learning_rate)

    num_training_steps = len(original_dataset) * script_args.ppo_train_epochs / NUM_GPUs * 400

    lr_scheduler = get_scheduler(
        name="cosine",  # Type of scheduler
        optimizer=optimizer,
        num_warmup_steps=script_args.ppo_num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = MyPPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=RL_dataset, optimizer=optimizer, lr_scheduler=lr_scheduler, data_collator=collator)
    accelerator = ppo_trainer.accelerator

    if script_args.ppo_resume_from_checkpoint:
        accelerator.load_state(checkpoint_dir, load_module_strict=False)

        ppo_trainer.running.mean = torch.load(f"{checkpoint_dir}/running_mean.pkl")
        ppo_trainer.running.std = torch.load(f"{checkpoint_dir}/running_std.pkl")
        ppo_trainer.running.var = torch.load(f"{checkpoint_dir}/running_var.pkl")
        ppo_trainer.running.count = torch.load(f"{checkpoint_dir}/running_count.pkl")
        ppo_trainer.current_step = torch.load(f"{checkpoint_dir}/current_step.pkl")
        ppo_trainer.kl_ctl.value = torch.load(f"{checkpoint_dir}/kl_ctl_value.pkl")

    # We then build the reward evaluator. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            reward_evaluator = RewardEvaluator(device)
    else:
        reward_evaluator = RewardEvaluator(device)

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 650,
        # "max_new_tokens": 512,
    }

    if ppo_trainer.accelerator.is_main_process and ppo_trainer.accelerator.is_local_main_process:
        print(f"args:{script_args, ppo_config}")
        print(f"Dataset Size: {len(original_dataset)}")
        print(original_dataset[randrange(len(original_dataset))])
        model.pretrained_model.print_trainable_parameters()

    first_epoch = script_args.ppo_first_epoch if script_args.ppo_resume_from_checkpoint else 0
    for epoch in tqdm(range(first_epoch, script_args.ppo_train_epochs), "epoch: "):
        dataloader = accelerator.skip_first_batches(ppo_trainer.dataloader, script_args.ppo_first_step) \
            if epoch == first_epoch and script_args.ppo_resume_from_checkpoint else ppo_trainer.dataloader

        enum = enumerate(dataloader, script_args.ppo_first_step + 1) \
            if epoch == first_epoch and script_args.ppo_resume_from_checkpoint else enumerate(dataloader)

        for step, batch in tqdm(enum, "step: "):
            query_tensors = batch["input_ids"]

            max_length = len(max(batch["output"], key=len))
            generation_kwargs["max_new_tokens"] = int(max_length * 1.2)

            # print(f"Enter Generate: {accelerator.local_process_index} {accelerator.process_index}")

            # Get response from generation model
            response_tensors = ppo_trainer.generate(
                query_tensors, batch_size=script_args.ppo_per_device_infer_batch_size, return_prompt=False, **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # print(f"Enter compute_batch_reward: {accelerator.local_process_index} {accelerator.process_index}")

            # Compute reward score
            batch_list_of_dict = [dict(zip(batch,t)) for t in zip(*batch.values())]
            rewards = reward_evaluator.compute_batch_reward(batch_list_of_dict, batch["response"])

            # print(f"Enter Step: {accelerator.local_process_index} {accelerator.process_index}")

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            ppo_trainer.log_stats(stats, batch, rewards)

            # Save checkpoint
            if script_args.ppo_save_freq and step and step % script_args.ppo_save_freq == 0:
                new_checkpoint_dir = f"{ppo_config.project_kwargs['logging_dir']}/ppo_checkpoint_{epoch}_{step}"
                accelerator.save_state(new_checkpoint_dir)

                accelerator.save(ppo_trainer.running.mean, f"{new_checkpoint_dir}/running_mean.pkl")
                accelerator.save(ppo_trainer.running.std, f"{new_checkpoint_dir}/running_std.pkl")
                accelerator.save(ppo_trainer.running.var, f"{new_checkpoint_dir}/running_var.pkl")
                accelerator.save(ppo_trainer.running.count, f"{new_checkpoint_dir}/running_count.pkl")
                accelerator.save(ppo_trainer.current_step, f"{new_checkpoint_dir}/current_step.pkl")
                accelerator.save(ppo_trainer.kl_ctl.value, f"{new_checkpoint_dir}/kl_ctl_value.pkl")

                if ppo_trainer.accelerator.is_main_process and ppo_trainer.accelerator.is_local_main_process:
                    ppo_trainer.save_pretrained(f"{new_checkpoint_dir}/trained_model")

                # save_solution(batch_list_of_dict, f"{ppo_config.project_kwargs['logging_dir']}/ppo_rollout.jsonl")

    if ppo_trainer.accelerator.is_main_process and ppo_trainer.accelerator.is_local_main_process:
        ppo_trainer.save_pretrained(ppo_config.project_kwargs["logging_dir"])


def main():
    parser = HfArgumentParser((ScriptArguments, PPOConfig))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        script_args, ppo_config = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        script_args, ppo_config = parser.parse_args_into_dataclasses()

    # ppo_config.accelerator_kwargs["kwargs_handlers"] = [InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600*2))]

    login(token=cfg.HUGGINGFACE_TOKEN, add_to_git_credential=True)
    set_seed(ppo_config.seed)

    train_PPO(script_args, ppo_config)


if __name__ == "__main__":
    main()
