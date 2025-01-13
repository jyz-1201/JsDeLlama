import datetime
import os
import shutil

from accelerate import PartialState, Accelerator
from accelerate.utils import gather_object

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
import time

import torch
import logging
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, HfArgumentParser, TrainingArguments, \
    BitsAndBytesConfig, pipeline

from codes.build_dataset.data_io import read_dataset, save_solution
from codes.build_dataset.dataset_preprocess import obfuscation_types
from codes.models.AugmentLLM import AugmentLLM
from codes.Config import ScriptArguments

torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*2))


def evaluate(augment_LLM, dataset, save_dir):
    if augment_LLM.distributed_state.is_main_process and augment_LLM.distributed_state.is_local_main_process:
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        print(f"Total Num: {len(dataset)}")

    augment_LLM.distributed_state.wait_for_everyone()
    start_time = time.time()

    generations_per_process = []

    # We automatically split the batched data we passed to it across all the processes. We also set apply_padding=True
    # so that the GPUs will have the same number of prompts, and you can then gather the results.
    with augment_LLM.distributed_state.split_between_processes(dataset) as dataset_on_process:
        dataset_with_generations = augment_LLM.generate(dataset_on_process)
        generations_per_process.extend(dataset_with_generations)

    # We are gathering string, so we need to use gather_object.
    # If you need to gather tensors, you can use gather from accelerate.utils
    generations_gather = gather_object(generations_per_process)

    # Remove duplicated from list
    generations_deduplicated = [i for n, i in enumerate(generations_gather)
                                if i not in generations_gather[:n]]

    if augment_LLM.distributed_state.is_main_process and augment_LLM.distributed_state.is_local_main_process:
        elapsed_time = time.time() - start_time
        print(f"Average time: {round(elapsed_time / len(generations_deduplicated), 2)}s/it")

        save_solution(generations_deduplicated, save_dir)
        print(f"Saved to {save_dir}")


def main(script_args, training_args):
    print(f"read from {script_args.data_path}")

    huggingface_dataset = load_from_disk(script_args.data_path)["test"]
    df_raw = huggingface_dataset.to_pandas()

    df_selected = df_raw[df_raw["task_type"] == "deobfuscate"]

    # Convert the test_case column of ndarray to nested lists
    df_selected['test_case'] = df_selected['test_case'].apply(list)
    df_selected['test_case'] = df_selected['test_case'].apply(lambda list_of_ndarray:
                                                    [list(ndarray_ele) for ndarray_ele in list_of_ndarray])

    df_split = [df_selected[df_selected["obfuscation_type"] == ob_type] for ob_type in obfuscation_types]
    datasets = [df.to_dict('records') for df in df_split]

    exp_name_component = "" if script_args.infer_exp_name == "" else f"_{script_args.infer_exp_name}"

    save_dirs = [f"{training_args.output_dir}/inference{exp_name_component}_{ob_type}.jsonl" for ob_type in obfuscation_types]

    load_model_dir = script_args.model_name_or_path if script_args.untrained else f"{training_args.output_dir}"

    tokenizer = AutoTokenizer.from_pretrained(load_model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()
    # distributed_state = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        load_model_dir,
        low_cpu_mem_usage=True,
        device_map=distributed_state.device,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        use_flash_attention_2=script_args.use_flash_attention,
    )
    model.eval()
    model = torch.compile(model)
    # model = distributed_state.prepare(model)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=distributed_state.device,
                    batch_size=training_args.per_device_eval_batch_size)

    augment_LLM = AugmentLLM(pipe, distributed_state, tokenizer, decoding_strategy=script_args.decoding_strategy,
                             num_beams=script_args.bestofN_num_beams)

    for dataset, save_dir in zip(datasets, save_dirs):
        evaluate(augment_LLM, dataset, save_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        script_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        script_args, training_args = parser.parse_args_into_dataclasses()

    print(f"args:{script_args, training_args}")

    main(script_args, training_args)
