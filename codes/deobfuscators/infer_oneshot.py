import os
import shutil

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
import time

import torch
import logging
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, HfArgumentParser, TrainingArguments, \
    BitsAndBytesConfig

from codes.build_dataset.data_io import read_dataset, save_solution
from codes.deobfuscators.utils import chunks
from codes.Config import ScriptArguments, format_prompt, format_prompt_notrain

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def generate_response(
        prompts,
        tokenizer,
        model,
        max_length=1024*6,
        max_new_tokens=1024*2
):
    # print(prompts)
    input_ids = tokenizer(prompts, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        top_k=1,
        top_p=1,
        temperature=0.1,
        #num_return_sequences=1,
        #use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            generation_config=generation_config,
            # return_dict_in_generate=True,
            # output_scores=True,
        )
    
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    outputs = [output[len(prompt):] for output, prompt in zip(decoded_outputs, prompts)]
    return decoded_outputs, outputs


def evaluate(model, tokenizer, dataset, save_dir):
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    print(f"Total Num: {len(dataset)}")
    start_time = time.time()
    for chunk in tqdm(list(chunks(dataset, training_args.per_device_eval_batch_size))):
        prompt_text = [format_prompt_notrain(sample, oneshot_prompting=script_args.oneshot_prompting, path=save_dir) for sample in chunk]
        raw_outputs, outputs = generate_response(prompt_text, tokenizer, model)
        for sample, raw_output, output in zip(chunk, raw_outputs, outputs):
            sample['raw_deobfuscated'] = raw_output
            sample['deobfuscated'] = output

        save_solution(dataset, save_dir)
    elapsed_time = time.time() - start_time
    logging.info(f"average time: {round(elapsed_time / len(dataset), 4)}s/it")
    logging.info(f"save to {save_dir}")


def main(script_args, training_args):
    logging.info(f"read from {script_args.data_path}")

    huggingface_dataset = load_from_disk(script_args.data_path)["test"]
    df_raw = huggingface_dataset.to_pandas()

    df_selected = df_raw[df_raw["task_type"] == "deobfuscate"]

    # Convert the test_case column of ndarray to nested lists
    df_selected['test_case'] = df_selected['test_case'].apply(list)
    df_selected['test_case'] = df_selected['test_case'].apply(lambda list_of_ndarray:
                                                    [list(ndarray_ele) for ndarray_ele in list_of_ndarray])

    # obfuscation_types = ["name-obfuscation", "code-compact", "self-defending", "control-flow-flattening",
    #                      "string-obfuscation", "deadcode-injection", "debug-protection"]

    obfuscation_types = ["debug-protection"]

    df_split = [df_selected[df_selected["obfuscation_type"] == ob_type] for ob_type in obfuscation_types]
    datasets = [df.to_dict('records') for df in df_split]

    # if os.path.exists(training_args.output_dir) and os.path.isdir(training_args.output_dir):
    #     shutil.rmtree(training_args.output_dir)

    if not os.path.exists(os.path.dirname(training_args.output_dir)):
        os.makedirs(os.path.dirname(training_args.output_dir), exist_ok=True)

    save_dirs = [f"{training_args.output_dir}/inference_{ob_type}.jsonl" for ob_type in obfuscation_types]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.eval()

    for dataset, save_dir in zip(datasets, save_dirs):
        evaluate(model, tokenizer, dataset, save_dir)


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
