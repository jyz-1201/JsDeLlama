import os
import sys
from random import randrange

import torch

from accelerate import PartialState
from datasets import load_from_disk
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer, pipeline, set_seed, \
    HfArgumentParser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from codes.Config import cfg, format_prompt, ScriptArguments


def train(script_args, training_args):
    # Load the tokenized dataset from Hugging Face datasets
    dataset = load_from_disk(script_args.data_path)["train"]
    print(f"Dataset Size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])
    print(format_prompt(dataset[randrange(len(dataset))]))

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=script_args.use_flash_attention,
        torch_dtype=torch.float16,
        # device_map="auto",
    )

    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=script_args.peft_lora_dropout,
        r=script_args.peft_lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=script_args.max_seq_length,
            tokenizer=tokenizer,
            packing=True,
            formatting_func=format_prompt,
            args=training_args,
        )

    # Train
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save model
    trainer.save_model()

    PartialState().wait_for_everyone()


def predict(script_args, training_args, merged_model_nickname="merged_model"):
    # Load the tokenized dataset from Hugging Face datasets
    dataset = load_from_disk(script_args.data_path)["test"]
    print(f"Dataset Size: {len(dataset)}")
    sample = dataset[randrange(len(dataset))]
    prompt = format_prompt(sample, with_groundtruth=False)

    # Load finetuned LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        training_args.output_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.6, temperature=0.9)

    print(f"Input:\n{prompt}\n")
    print(
        f"Generated Response:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}\n")


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        script_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        script_args, training_args = parser.parse_args_into_dataclasses()

    print(f"args:{script_args, training_args}")

    login(token=cfg.HUGGINGFACE_TOKEN)
    set_seed(training_args.seed)

    if training_args.do_train:
        train(script_args, training_args)

    if training_args.do_predict:
        predict(script_args, training_args)

if __name__ == "__main__":
    main()
