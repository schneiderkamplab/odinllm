import click
from contextlib import nullcontext
from dataclasses import dataclass
from datasets import load_dataset
import datasets
import os
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
)

from .utils import Concatenator, EXAMPLE_PROMPTS, FEATURES2PROMPT, Namespace, end, parse_device_map, start, status

@click.group()
def _train():
    pass
@_train.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("lora-model", type=click.Path(exists=False))
@click.option("--max-steps", "-s", default=-1, type=int)
@click.option("--gradient-accumulation-steps", "-g", default=1, type=int)
@click.option("--per-device-train-batch-size", "-b", default=1, type=int)
@click.option("--max-steps", "-s", default=-1, type=int)
@click.option("--dataset", "-d", default="samsum", type=str)
@click.option("--concatenate/--no-concatenate", default=True)
@click.option("--device-map", "-m", default="auto")
def train(**kwargs):
    args = Namespace(**kwargs)
    args.device_map = parse_device_map(args.device_map)

    start("Loading tokenizer from", args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        padding_side="right",
        truncation_side="right",
    )
    end()

    start("Loading pretrained model from", args.pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        device_map=args.device_map,
        torch_dtype=torch.float16,
    )
    end()

    start("Loading dataset from", args.dataset)
    if os.path.isfile(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = datasets.load_dataset(args.dataset, split="train")
    end()

    for key, prompt in EXAMPLE_PROMPTS:
        start(f"Testing {key} prompt with pretrained model")
        res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
        end(end='')
        status(res)

    start("Creating prompts from data")
    def apply_prompt_template(sample):
        sample_clean = {k: v for k, v in sample.items() if v.strip()}
        features = tuple(sorted(sample_clean.keys()))
        prompt = FEATURES2PROMPT.get(features, None)
        if prompt is None:
            features = tuple(sorted(sample.keys()))
            prompt = FEATURES2PROMPT.get(features, None)
            if prompt is None:
                raise RuntimeError(f"no prompt template for feature combination {features} for sample {sample}")
            sample_clean = sample
        return {
            "text": prompt.format(
                eos_token=tokenizer.eos_token,
                **sample_clean,
            )
        }
    dataset = dataset.map(
        apply_prompt_template,
        remove_columns=list(dataset.features),
    )
    end()

    start("Tokenizing data")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_kwargs = {} if args.concatenate else {
        "padding": "max_length",
        "max_length": model.base_model.config.max_position_embeddings,
        "truncation": True,
    }
    def apply_tokenizer(sample):
        sample = tokenizer(sample["text"], return_tensors="pt", **tokenizer_kwargs)
        sample["labels"] = sample["input_ids"].clone()
        return sample
    dataset = dataset.map(
        apply_tokenizer,
        batched=True,
        batch_size=1,
        remove_columns=list(dataset.features),
    )
    end()

    start("Concatenating samples")
    if args.concatenate:
        dataset = dataset.map(Concatenator(chunk_size=model.base_model.config.max_position_embeddings), batched=True)
    end()

    start("Prepare for PEFT")
    model.train()
    def create_peft_config(model):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )

        # prepare int-8 model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, peft_config

    # create peft config
    model, lora_config = create_peft_config(model)
    end()

    for key, prompt in EXAMPLE_PROMPTS:
        start(f"Testing {key} prompt with PEFT model")
        res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
        end(end='')
        status(res)

    start("Training")
    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 5,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_checkpointing': False,
    }

    profiler = nullcontext()

    # Define training args
    training_args = TrainingArguments(
        output_dir=args.lora_model,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=os.path.join(args.lora_model,"logs"),
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=args.max_steps,
        **{k:v for k, v in config.items() if k != 'lora_config'},
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
            callbacks=[],
        )

        # Start training
        trainer.train()
    end()

    start("Saving LoRA adapter to", args.lora_model)
    model.save_pretrained(args.lora_model)
    end()

    for key, prompt in EXAMPLE_PROMPTS:
        start(f"Testing {key} prompt with trained PEFT model")
        res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
        end(end='')
        status(res)
