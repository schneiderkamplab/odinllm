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

from .utils import Concatenator, FEATURES2PROMPT, Namespace, parse_device_map

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        padding_side="right",
        truncation_side="right",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        device_map=args.device_map,
        torch_dtype=torch.float16,
    )

    if os.path.isfile(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = datasets.load_dataset(args.dataset, split="train")
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
    if args.concatenate:
        dataset = dataset.map(Concatenator(chunk_size=model.base_model.config.max_position_embeddings), batched=True)

    eval_prompt = """
    Summarize this dialog:
    A: Hi Tom, are you busy tomorrow’s afternoon?
    B: I’m pretty sure I am. What’s up?
    A: Can you go with me to the animal shelter?.
    B: What do you want to do?
    A: I want to get a puppy for my son.
    B: That will make him so happy.
    A: Yeah, we’ve discussed it many times. I think he’s ready now.
    B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
    A: I'll get him one of those little dogs.
    B: One that won't grow up too big;-)
    A: And eat too much;-))
    B: Do you know which one he would like?
    A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
    B: I bet you had to drag him away.
    A: He wanted to take it home right away ;-).
    B: I wonder what he'll name it.
    A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
    ---
    Summary:
    """

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

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

    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
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
        logging_steps=1,
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

    model.save_pretrained(args.lora_model)

