import click
from contextlib import nullcontext
from datasets import load_dataset
import os
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import (
    default_data_collator,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .shared import apply_concatenation, apply_templates, apply_tokenization, load_dataset, load_model, load_tokenizer, run_prompt, save_lora
from .utils import Concatenator, FEATURES2PROMPT, end, parse_args, start

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
@click.option("--test/--no-test", default=True, help="Whehter to run inference with pretrained, PEFT, and trained model")
@click.option("--run-prompt", "-p", default=None, help="Prompt to run instead of example prompts")
@click.option("--load-in-4bit/--no-load-in-4bit", default=True)
@click.option("--device-map", "-m", default="auto")
@parse_args
def train(args):
    tokenizer = load_tokenizer(args.pretrained_model)
    model = load_model(
        args.pretrained_model,
        device_map=args.device_map,
        qualifier="pretrained model",
        load_in_4bit=args.load_in_4bit,
    )
    if args.test:
        print(run_prompt(model, tokenizer))
    dataset = load_dataset(args.dataset)
    dataset = apply_templates(dataset, tokenizer)
    dataset = apply_tokenization(
        dataset,
        tokenizer,
        padding_length=None if args.concatenate else model.base_model.config.max_position_embeddings,
    )
    if args.concatenate:
        dataset = apply_concatenation(dataset, chunk_length=model.base_model.config.max_position_embeddings)
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

    if args.test:
        print(run_prompt(model, tokenizer, run_prompt=args.run_prompt))

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
    end()

    save_lora(model, args.lora_model)
    if args.test:
        print(run_prompt(model, tokenizer, run_prompt=args.run_prompt))
