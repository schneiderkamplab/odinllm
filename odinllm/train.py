import click
from datasets import load_dataset
import os
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from .shared import load_model, load_tokenizer, save_metadata
from .utils import chars_token_ratio, end, get_prepare_sample_text, parse_args, start, status, trainable_parameters

def get_peft_config(model, target_modules):
    start("Target modules")
    if not target_modules:
        target_modules = set()
        for name, module in model.named_modules():
            if type(module).__name__ in ("Linear", "Linear4bit"):
                target_modules.add(name.split(".")[-1])
        target_modules = list(target_modules)
    status(target_modules)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return config

def do_train(trainer, output_dir):
    start("Supervised fine tuning")
    trainable_params, all_params = trainable_parameters(trainer.model)
    status(f"#trainable-params: {trainable_params}; #all-params: {all_params}; %trainable: {100 * trainable_params / all_params}", end='')
    trainer.train()
    trainer.save_model(output_dir)
    end()

def get_training_args(output_dir, qualifier, max_steps, per_device_train_batch_size, gradient_accumulation_steps, num_train_epochs):
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=True,
        eval_steps=1000,
        do_eval=True,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        group_by_length=False,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        optim="paged_adamw_32bit",
        bf16=True,
        remove_unused_columns=False,
        run_name=f"{qualifier}_{output_dir}",
        report_to="wandb",
    )
    return training_arguments    

def load_datasets(tokenizer, dataset_name, split, num_workers, streaming, size_valid_set, shuffle_buffer, seq_length):
    start("Loading dataset from", dataset_name)
    if os.path.isfile(dataset_name):
        dataset = load_dataset(
            "json",
            data_files=dataset_name,
            split=split,
            num_proc=num_workers if not streaming else None,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            num_proc=num_workers if not streaming else None,
            streaming=streaming,
        )
    if streaming:
        status("Loading the dataset in streaming mode", end='')
        valid_data = dataset.take(size_valid_set)
        train_data = dataset.skip(size_valid_set)
        train_data = train_data.shuffle(buffer_size=shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=100, seed=42)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        status(f"#train: {len(train_data)}; #eval: {len(valid_data)}", end='')
    end()
    start("Preprocessing the dataset")
    prepare_sample_text = get_prepare_sample_text(tokenizer)
    chars_per_token = chars_token_ratio(train_data, tokenizer, prepare_sample_text=prepare_sample_text)
    status(f"{chars_per_token:.2f} chars/token", end='')
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    end()
    return train_dataset, valid_dataset

@click.group()
def _train():
    pass
@_train.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=False))
@click.option("--peft/--no-peft", default=True)
@click.option("--max-steps", "-s", default=-1, type=int)
@click.option("--gradient-accumulation-steps", "-g", default=1, type=int)
@click.option("--per-device-train-batch-size", "-b", default=1, type=int)
@click.option("--dataset", "-d", default="samsum", type=str)
@click.option("--packing/--no-packing", default=True)
@click.option("--load-in-4bit/--no-load-in-4bit", default=True)
@click.option("--device-map", "-m", default="auto")
@click.option("--split", default="train")
@click.option("--num_workers", default=4, type=int)
@click.option("--streaming/--no-streaming", default=False)
@click.option("--size-valid-set", default=4000, type=int)
@click.option("--shuffle-buffer", default=5000, type=int)
@click.option("--seq-length", default=1024, type=int)
@click.option("--target-modules", default="['q_proj', 'v_proj']", type=str)
@click.option("--num-train-epochs", default=3, type=int)
@parse_args
def train(args):
    if not args.peft and args.load_in_4bit:
        status("--no-peft implies --no-load-in-4bit")
        args.load_in_4bit = False
    model = load_model(
        args.pretrained_model,
        device_map=args.device_map,
        qualifier="pretrained model",
        load_in_4bit=args.load_in_4bit,
    )
    peft_config = get_peft_config(model, args.target_modules) if args.peft else None
    tokenizer = load_tokenizer(args.pretrained_model)
    training_args = get_training_args(
        output_dir=args.output_dir,
        qualifier="train",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
    )
    train_dataset, eval_dataset = load_datasets(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        split=args.split,
        num_workers=args.num_workers,
        streaming=args.streaming,
        size_valid_set=args.size_valid_set,
        shuffle_buffer=args.shuffle_buffer,
        seq_length=args.seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=args.packing,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    do_train(
        trainer,
        output_dir=args.output_dir,
    )
    save_metadata(model.metadata, args, args.output_dir)
