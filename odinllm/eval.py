import click
from datasets import load_dataset
import os
from peft import LoraConfig
from transformers import EarlyStoppingCallback,TrainerCallback, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from .shared import load_model, load_tokenizer
from .train import load_datasets
from .utils import chars_token_ratio, end, get_prepare_sample_text, parse_args, start, status

def do_eval(trainer):
    start("Evaluating")
    print(trainer.evaluate())
    end()

def get_training_args():
    training_arguments = TrainingArguments(
        output_dir="ignore",
        per_device_eval_batch_size=1,
        bf16=True,
        remove_unused_columns=False,
        report_to=None,
    )
    return training_arguments    

@click.group()
def _eval():
    pass
@_eval.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
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
@click.option("--test-size", default=100, type=int)
@parse_args
def eval(args):
    model = load_model(
        args.pretrained_model,
        device_map=args.device_map,
        qualifier="pretrained model",
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = load_tokenizer(args.pretrained_model)
    training_args = get_training_args()
    train_dataset, eval_dataset = load_datasets(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        split=args.split,
        num_workers=args.num_workers,
        streaming=args.streaming,
        size_valid_set=args.size_valid_set,
        shuffle_buffer=args.shuffle_buffer,
        seq_length=args.seq_length,
        test_size=args.test_size,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=args.packing,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    do_eval(trainer)
