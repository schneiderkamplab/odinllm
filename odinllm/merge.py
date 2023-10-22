import click
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .shared import load_lora, load_model, load_tokenizer, merge_model, save_model, save_tokenizer
from .utils import end, parse_args, start

@click.group()
def _merge():
    pass
@_merge.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("lora-model", type=click.Path(exists=True))
@click.argument("merged-model", type=click.Path(exists=False))
@click.option("--device-map", "-m", default="auto")
@parse_args
def merge(args):
    model = load_model(args.pretrained_model)
    model = load_lora(args.lora_model)
    model = merge_model(model)
    save_model(model, args.merged_model, qualifier="merged model")
    tokenizer = load_tokenizer(args.pretrained_model)
    save_tokenizer(tokenizer, args.merged_model)
