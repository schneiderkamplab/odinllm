import click
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import Namespace, parse_device_map

@click.group()
def _merge():
    pass
@_merge.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("lora-model", type=click.Path(exists=True))
@click.argument("merged-model", type=click.Path(exists=False))
@click.option("--device-map", "-m", default="auto")
def merge(**kwargs):
    args = Namespace(**kwargs)
    args.device_map = parse_device_map(args.device_map)

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map=args.device_map, torch_dtype=torch.float16)

    model = PeftModel.from_pretrained(model, args.lora_model)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merged_model)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    tokenizer.save_pretrained(args.merged_model)
