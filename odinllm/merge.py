import click
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import end, parse_args, start

@click.group()
def _merge():
    pass
@_merge.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("lora-model", type=click.Path(exists=True))
@click.argument("merged-model", type=click.Path(exists=False))
@click.option("--device-map", "-m", default="auto")
def merge(**kwargs):
    args = parse_args(kwargs)

    start("Loading pretrained model from", args.pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map=args.device_map, torch_dtype=torch.float16)
    end()

    start("Loading LoRA adapter from", args.lora_model)
    model = PeftModel.from_pretrained(model, args.lora_model)
    end()

    start("Merging LoRA adapter into pretrained model")
    merged_model = model.merge_and_unload()
    end()

    start("Saving merged model to", args.merged_model)
    merged_model.save_pretrained(args.merged_model)
    end()

    start("Loading tokenizer from", args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    end()

    start("Saving tokenizer to", args.merged_model)
    tokenizer.save_pretrained(args.merged_model)
    end()
