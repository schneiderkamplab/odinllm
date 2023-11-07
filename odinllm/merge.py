import click
from .shared import load_lora, load_model, load_tokenizer, save_metadata, save_model, save_tokenizer
from .utils import end, parse_args, start

def merge_model(model):
    start("Merging LoRA adapter into pretrained model")
    merged_model = model.merge_and_unload()
    merged_model.metadata = model.metadata
    end()
    return merged_model

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
    base_model = load_model(args.pretrained_model, device_map=args.device_map, qualifier="pretrained model", load_in_4bit=False)
    lora_model = load_lora(base_model, args.lora_model)
    merged_model = merge_model(lora_model)
    save_model(merged_model, args.merged_model, qualifier="merged model")
    tokenizer = load_tokenizer(args.pretrained_model)
    save_tokenizer(tokenizer, args.merged_model)
    save_metadata(merged_model.metadata, args, args.merged_model)
