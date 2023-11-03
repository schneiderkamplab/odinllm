import click
from transformers import AutoModelForCausalLM, GPTQConfig

from .shared import load_tokenizer, save_model, save_tokenizer
from .utils import end, parse_args, start

def load_and_quantize(model_dir, bits, group_size, act_order, dataset, tokenizer, device_map):
    start(
        "Loading and quantizing model from", model_dir,
        "to", bits, "bits with group size", group_size,
        f"and {'' if act_order else 'no'} act order using dataset", dataset,
    )
    quantization_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=act_order,
        dataset=dataset,
        tokenizer=tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=quantization_config, device_map=device_map)
    end()
    return model

@click.group()
def _quantize():
    pass
@_quantize.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("quantized-model", type=click.Path(exists=False))
@click.option("--bits", "-b", default=4, type=int)
@click.option("--group-size", "-g", default=32, type=int)
@click.option("--act-order/--no-act-order", default=False)
@click.option("--dataset", "-d", default="c4")
@click.option("--device-map", "-m", default="auto")
@parse_args
def quantize(args):
    tokenizer = load_tokenizer(args.pretrained_model)
    model = load_and_quantize(
        args.pretrained_model,
        bits=args.bits,
        group_size=args.group_size,
        act_order=args.act_order,
        dataset=args.dataset,
        tokenizer=tokenizer,
        device_map=args.device_map,
    )
    save_tokenizer(tokenizer, args.quantized_model)
    save_model(model, args.quantized_model, qualifier="quantized model")

if __name__ == "__main__":
    quantize()