import click
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from .utils import end, parse_args, start

@click.group()
def _quantize():
    pass
@_quantize.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("quantized-model", type=click.Path(exists=False))
@click.option("--bits", "-b", default="4", type=click.Choice(["2","4","8"]))
@click.option("--group-size", "-g", default="32", type=click.Choice([str(2**i) for i in range(11)]))
@click.option("--act-order/--no-act-order", default=False)
@click.option("--dataset", "-d", default="c4")
@click.option("--device-map", "-m", default="auto")
def quantize(**kwargs):
    args = parse_args(kwargs)

    start("Loading tokenizer from", args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    end()

    start("Loading and quantizing model from", args.pretrained_model, "to", args.bits, "bits with group size", args.group_size, f"and {'' if args.act_order else 'no'} act order using dataset", args.dataset)
    quantization_config = GPTQConfig(
        bits=args.bits,  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=args.act_order,  # set to False can significantly speed up inference but the perplexity may slightly bad
        dataset = args.dataset,
        tokenizer = tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, quantization_config=quantization_config, device_map=args.device_map)
    end()

    start("Saving quantized model to", args.quantized_model)
    model.save_pretrained(args.quantized_model)
    end()

    start("Saving tokenizer to", args.quantized_model)
    tokenizer.save_pretrained(args.quantized_model)
    end()

if __name__ == "__main__":
    quantize()