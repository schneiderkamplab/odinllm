from argparse import Namespace
import click
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from .utils import parse_device_map

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
    args = Namespace(**kwargs)
    args.device_map = parse_device_map(args.device_map)
    args.bits = int(args.bits)
    args.group_size = int(args.group_size)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)

    quantization_config = GPTQConfig(
        bits=args.bits,  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=args.act_order,  # set to False can significantly speed up inference but the perplexity may slightly bad
        dataset = args.dataset,
        tokenizer = tokenizer,
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, quantization_config=quantization_config, device_map=args.device_map)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    #model.quantize(examples)

    # save quantized model
    model.save_pretrained(args.quantized_model)
    tokenizer.save_pretrained(args.quantized_model)

if __name__ == "__main__":
    quantize()