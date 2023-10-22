import click

from .shared import load_and_quantize, load_tokenizer, save_model, save_tokenizer
from .utils import parse_args

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