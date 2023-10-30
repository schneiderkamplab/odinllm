import click

from .shared import load_tokenizer, load_model, load_lora, run_prompt
from .utils import parse_args

@click.group()
def _infer():
    pass
@_infer.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.option("--lora-adapter", "-l", default=None, help="Optional LoRA adapter to load with PEFT")
@click.option("--run-prompt", "-p", default=None, help="Prompt to run instead of example prompts")
@click.option("--load-in-4bit/--no-load-in-4bit", default=True)
@click.option("--device-map", "-m", default="auto")
@parse_args
def infer(args):
    tokenizer = load_tokenizer(args.pretrained_model)
    model = load_model(
        args.pretrained_model,
        device_map=args.device_map,
        qualifier="pretrained model",
        load_in_4bit=args.load_in_4bit,
    )
    if args.lora_adapter is not None:
        model = load_lora(model, args.lora_adapter)
    print(run_prompt(model, tokenizer, args.run_prompt))

if __name__ == "__main__":
    infer()