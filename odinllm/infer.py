import click
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import logging

from .utils import EXAMPLE_PROMPTS, end, parse_args, start, status

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

@click.group()
def _infer():
    pass
@_infer.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.option("--lora-adapter", "-l", default=None, help="Optional LoRA adapter to load with PEFT")
@click.option("--eval-prompt", "-p", default=None, help="Prompt to run instead of example prompts")
@click.option("--device-map", "-m", default="auto")
def infer(**kwargs):
    args = parse_args(kwargs)
    
    start("Loading tokenizer from", args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    end()

    start("Loading pretrained model from", args.pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map=args.device_map)
    end()

    if args.lora_adapter is not None:
        start("Loading LoRA adapter from", args.lora_adapter)
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        end()

    if args.eval_prompt is not None:
        start("Testing given prompt", args.eval_prompt)
        res = tokenizer.decode(model.generate(**tokenizer(args.eval_prompt, return_tensors="pt").to(model.device),max_new_tokens=128)[0])
        end(end='')
        status(res)
    else:
        for key, prompt in EXAMPLE_PROMPTS.items():
            start(f"Testing {key} prompt")
            res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
            end(end='')
            status(res)

if __name__ == "__main__":
    infer()