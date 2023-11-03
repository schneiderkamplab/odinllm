import click

from .shared import load_tokenizer, load_model, load_lora
from .utils import EXAMPLE_PROMPTS, end, parse_args, start

def run_prompt(model, tokenizer, run_prompt=None, max_new_tokens=128):
    results = []
    if run_prompt is not None:
        start("Testing given prompt", run_prompt)
        res = tokenizer.decode(model.generate(**tokenizer(run_prompt, return_tensors="pt").to(model.device),max_new_tokens=max_new_tokens)[0])
        end()
        results.append(res)
    else:
        for key, prompt in EXAMPLE_PROMPTS.items():
            start(f"Testing {key} prompt")
            res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=max_new_tokens)[0])
            end()
            results.append(res)
    return "\n>>>>>>>>> DIVIDER <<<<<<<<<\n".join(results)

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