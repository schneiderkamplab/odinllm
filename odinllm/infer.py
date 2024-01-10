import click

from .shared import load_tokenizer, load_model, load_lora
from .utils import EXAMPLE_PROMPTS, args_config, end, start

def run_prompt(model, tokenizer, run_prompts=[], max_new_tokens=128):
    results = []
    if run_prompts:
        for run_prompt in run_prompts:
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
@click.argument("config", type=click.Path(exists=True), nargs=-1)
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.option("--lora-adapter", "-l", default=None, help="Optional LoRA adapter to load with PEFT")
@click.option("--run-prompt", "-p", default=[], help="Prompt to run instead of example prompts", multiple=True)
@click.option("--max-new-tokens", "-t", default=128, help="Maximum number of new tokens to generate")
@args_config
def infer(args, config):
    tokenizer = load_tokenizer(args.pretrained_model, config)
    model = load_model(
        args.pretrained_model,
        config=config,
    )
    if args.lora_adapter is not None:
        model = load_lora(model, args.lora_adapter)
    print(run_prompt(model, tokenizer, args.run_prompt, args.max_new_tokens))

if __name__ == "__main__":
    infer()