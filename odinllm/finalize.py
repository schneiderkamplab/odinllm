import click
import os
import torch

from .shared import load_lora, load_model, load_tokenizer, save_metadata, save_model, save_tokenizer
from .utils import args_config, start, status

@click.group()
def _finalize():
    pass
@_finalize.command()
@click.argument("config", type=click.Path(exists=True), nargs=-1)
@click.argument("training-dir", type=click.Path(exists=True))
@click.option("--checkpoint", default=None, type=int)
@click.option("--base-model", default=None, type=click.Path(exists=True))
@args_config
def finalize(args, config):
    start("Determining checkpoint location")
    if args.checkpoint is None:
        best_link = os.path.join(args.training_dir, "best")
        latest_link = os.path.join(args.training_dir, "latest")
        if os.path.islink(best_link):
            args.checkpoint = int(os.readlink(best_link).split("checkpoint-")[1])
        elif os.path.islink(latest_link):
            args.checkpoint = int(os.readlink(latest_link).split("checkpoint-")[1])
        else:
            candidates = [int(entry.split("checkpoint-")[1]) for entry in os.listdir(args.training_dir) if entry.startswith("checkpoint-")]
            candidates.sort(reverse=True)
            if candidates:
                args.checkpoint = candidates[0]
            else:
                raise ValueError("Could not determine checkpoint automatically")
    checkpoint_dir = os.path.join(args.training_dir, f"checkpoint-{args.checkpoint}")
    status(checkpoint_dir)
    if args.base_model is None:
        model = load_model(
            model_dir=checkpoint_dir,
            config=config,
        )
    else:
        model = load_model(
            model_dir=args.base_model,
            config=config,
        )
        model = load_lora(model, checkpoint_dir)
    save_model(model, args.training_dir)
    tokenizer = load_tokenizer(checkpoint_dir, config)
    save_tokenizer(tokenizer, args.training_dir)
    training_args = torch.load(os.path.join(checkpoint_dir, "training_args.bin"))
    torch.save(training_args, os.path.join(args.training_dir, "training_args.bin"))
    save_metadata(model.metadata, config, args.training_dir)
