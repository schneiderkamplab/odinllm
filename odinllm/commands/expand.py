import click
import copy
import re
import torch

from ..shared import load_model, load_tokenizer, save_metadata, save_model, save_tokenizer
from ..utils import args_config, end, start, status

@click.group()
def _expand():
    pass
@_expand.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("expanded-model", type=click.Path(exists=False))
@click.option("--every", "-e", default=None, type=int)
@click.option("--layers", "-t", default=None, type=str)
@click.option("--zero", "-t", default=None, type=str, multiple=True)
@args_config
def expand(args, config):
    return __expand(args, config)

def __expand(args, config):
    model = load_model(
        args.pretrained_model,
        config=config,
    )
    layers = model.get_submodule(args.layers)
    start(f"Expanding model from {len(layers)} layers")
    inserts = reversed(range(args.every-1, len(layers), args.every))
    for i in inserts:
        layers.insert(i, copy.deepcopy(layers[i]))
        model.config.num_hidden_layers += 1
        for zero in args.zero:
            try:
                module = layers[i].get_submodule(zero)
                module.weight.data = torch.zeros_like(module.weight.data)
            except AttributeError as e:
                status(f"WARNING could not find {args.layers}.{i}.{zero}")
    status(f"#layers: {len(layers)}", end='')
    end()
    save_model(model, args.expanded_model)
    tokenizer = load_tokenizer(args.pretrained_model, config)
    save_tokenizer(tokenizer, args.expanded_model)
    save_metadata(model.metadata, config, args.expanded_model)
    
