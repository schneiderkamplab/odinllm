import click

from .infer import _infer
from .quantize import _quantize
from .train import _train

cli = click.CommandCollection(sources=[_infer, _quantize, _train])

if __name__ == "__main__":
    cli()