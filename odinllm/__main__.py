import click

from .infer import _infer
from .quantize import _quantize

cli = click.CommandCollection(sources=[_infer, _quantize])

if __name__ == "__main__":
    cli()