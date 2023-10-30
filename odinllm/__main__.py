import click

from .infer import _infer
from .merge import _merge
from .quantize import _quantize
from .snapshot import _snapshot
from .train import _train

cli = click.CommandCollection(sources=[_infer, _merge, _quantize, _snapshot, _train])

if __name__ == "__main__":
    cli()