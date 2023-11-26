import click

from .eval import _eval
from .finalize import _finalize
from .infer import _infer
from .init import _init
from .merge import _merge
from .quantize import _quantize
from .snapshot import _snapshot
from .train import _train

cli = click.CommandCollection(sources=[
    _eval,
    _finalize,
    _infer,
    _init,
    _merge,
    _quantize,
    _snapshot,
    _train
])

if __name__ == "__main__":
    cli()