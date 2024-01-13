import click

from .commands.eval import _eval
from .commands.expand import _expand
from .commands.finalize import _finalize
from .commands.infer import _infer
from .commands.init import _init
from .commands.launch import _launch
from .commands.merge import _merge
from .commands.quantize import _quantize
from .commands.snapshot import _snapshot
from .commands.train import _train

cli = click.CommandCollection(sources=[
    _eval,
    _expand,
    _finalize,
    _infer,
    _init,
    _launch,
    _merge,
    _quantize,
    _snapshot,
    _train
])

if __name__ == "__main__":
    cli()