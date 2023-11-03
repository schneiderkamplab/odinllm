import click

from .utils import parse_args

@click.group()
def _train():
    pass
@_train.command()
@parse_args
def train(args):
    raise NotImplementedError()