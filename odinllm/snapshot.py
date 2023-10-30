import click
from huggingface_hub import  hf_hub_download, snapshot_download
from os import getcwd
from shutil import copy, copytree

from .utils import end, parse_args, start

@click.group()
def _snapshot():
    pass
@_snapshot.command()
@click.argument("repo-id", type=str)
@click.option("--file-name", default=[], type=str, multiple=True)
@parse_args
def snapshot(args):
    if not args.file_name:
        start("Snapshotting repository", args.repo_id)
        cache_dirname = snapshot_download(repo_id=args.repo_id)
        end()
        start("Copying repository from", cache_dirname, "to", args.repo_id.split("/")[-1])
        copytree(cache_dirname, args.repo_id.split("/")[-1])
        end()
    else:
        for file_name in args.file_name:
            start("Downloading", file_name, "from repository", args.repo_id)
            cache_file_name = hf_hub_download(repo_id=args.repo_id, filename=file_name)
            end()
            start("Copying", cache_file_name, "to", getcwd())        
            copy(cache_file_name, getcwd())
            end()
