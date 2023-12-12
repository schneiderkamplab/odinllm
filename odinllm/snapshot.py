import click
from huggingface_hub import  hf_hub_download, snapshot_download
from os import getcwd
from shutil import copy, copytree

from .shared import save_metadata
from .utils import args_config, end, start

@click.group()
def _snapshot():
    pass
@_snapshot.command()
@click.argument("config", type=click.Path(exists=True), nargs=-1)
@click.argument("repo-id", type=str)
@click.option("--file-name", default=[], type=str, multiple=True)
@click.option("--target-dir", default=None, type=str)
@args_config
def snapshot(args, config):
    if not args.file_name:
        start("Snapshotting repository", args.repo_id)
        cache_dirname = snapshot_download(repo_id=args.repo_id)
        end()
        repo_dir = args.repo_id.split("/")[-1] if args.target_dir is None else args.target_dir
        start("Copying repository from", cache_dirname, "to", repo_dir)
        copytree(cache_dirname, repo_dir)
        save_metadata([], config, repo_dir)
        end()
    else:
        for file_name in args.file_name:
            start("Downloading", file_name, "from repository", args.repo_id)
            cache_file_name = hf_hub_download(repo_id=args.repo_id, filename=file_name)
            end()
            start("Copying", cache_file_name, "to", getcwd())        
            copy(cache_file_name, getcwd())
            end()
