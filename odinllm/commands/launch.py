from argparse import Namespace
import click
import yaml

from ..utils import args_config, eval_config, merge_config, start, status

@click.group()
def _launch():
    pass
@_launch.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("command-file", type=click.Path(exists=True))
@args_config
def launch(args, config):
    return __launch(args, config)

def __launch(args, config):
    _config = config._config
    del _config["command"]
    del _config["launch"]
    configs = yaml.safe_load_all(open(args.command_file))
    for proto_config in configs:
        start("Identifying command")
        cmd = proto_config["command"]
        status(cmd)
        merge_config(proto_config, _config, ignore_none=False)
        kwconfig = eval_config(proto_config)
        kwconfig["_config"] = proto_config
        new_args = Namespace(**kwconfig[cmd])
        new_config = Namespace(**kwconfig)
        cmd_dict = {}
        exec(f"from .{cmd} import __{cmd} as {cmd}", globals(), cmd_dict)
        cmd_dict[cmd](new_args, new_config)
