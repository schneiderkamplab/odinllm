import click
from datasets import Dataset
from transformers import TrainingArguments

from ..shared import load_datasets, load_model, load_tokenizer
from ..trainer import OdinTrainer
from ..utils import args_config, end, start

def do_eval(trainer):
    start("Evaluating")
    print(trainer.evaluate())
    end()

@click.group()
def _eval():
    pass
@_eval.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("pretrained-model", type=click.Path(exists=True))
@args_config
def eval(args, config):
    return __eval(args, config)

def __eval(args, config):
    if config.training_args["run_name"] is None:
        config.training_args["run_name"] = f"{config.command}_{args.pretrained_model.split('/')[-1]}"
    if config.training_args["output_dir"] is None:
        config.training_args["output_dir"] = "output"
    model = load_model(
        args.pretrained_model,
        config,
    )
    tokenizer = load_tokenizer(args.pretrained_model, config)
    training_args = TrainingArguments(
        **config.training_args,
    )
    train_dataset, eval_dataset = load_datasets(tokenizer=tokenizer, config=config)
    trainer = OdinTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        **config.trainer_args,
    )
    do_eval(trainer)
