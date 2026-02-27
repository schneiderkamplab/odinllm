import click
from datasets import load_dataset
import os

from ..shared import save_metadata, save_model, save_tokenizer
from ..utils import args_config, end, start, status, trainable_parameters

def init_model(config_class, model_class, torch_dtype, config):
    start("Creating untrained model")
    model_config = config_class(**config.model_config)
    model = model_class(model_config)
    model = model.to(torch_dtype)
    _, all_params = trainable_parameters(model)
    status(f"#params: {all_params}", end='')
    assert "metadata" not in model.__dict__
    model.metadata = []
    end()
    return model

def init_tokenizer(tokenizer_class, tokenizer_template, vocab_size, min_frequency, corpora):
    start("Training tokenizer")
    old_tokenizer = tokenizer_class.from_pretrained(tokenizer_template)
    def iterator():
        for dataset_name in corpora:
            status(dataset_name)
            if os.path.isfile(dataset_name):
                dataset = load_dataset("json", data_files=dataset_name, streaming=True)
            else:
                dataset = load_dataset(dataset_name, "da", streaming=True)
            for ds in dataset.values():
                column = ds.select_columns("text")
                for item in column:
                    item = item["text"]
                    if isinstance(item, str):
                        yield item
    tokenizer = old_tokenizer.train_new_from_iterator(
        iterator(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    end()
    return tokenizer

@click.group()
def _init():
    pass
@_init.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("untrained-model", type=str)
@click.option("--corpora", default=None, type=click.Path(exists=False), multiple=True)
@args_config
def init(args, config):
    return __init(args, config)

def __init(args, config):
    model = init_model(
        config_class=args.config_class,
        model_class=args.model_class,
        torch_dtype=args.torch_dtype,
        config=config,
    )
    save_model(model, args.untrained_model)
    if args.corpora:
        tokenizer = init_tokenizer(
            tokenizer_class=args.tokenizer_class,
            tokenizer_template=args.tokenizer_template,
            vocab_size=model.config.vocab_size,
            min_frequency=args.tokenizer_min_frequency,
            corpora=args.corpora,
        )
        save_tokenizer(tokenizer, args.untrained_model)
    save_metadata(model.metadata, config, args.untrained_model)
