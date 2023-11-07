import click
from datasets import load_dataset
import os
import torch

from .shared import save_metadata, save_model, save_tokenizer
from .utils import end, parse_args, start, status, trainable_parameters

def get_model_config(args):
    start("Building model config")
    config_kwargs = {
        "rms_norm_eps": args.model_rms_norm_eps,
        "torch_dtype": args.model_torch_dtype,
    }
    for key, val in vars(args).items():
        if key.startswith("model_") and isinstance(val, int):
            config_kwargs[key[len("model_"):]] = val
    config = args.config_class(**config_kwargs)
    status(config)
    return config

def init_model(model_class, model_config):
    start("Creating untrained model")
    model = model_class(model_config)
    model = model.to(torch.bfloat16)
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
                dataset = load_dataset("json", data_files=dataset_name)
            else:
                dataset = load_dataset(dataset_name)
            for split, column_names in dataset.column_names.items():
                ds = dataset[split]
                for column_name in column_names:
                    column = ds[column_name]
                    for item in column:
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
@click.argument("untrained-model", type=str)
@click.argument("corpora", type=click.Path(exists=True), nargs=-1)
@click.option("--tokenizer-class", default="LlamaTokenizerFast", type=str)
@click.option("--tokenizer-template", default="meta-llama/Llama-2-7b-hf", type=str)
@click.option("--config-class", default="LlamaConfig", type=str)
@click.option("--model-class", default="LlamaForCausalLM", type=str)
@click.option("--model-rms-norm-eps", default=1e-05, type=float)
@click.option("--model-torch-dtype", default="bfloat16", type=str)
@click.option("--model-hidden-size", default=None, type=int)
@click.option("--model-max-position-embeddings", default=None, type=int)
@click.option("--model-num-attention-heads", default=None, type=int)
@click.option("--model-num-hidden-layers", default=None, type=int)
@click.option("--model-num-key-value-heads", default=None, type=int)
@click.option("--model-vocab-size", default=None, type=int)
@click.option("--tokenizer-min-frequency", default=0, type=int)
@parse_args
def init(args):
    model_config = get_model_config(args)
    model = init_model(
        model_class=args.model_class,
        model_config=model_config,
    )
    save_model(model, args.untrained_model, qualifier="untrained model")
    tokenizer = init_tokenizer(
        tokenizer_class=args.tokenizer_class,
        tokenizer_template=args.tokenizer_template,
        vocab_size=model.config.vocab_size,
        min_frequency=args.tokenizer_min_frequency,
        corpora=args.corpora,
    )
    save_tokenizer(tokenizer, args.untrained_model)
    save_metadata(model.metadata, args, args.untrained_model)
