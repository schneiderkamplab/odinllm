import click
from transformers import AutoModelForCausalLM, GPTQConfig

from ..shared import load_datasets_quantize, load_metadata, load_model, load_tokenizer, save_metadata, save_model, save_tokenizer
from ..utils import args_config, end, start

def load_and_quantize(model_dir, tokenizer, dataset, config):
    start("Loading and quantizing model from", model_dir)
    quantization_config = GPTQConfig(
        tokenizer=tokenizer,
        dataset=dataset,
        **config.gptq_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        **config.model,
    )
    assert "metadata" not in model.__dict__
    model.metadata = load_metadata(model_dir)
    end()
    return model

@click.group()
def _quantize():
    pass
@_quantize.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("quantized-model", type=click.Path(exists=False))
@args_config
def quantize(args, config):
    return __quantize(args, config)

def __quantize(args, config):
    tokenizer = load_tokenizer(args.pretrained_model, config)
    dataset = load_datasets_quantize(tokenizer, config)
    if config.model["load_in_4bit"] or config.model["load_in_8bit"]:
        model = load_model(args.pretrained_model, config=config)
    else:
        model = load_and_quantize(args.pretrained_model, tokenizer=tokenizer, dataset=dataset, config=config)
    save_tokenizer(tokenizer, args.quantized_model)
    save_model(model, args.quantized_model)
    save_metadata(model.metadata, config, args.quantized_model)

if __name__ == "__main__":
    quantize()