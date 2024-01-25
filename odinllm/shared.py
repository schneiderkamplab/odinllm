from datasets import concatenate_datasets, load_dataset
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.trainer import ConstantLengthDataset
import yaml

from .utils import chars_token_ratio, end, format_text, metadata_filename, start, status

def load_datasets(tokenizer, config):
    train_datasets = {}
    for dataset in config.datasets["train_datasets"]:
        train_datasets[dataset["name"]] = load_dataset_single(
            tokenizer=tokenizer,
            dataset=dataset,
            num_proc=config.datasets["num_proc"],
            seed=config.datasets["seed"],
        )
    if len(train_datasets) == 0:
        train_datasets = None
    elif len(train_datasets) == 1:
        train_datasets = next(iter(train_datasets.values()))
    else:
        train_datasets = concatenate_datasets(train_datasets.values())
        if config.datasets["shuffle"]:
            train_datasets = train_datasets.shuffle(seed=config.datasets["seed"])
            train_datasets = train_datasets.flatten_indices()
    if train_datasets is not None:
        train_datasets = load_dataset_finalize(
            train_datasets,
            tokenizer=tokenizer,
            dataset=dataset,
            seq_length=config.datasets["seq_length"],
        )
    eval_datasets = {}
    for dataset in config.datasets["eval_datasets"]:
        eval_dataset = load_dataset_single(
            tokenizer=tokenizer,
            dataset=dataset,
            num_proc=config.datasets["num_proc"],
            seed=config.datasets["seed"],
        )
        eval_dataset = load_dataset_finalize(
            eval_dataset,
            tokenizer=tokenizer,
            dataset=dataset,
            seq_length=config.datasets["seq_length"],
        )
        eval_datasets[dataset["name"]] = eval_dataset
    if len(eval_datasets) == 0:
        eval_datasets = None
    elif len(eval_datasets) == 1:
        eval_datasets = next(iter(eval_datasets.values()))
    return train_datasets, eval_datasets

def load_datasets_quantize(tokenizer, config):
    quantize_datasets = {}
    for dataset in config.datasets["quantize_datasets"]:
        quantize_dataset = load_dataset_single(
            tokenizer=tokenizer,
            dataset=dataset,
            num_proc=config.datasets["num_proc"],
            seed=config.datasets["seed"],
        )
        quantize_datasets[dataset["name"]] = quantize_dataset
    quantize_datasets = [row["text"] for quantize_dataset in quantize_datasets.values() for row in quantize_dataset]
    return quantize_datasets

def load_dataset_finalize(ds, tokenizer, dataset, seq_length):
    start("Preprocessing the dataset")
    chars_per_token = chars_token_ratio(ds, tokenizer, prepare_sample_text=format_text, prepare_sample_text_kwargs={"tokenizer": tokenizer})
    status(f"chars/token: {chars_per_token:.2f}", end='')
    ds = ConstantLengthDataset(
        tokenizer,
        ds,
        dataset_text_field="text",
        infinite=dataset["infinite"],
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    end()
    return ds

def load_dataset_single(tokenizer, dataset, num_proc, seed):
    start("Loading dataset from", dataset["name"])
    if os.path.isfile(dataset["name"]):
        ds = load_dataset(
            "json",
            data_files=dataset["name"],
            split=dataset["split"],
            num_proc=num_proc,
        )
    else:
        ds = load_dataset(
            dataset["name"],
            split=dataset["split"],
            num_proc=num_proc,
        )
    end()
    if dataset.get("shuffle", False):
        start("Shuffling dataset")
        ds = ds.shuffle(seed=seed)
        ds = ds.flatten_indices()
        end()
    start("Slicing dataset")
    def fix(index):
        if index >= len(ds):
            index = len(ds)
        while index < 0:
            index += len(ds)
        return index
    from_index = fix(dataset.get("from", 0))
    to_index = fix(dataset.get("to", len(ds)))
    ds = ds.select(range(from_index, to_index))
    end()
    status("Applying templates")
    ds = ds.map(lambda x: {"text": format_text(x, tokenizer=tokenizer)})
    end()
    return ds

def load_lora(model, lora_dir):
    start("Loading LoRA adapter from", lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir)
    assert "metadata" not in model.__dict__
    model.metadata = load_metadata(lora_dir)
    end()
    return model

def load_metadata(dir):
    meta = metadata_filename(dir)
    if os.path.isfile(meta):
        with open(meta, "rt") as f:
            return list(yaml.safe_load_all(f))
    return []

def load_model(model_dir, config):
    start("Loading pretrained model from", model_dir)
    bnb_config = BitsAndBytesConfig(**config.bnb_config) if config.model["load_in_4bit"] or config.model["load_in_8bit"] else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        **config.model,
    )
    model.config.use_cache = False
    status(model.device)
    assert "metadata" not in model.__dict__
    model.metadata = load_metadata(model_dir)
    end()
    return model

def load_tokenizer(model_dir, config):
    start("Loading tokenizer from", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        **config.tokenizer,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.model_max_length > 1_000_000:
        tokenizer.model_max_length = 2048
    end()
    return tokenizer

def save_metadata(metadata, config, model_dir):
    start("Saving metadata to", model_dir)
    metadata.append(config._config)
    with open(metadata_filename(model_dir), "wt") as f:
        yaml.safe_dump_all(metadata, f)
    status(f"#entries: {len(metadata)}", end='')
    end()

def save_model(model, model_dir):
    start("Saving model to", model_dir)
    model.save_pretrained(model_dir)
    end()

def save_tokenizer(tokenizer, model_dir):
    start("Saving tokenizer to", model_dir)
    tokenizer.save_pretrained(model_dir)
    end()
