import json
import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import end, metadata_filename, start, status, unparse_args

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
            return json.load(f)
    return []

def load_model(model_dir, device_map, qualifier, load_in_4bit):
    start("Loading", qualifier, "from", model_dir, "with device map", device_map)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    status(model.device)
    assert "metadata" not in model.__dict__
    model.metadata = load_metadata(model_dir)
    end()
    return model

def load_tokenizer(model_dir):
    start("Loading tokenizer from", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    end()
    return tokenizer

def save_metadata(metadata, args, model_dir):
    metadata.append(unparse_args(args))
    with open(metadata_filename(model_dir), "wt") as f:
        json.dump(metadata, f, indent=2)

def save_model(model, model_dir, qualifier):
    start("Saving", qualifier, "to", model_dir)
    model.save_pretrained(model_dir)
    end()

def save_tokenizer(tokenizer, model_dir):
    start("Saving tokenizer to", model_dir)
    tokenizer.save_pretrained(model_dir)
    end()
