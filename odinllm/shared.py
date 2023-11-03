from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import chars_token_ratio, end, get_prepare_sample_text, start, status, trainable_parameters

def load_lora(model, lora_dir):
    start("Loading LoRA adapter from", lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir)
    end()
    return model

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

def save_model(model, model_dir, qualifier):
    start("Saving", qualifier, "to", model_dir)
    model.save_pretrained(model_dir)
    end()

def save_tokenizer(tokenizer, model_dir):
    start("Saving tokenizer to", model_dir)
    tokenizer.save_pretrained(model_dir)
    end()
