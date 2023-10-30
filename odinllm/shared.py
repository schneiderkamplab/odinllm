import datasets
import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from .utils import Concatenator, EXAMPLE_PROMPTS, FEATURES2PROMPT, end, start

def apply_concatenation(dataset, chunk_length):
    start("Concatenating samples")
    dataset = dataset.map(Concatenator(chunk_size=chunk_length), batched=True)
    end()
    return dataset

def apply_templates(dataset, tokenizer):
    start("Creating prompts from data")
    def apply_prompt_template(sample):
        sample_clean = {k: v for k, v in sample.items() if v.strip()}
        features = tuple(sorted(sample_clean.keys()))
        prompt = FEATURES2PROMPT.get(features, None)
        if prompt is None:
            features = tuple(sorted(sample.keys()))
            prompt = FEATURES2PROMPT.get(features, None)
            if prompt is None:
                raise RuntimeError(f"no prompt template for feature combination {features} for sample {sample}")
            sample_clean = sample
        return {
            "text": prompt.format(
                eos_token=tokenizer.eos_token,
                **sample_clean,
            )
        }
    dataset = dataset.map(
        apply_prompt_template,
        remove_columns=list(dataset.features),
    )
    end()
    return dataset

def apply_tokenization(dataset, tokenizer, padding_length=None):
    start("Tokenizing data")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_kwargs = {} if padding_length is None else {
        "padding": "max_length",
        "max_length": padding_length,
        "truncation": True,
    }
    def apply_tokenizer(sample):
        sample = tokenizer(sample["text"], return_tensors="pt", **tokenizer_kwargs)
        sample["labels"] = sample["input_ids"].clone()
        return sample
    dataset = dataset.map(
        apply_tokenizer,
        batched=True,
        batch_size=1,
        remove_columns=list(dataset.features),
    )
    end()
    return dataset

def load_and_quantize(model_dir, bits, group_size, act_order, dataset, tokenizer, device_map):
    start(
        "Loading and quantizing model from", model_dir,
        "to", bits, "bits with group size", group_size,
        f"and {'' if act_order else 'no'} act order using dataset", dataset,
    )
    quantization_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=act_order,
        dataset=dataset,
        tokenizer=tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=quantization_config, device_map=device_map)
    end()
    return model

def load_dataset(dataset):
    start("Loading dataset from", dataset)
    if os.path.isfile(dataset):
        dataset = datasets.load_dataset("json", data_files=dataset, split="train")
    else:
        dataset = datasets.load_dataset(dataset, split="train")
    end()
    return dataset

def load_lora(model, lora_dir):
    start("Loading LoRA adapter from", lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir)
    end()
    return model

def load_model(model_dir, device_map, qualifier, dtype=torch.float16, load_in_4bit=False):
    start("Loading", qualifier, "from", model_dir)
    kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
    } if load_in_4bit else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device_map,
        torch_dtype=dtype,
        **kwargs,
    )
    end()
    return model

def load_tokenizer(model_dir):
    start("Loading tokenizer from", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        padding_side="right",
        truncation_side="right",
    )
    end()
    return tokenizer

def merge_model(model):
    start("Merging LoRA adapter into pretrained model")
    model = model.merge_and_unload()
    end()
    return model

def run_prompt(model, tokenizer, run_prompt=None, max_new_tokens=128):
    results = []
    if run_prompt is not None:
        start("Testing given prompt", run_prompt)
        res = tokenizer.decode(model.generate(**tokenizer(run_prompt, return_tensors="pt").to(model.device),max_new_tokens=max_new_tokens)[0])
        end()
        results.append(res)
    else:
        for key, prompt in EXAMPLE_PROMPTS.items():
            start(f"Testing {key} prompt")
            res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=max_new_tokens)[0])
            end()
            results.append(res)
    return "\n>>>>>>>>> DIVIDER <<<<<<<<<\n".join(results)

def save_lora(model, lora_dir):
    start("Saving LoRA adapter to", lora_dir)
    model.save_pretrained(lora_dir)
    end()

def save_model(model, model_dir, qualifier):
    start("Saving", qualifier, "to", model_dir)
    model.save_pretrained(model_dir)
    end()

def save_tokenizer(tokenizer, model_dir):
    start("Saving tokenizer to", model_dir)
    tokenizer.save_pretrained(model_dir)
    end()
