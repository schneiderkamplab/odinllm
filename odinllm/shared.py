from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from .utils import EXAMPLE_PROMPTS, end, start

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

def load_lora(model, lora_dir):
    start("Loading LoRA adapter from", lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir)
    end()
    return model

def load_model(model_dir, device_map):
    start("Loading pretrained model from", model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
    end()
    return model

def load_tokenizer(model_dir):
    start("Loading tokenizer from", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    end()
    return tokenizer

def run_prompt(model, tokenizer, run_prompt):
    results = []
    if run_prompt is not None:
        start("Testing given prompt", run_prompt)
        res = tokenizer.decode(model.generate(**tokenizer(run_prompt, return_tensors="pt").to(model.device),max_new_tokens=128)[0])
        end()
        results.append(res)
    else:
        for key, prompt in EXAMPLE_PROMPTS.items():
            start(f"Testing {key} prompt")
            res = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
            end()
            results.append(res)
    return "\n>>>>>>>>> DIVIDER <<<<<<<<<\n".join(results)

def save_model(model, model_dir, qualifier=None):
    start("Saving","" if qualifier is None else qualifier, "to", model_dir)
    model.save_pretrained(model_dir)
    end()

def save_tokenizer(tokenizer, model_dir):
    start("Saving tokenizer to", model_dir)
    tokenizer.save_pretrained(model_dir)
    end()
