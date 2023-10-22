from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import EXAMPLE_PROMPTS, end, start, status

def load_tokenizer(model_dir):
    start("Loading tokenizer from", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    end()
    return tokenizer

def load_model(model_dir, device_map):
    start("Loading pretrained model from", model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
    end()
    return model

def load_lora(model, lora_dir):
    start("Loading LoRA adapter from", lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir)
    end()
    return model

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
