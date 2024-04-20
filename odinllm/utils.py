from accelerate import Accelerator
from argparse import Namespace
from datetime import datetime
import gc
import logging
import os
import time
import torch
from tqdm import tqdm
import transformers
from typing import Any
import yaml

# logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

# arguments
def merge_config(base, update, ignore_none):
    if isinstance(base, dict):
        assert(isinstance(update, dict))
        for key, val in update.items():
            if isinstance(val, str) and val == "::":
                del base[key]
            else:
                base[key] = merge_config(base.get(key, None), val, ignore_none=ignore_none)
        return base
    if isinstance(base, list):
        assert(isinstance(update, list))
        base.extend(update)
        return base
    if ignore_none and update is None:
        return base
    return update

def eval_config(config):
    if isinstance(config, dict):
        return {key: eval_config(val) for key, val in config.items()}
    elif isinstance(config, list):
        return [eval_config(item) for item in config]
    return eval(config[2:]) if isinstance(config, str) and config.startswith("::") else config

def arg_to_yaml(parts, value):
    return ':\n'.join(f"{'  '*i}{key}" for i, key in enumerate(parts)) + f": {value}"

def args_config(func):
    def parse(**kwargs):
        proto_config = dict()
        for config_file in kwargs["config"]:
            if os.path.exists(config_file):
                with open(config_file, "rt") as f:
                    config = yaml.safe_load(f)
            else:
                parts, value = config_file.split("=", 1)
                config = yaml.safe_load(arg_to_yaml(parts.split("."), value))
            merge_config(proto_config, config, ignore_none=False)
        del kwargs["config"]
        kwargs = {key: (list(val) if isinstance(val, tuple) else val) for key, val in kwargs.items()}
        merge_config(proto_config, {func.__name__: kwargs}, ignore_none=True)
        merge_config(proto_config, {"command": func.__name__, "timestamp": datetime.now().isoformat()}, ignore_none=False)
        kwconfig = eval_config(proto_config)
        kwconfig["_config"] = proto_config
        args = Namespace(**kwconfig[func.__name__])
        config = Namespace(**kwconfig)
        return func(args, config)
    parse.__name__ = func.__name__
    return parse

# prompt templates
FEATURES2PROMPT = {
    ("dialogue", "id", "summary"): (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize this dialog.\n\n### Input:\n{dialogue}\n\n### Response:\n{summary}{eos_token}"
    ),
    ("input", "instruction", "output"): (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}{eos_token}"
    ),
    ("instruction", "output"): (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{output}{eos_token}"
    ),
    ("correct", "incorrect"): (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nPlease correct the following text.\n\n###Input:\n{incorrect}\n\n### Response:\n{correct}{eos_token}"
    ),
    ("text",): "{text}",
}

def format_text(sample, tokenizer=None, append_eos=True):
    if "messages" in sample:
        messages = sample["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        sample["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    if "text" in sample:
        sample = {"text": sample["text"]}
    sample_clean = {k: v for k, v in sample.items() if v.strip()}
    features = tuple(sorted(sample_clean.keys()))
    prompt = FEATURES2PROMPT.get(features, None)
    if prompt is None:
        features = tuple(sorted(sample.keys()))
        prompt = FEATURES2PROMPT.get(features, None)
        if prompt is None:
            raise RuntimeError(f"no prompt template for feature combination {features} for sample {sample}")
        sample_clean = sample
    return prompt.format(
        eos_token=tokenizer.eos_token if append_eos else "",
        **sample_clean,
    )

# example prompts
EXAMPLES = {
    "dia": """A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))""",
}
EXAMPLE_PROMPTS = {
    "raw": "The main difference between a llama and alpaca is ",
    "sum": f"Summarize this dialog:\n{EXAMPLES['dia']}\n---\nSummary:\n",
    "ins": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nSummarize this dialog.\n\n### Input:\n{EXAMPLES['dia']}\n\n### Response:\n",
}

# stats
def chars_token_ratio(dataset, tokenizer, prepare_sample_text, prepare_sample_text_kwargs, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, **prepare_sample_text_kwargs)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_params

# progress
started = 0
def start(*msg):
    global started
    global counter
    counter = 0
    if msg:
        print(" ".join(map(str,msg)).ljust(60),"... ",end='',flush=True)
    started = time.time()
def end(end='\n'):
    global started
    print(" %.3f seconds   " % (time.time()-started),end=end,flush=True)
    started = time.time()
def status(msg,end='\n'):
    print("%s   " % msg,end=end,flush=True)
def file_size(file_name,end='\n'):
    from os import stat
    print("%.0fK" % (stat(file_name).st_size/1024),end=end,flush=True)

# metadata
METADATA_FILENAME="metadata.yaml"

def metadata_filename(dir):
    return os.path.join(dir, METADATA_FILENAME)

# distributed training
def get_current_device():
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def get_device_map():
    return {"": get_current_device()} if torch.cuda.is_available() else None

# moving tensors around
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    for param_group in optim.param_groups:
        for key, val in param_group.items():
            if isinstance(val, torch.Tensor):
                param_group[key] = val.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

def find_all(type_to_find: type = None):
    def _find_all(slist: list, olist: list, seen: dict[Any, None], type_to_find: type):
        for e in slist:
            if id(e) in seen:
                continue
            seen[id(e)] = None
            if type_to_find is None or isinstance(e, type_to_find):
                olist.append(e)
            tl = gc.get_referents(e)
            if tl:
                _find_all(tl, olist, seen, type_to_find=type_to_find)
    gcl = gc.get_objects()
    olist = []
    seen = {}
    seen[id(_find_all)] = None
    seen[id(gcl)] = None
    seen[id(olist)] = None
    seen[id(seen)] = None
    _find_all(gcl, olist, seen, type_to_find=type_to_find)
    return olist