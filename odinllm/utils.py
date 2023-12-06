from accelerate import Accelerator
from argparse import Namespace
import logging
import os
import sys
import time
import torch
from tqdm import tqdm

# logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

# arguments
def update_key(kwargs, key, val):
    kwargs["_"+key] = kwargs[key]
    kwargs[key] = val

def parse_args(func):
    def parse(**kwargs):
        for key, val in list(kwargs.items()):
            if key == "device_map":
                update_key(kwargs, key, parse_device_map(val))
            elif key == "target_modules":
                update_key(kwargs, key, eval(val))
            elif key.endswith("_class"):
                import transformers
                update_key(kwargs, key, eval(f"transformers.{val}"))
            elif key.endswith("_dtype"):
                import torch
                update_key(kwargs, key, eval(f"torch.{val}"))
            elif isinstance(val, str) and val[0] in ("[", "{"):
                update_key(kwargs, key, eval(val))
        kwargs["command"] = func.__name__
        kwargs["argv"] = sys.argv
        args = Namespace(**kwargs)
        return func(args)
    parse.__name__ = func.__name__
    return parse

def parse_device_map(device_map):
    if device_map.isnumeric():
        return int(device_map)
    if device_map.strip().startswith("{"):
        return eval(device_map)
    return device_map

def unparse_args(original_args):
    args = dict(vars(original_args))
    for key in [key for key in args if key.startswith("_")]:
        args[key[1:]] = args[key]
        del args[key]
    return args

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

def format_text(sample, tokenizer=None):
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
        eos_token=tokenizer.eos_token,
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
METADATA_FILENAME="metadata.json"

def metadata_filename(dir):
    return os.path.join(dir, METADATA_FILENAME)

# distributed training
def get_current_device():
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def get_device_map():
    return {"": get_current_device()} if torch.cuda.is_available() else None
