from argparse import Namespace
from itertools import chain
import logging
import time

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args(func):

    def parse(**kwargs):
        args = Namespace(**kwargs)
        if "device_map" in kwargs:
            args.device_map = parse_device_map(args.device_map)
        if "bits" in kwargs:
            args.bits = int(args.bits)
        if "group_size" in kwargs:
            args.group_size = int(args.group_size)
        return func(args)

    parse.__name__ = func.__name__
    return parse

def parse_device_map(device_map):
    if device_map.isnumeric():
        return int(device_map)
    if device_map.strip().startswith("{"):
        return eval(device_map)
    return device_map

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
    ("text"): "{text}",
}

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

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result

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