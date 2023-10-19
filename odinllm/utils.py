from argparse import Namespace
from itertools import chain
import logging
import time

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

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