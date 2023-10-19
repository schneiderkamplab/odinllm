import click
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import logging

from .utils import Namespace, end, parse_device_map, start, status

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

@click.group()
def _infer():
    pass
@_infer.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.option("--lora-adapter", "-l", default=None, help="Optional LoRA adapter to load with PEFT")
@click.option("--eval-prompt", "-p", default="The main difference between a llama and alpaca is ")
@click.option("--device-map", "-m", default="auto")
def infer(**kwargs):
    args = Namespace(**kwargs)
    args.device_map = parse_device_map(args.device_map)
    
    start("Loading tokenizer from", args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    end()

    start("Loading pretrained model from", args.pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map=args.device_map)
    end()

    if args.lora_adapter is not None:
        start("Loading LoRA adapter from", args.lora_adapter)
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        end()

    start("Testing given/default prompt", args.pretrained_model)
    eval_prompt = args.eval_prompt
    res = tokenizer.decode(model.generate(**tokenizer(eval_prompt, return_tensors="pt").to(model.device),max_new_tokens=128)[0])
    end(end='')
    status(res[len(eval_prompt):].split("\n\n### Instruction:\n")[0].strip())

    start("Testing summarization prompt")
    eval_prompt = """
    Summarize this dialog:
    A: Hi Tom, are you busy tomorrow’s afternoon?
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
    A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
    ---
    Summary:
    """
    res = tokenizer.decode(model.generate(**tokenizer(eval_prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
    end(end='')
    status(res[len(eval_prompt):].split("\n\n### Instruction:\n")[0].strip())

    start("Testing instruction prompt")
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize this dialog.\n\n### Input:\n{dialogue}\n\n### Response:\n"
    )
    dialogue = """
    A: Hi Tom, are you busy tomorrow’s afternoon?
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
    A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
    """
    eval_prompt = prompt.format(dialogue=dialogue)
    res = tokenizer.decode(model.generate(**tokenizer(eval_prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0])
    end(end='')
    status(res.split("\n\n### Response:\n")[1])

if __name__ == "__main__":
    infer()