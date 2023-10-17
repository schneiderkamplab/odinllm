import click
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import logging

from .utils import Namespace, parse_device_map

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

@click.group()
def _infer():
    pass
@_infer.command()
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.option("--eval-prompt", "-p", default="The main difference between a llama and alpaca is ")
@click.option("--device-map", "-m", default="auto")
def infer(**kwargs):
    args = Namespace(**kwargs)
    args.device_map = parse_device_map(args.device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map=args.device_map)
    print(tokenizer.decode(model.generate(**tokenizer(args.eval_prompt, return_tensors="pt").to(model.device),max_new_tokens=128)[0]))
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
    print(tokenizer.decode(model.generate(**tokenizer(eval_prompt, return_tensors="pt").to(model.device),max_new_tokens=256)[0]))

if __name__ == "__main__":
    infer()