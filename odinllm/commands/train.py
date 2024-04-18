from accelerate import Accelerator
from bitlinear import replace_modules
import click
import json
import os
from peft import LoraConfig
import re
import time
import torch
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from trl.trainer import DataCollatorForCompletionOnlyLM

from ..shared import load_datasets, load_model, load_tokenizer, save_metadata
from ..trainer import OdinTrainer
from ..utils import (
    args_config,
    end,
    get_device_map,
    start,
    status,
    trainable_parameters,
)

class BitLinearCallback(TrainerCallback):
    def __init__(self, file_name):
        self.file_name = file_name
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        with torch.no_grad():
            logs = dict(logs)
            logs["step"] = state.global_step
            for name, module in model.named_modules():
                #print(name)
                if type(module).__name__ == "BitLinear":
                    scale = 1 / module.measure(module.weight.abs()).clamp_(min=module.eps)
                    abs_max = module.weight.abs().max().item()
                    abs_mean = module.weight.abs().mean().item()
                    abs_median = module.weight.abs().median().item()
                    weights = (module.weight * scale).round().clamp(-1, 1).to(torch.int8).flatten().tolist()
                    zeroes = sum([1 for weight in weights if weight == 0])
                    ones = sum([1 for weight in weights if weight == 1])
                    minus_ones = sum([1 for weight in weights if weight == -1])
                    total = len(weights)
                    logs[f"{name}/abs_max"] = abs_max
                    logs[f"{name}/abs_mean"] = abs_mean
                    logs[f"{name}/abs_median"] = abs_median
                    logs[f"{name}/zeroes"] = zeroes
                    logs[f"{name}/ones"] = ones
                    logs[f"{name}/minus_ones"] = minus_ones
                    logs[f"{name}/total"] = total
            open(self.file_name, "at").write(json.dumps(logs)+"\n")

class MetadataSavingCallback(TrainerCallback):
    def __init__(self, config):
        self.config = config
    def on_save(self, args, state, _, **kwargs):
        if args.should_save:
            checkpoint_name = f"checkpoint-{state.global_step}"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            save_metadata(kwargs["model"].metadata, self.config, checkpoint_path)
            latest_link = os.path.join(args.output_dir, "latest")
            if os.path.islink(latest_link):
                os.remove(latest_link)
            os.symlink(checkpoint_name, latest_link)
            if args.load_best_model_at_end:
                best_link = os.path.join(args.output_dir, "best")
                if os.path.islink(best_link):
                    os.remove(best_link)
                os.symlink(state.best_model_checkpoint.split("/")[-1], best_link)

class PausingCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, optimizer=None, **kwargs):
        if args.should_save:
            paused_path = os.path.join(args.output_dir, "paused")
            paused_model_path = paused_path+"_model.pickle"
            paused_optimizer_path = paused_path+"_optimizer.pickle"
            if os.path.exists(paused_path):
                status(f"Found {paused_path} - pausing training and saving to disk")
                torch.save(model.state_dict(), paused_model_path)
                torch.save(optimizer.state_dict(), paused_optimizer_path)
                dummy = torch.Tensor()
                for param in model.parameters():
                    param.data = dummy
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        param.data = dummy
                #TODO REMOVE WEIGHTS
                while os.path.exists(paused_path):
                    time.sleep(10)
                status(f"Pause file {paused_path} vanished - loading from disk and resuming ")
                model.load_state_dict(torch.load(paused_model_path))
                optimizer.load_state_dict(torch.load(paused_optimizer_path))
                status("Resumed")

def get_peft_config(model, config):
    start("Target modules")
    if not config.peft_config["target_modules"]:
        target_modules = set()
        for name, module in model.named_modules():
            if type(module).__name__ in ("Linear", "Linear4bit"):
                target_modules.add(name.split(".")[-1])
        config.peft_config["target_modules"] = list(target_modules)
    config = LoraConfig(**config.peft_config)
    status(target_modules)
    return config

def get_collator(tokenizer, instruction_template, response_template):
    if response_template is None:
        return None
    response_template = tokenizer(response_template, add_special_tokens=False)["input_ids"][1:]
    instruction_template = None if instruction_template is None else tokenizer(instruction_template, add_special_tokens=False)["input_ids"][1:]
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
    )
    return collator

def do_train(trainer, output_dir, resume_from_checkpoint):
    torch.set_warn_always(False)
    start("Training")
    trainable_params, all_params = trainable_parameters(trainer.model)
    status(f"#trainable-params: {trainable_params}; #all-params: {all_params}; %trainable: {100 * trainable_params / all_params}", end='')
    if trainer.eval_dataset is not None:
        print(trainer.evaluate())
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if trainer.eval_dataset is not None:
        print(trainer.evaluate())
    end()
    start("Saving model")
    trainer.save_model(output_dir)
    end()

def get_training_args(
    output_dir,
    eval_dataset,
    config,
):
    if config.training_args["metric_for_best_model"] is None and isinstance(eval_dataset, dict):
        config.training_args["metric_for_best_model"] = f"eval_{next(iter(eval_dataset))}_loss"
    if eval_dataset is None:
        config.training_args["evaluation_strategy"] = "no"
        config.training_args["load_best_model_at_end"] = False
        config.training_args["do_eval"] = False
    config.training_args["output_dir"] = output_dir
    training_arguments = TrainingArguments(
        **config.training_args,
    )
    return training_arguments

@click.group()
def _train():
    pass
@_train.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("base-model", type=click.Path(exists=True))
@click.argument("trained-model", type=click.Path(exists=False))
@click.option("--info", default=None, type=bool)
@click.option("--peft", default=None, type=bool)
@click.option("--early-stopping-patience", default=None, type=int)
@click.option("--instruction-template", default=None, type=str)
@click.option("--response-template", default=None, type=str)
@click.option("--resume-from-checkpoint", default=None, type=str)
@click.option("--every", "-e", default=None, type=int)
@click.option("--layers", "-t", default=None, type=str)
@click.option("--experts", "-x", default=None, type=str)
@click.option("--train-experts", "-a", default=None, type=int)
@click.option("--freeze", default=None, type=str, multiple=True)
@click.option("--bitlinear", default=None, type=str)
@click.option("--bitlinear-debug", default=None, type=str)
@args_config
def train(args, config):
    return __train(args, config)

def __train(args, config):
    if config.training_args["run_name"] is None:
        config.training_args["run_name"] = f"{config.command}_{args.trained_model.split('/')[-1]}"
    if not args.peft and config.model["load_in_4bit"]:
        status("deactivating load_in_4bit for full training")
        config.model["load_in_4bit"] = False
    tokenizer = load_tokenizer(args.base_model, config)
    train_dataset, eval_dataset = load_datasets(tokenizer=tokenizer, config=config)
    training_args = get_training_args(output_dir=args.trained_model, eval_dataset=eval_dataset, config=config)
    if args.info:
        start("Printing effective training arguments")
        status(training_args)
    if config.model["device_map"] == "auto" and training_args.local_rank != -1:
        config.model["device_map"] = get_device_map()
    model = load_model(
        args.base_model,
        config=config,
    )
    peft_config = get_peft_config(
        model=model,
        config=config,
    ) if args.peft else None
    if args.every is not None:
        start("Freezing layers")
        for i, layer in enumerate(model.get_submodule(args.layers)):
            status(i, end='')
            if i % args.every != args.every-1:
                for name, param in layer.named_parameters():
                    param.requires_grad = False
                    status(name, end='')
        end()
    if args.train_experts is not None:
        start("Freezing experts")
        for i, layer in enumerate(list(model.get_submodule(args.layers))):
            status(i, end='')
            experts = layer.get_submodule(args.experts)
            for expert in experts[:len(experts)-args.train_experts]:
                for name, param in expert.named_parameters():
                    param.requires_grad = False
                    status(name, end='')
        end()
    if args.freeze is not None:
        start("Freezing parameters")
        for name, param in model.named_parameters():
            for freeze in args.freeze:
               if re.search(freeze, name):
                    param.requires_grad = False
                    status(f"{freeze} matched {name}", end='')
                    continue
        end()
    if args.bitlinear is not None:
        start("Replacing nn.Linear with BitLinear")
        replace_modules(model, new_class_kwargs={"measure": args.bitlinear})
        model.to(model.device)
        model.to(config.model["torch_dtype"])
        print(model)
        end()
    start("Setting up training")
    callbacks = [MetadataSavingCallback(config), PausingCallback()]
    if args.bitlinear_debug is not None:
        bitlinear_callback = BitLinearCallback(args.bitlinear_debug)
        callbacks.append(bitlinear_callback)
    if args.early_stopping_patience is not None and args.early_stopping_patience >= 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    status(f"callbacks: {callbacks}", end='')
    collator = get_collator(
        tokenizer=tokenizer,
        instruction_template=args.instruction_template,
        response_template=args.response_template,
    )
    trainer = OdinTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        data_collator=collator,
        **config.trainer_args,
    )
    end()
    start("Saving metadata on main process")
    accelerator = Accelerator()
    if accelerator.is_main_process:
        save_metadata(model.metadata, config, args.trained_model)
    accelerator.wait_for_everyone()
    end()
    do_train(
        trainer,
        output_dir=args.trained_model,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
