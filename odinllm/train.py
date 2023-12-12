from accelerate import Accelerator
import click
import os
from peft import LoraConfig
import torch
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from trl.trainer import DataCollatorForCompletionOnlyLM

from .shared import load_datasets, load_model, load_tokenizer, save_metadata
from .trainer import OdinTrainer
from .utils import (
    args_config,
    end,
    get_device_map,
    start,
    status,
    trainable_parameters,
)

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

def do_train(trainer, output_dir):
    torch.set_warn_always(False)
    start("Training")
    trainable_params, all_params = trainable_parameters(trainer.model)
    status(f"#trainable-params: {trainable_params}; #all-params: {all_params}; %trainable: {100 * trainable_params / all_params}", end='')
    if trainer.eval_dataset is not None:
        print(trainer.evaluate())
    trainer.train()
    if trainer.eval_dataset is not None:
        print(trainer.evaluate())
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
    if config.training_args["run_name"] is None:
        config.training_args["run_name"] = output_dir
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        **config.training_args,
    )
    return training_arguments    

@click.group()
def _train():
    pass
@_train.command()
@click.argument("config", type=click.Path(exists=True), nargs=-1)
@click.argument("base-model", type=click.Path(exists=True))
@click.argument("trained-model", type=click.Path(exists=False))
@click.option("--peft", default=None, type=bool)
@click.option("--early-stopping-patience", default=None, type=int)
@click.option("--instruction-template", default=None, type=str)
@click.option("--response-template", default=None, type=str)
@args_config
def train(args, config):
    if not args.peft and config.model["load_in_4bit"]:
        status("deactivating load_in_4bit for full training")
        config.model["load_in_4bit"] = False
    tokenizer = load_tokenizer(args.base_model, config)
    train_dataset, eval_dataset = load_datasets(tokenizer=tokenizer, config=config)
    training_args = get_training_args(output_dir=args.trained_model, eval_dataset=eval_dataset, config=config)
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
    start("Setting up training")
    callbacks = [MetadataSavingCallback(config)]
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
    )
