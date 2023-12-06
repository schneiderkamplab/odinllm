from functools import wraps
from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer

class OdinTrainer(SFTTrainer):

    def __init__(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = False,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        **kwargs,
    ):
        if eval_dataset is not None:
            multiple = isinstance(eval_dataset, dict)
            eval_datasets = eval_dataset if multiple else {"singleton": eval_dataset}
            for eval_dataset_name, _eval_dataset in eval_datasets.items():
                eval_datasets[eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    tokenizer,
                    packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    infinite,
                    num_of_sequences,
                    chars_per_token,
                )
            if not multiple:
                eval_dataset = eval_datasets["singleton"]
        super().__init__(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            **kwargs,
        )

    @wraps(SFTTrainer.evaluate)
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            *args,
            **kwargs,
        )

    @wraps(SFTTrainer._prepare_dataset)
    def _prepare_dataset(
        self,
        dataset,
        *args,
    ):
        if isinstance(dataset, dict):
            return dataset
        return super()._prepare_dataset(dataset, *args)
