from functools import wraps
from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer

class OdinTrainer(SFTTrainer):

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
        **kwargs,
    ):
        if isinstance(dataset, dict):
            return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)
