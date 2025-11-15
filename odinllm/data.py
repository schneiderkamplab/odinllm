from dataclasses import dataclass
import itertools
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional, Dict, Any, Union

class ConstantLengthDataset(IterableDataset):
    """
    TRL ConstantLengthDataset-compatible packer (behaviorally close),
    with knobs to match old runs.
    """
    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field="text",
        seq_length=2048,
        infinite=False,
        # Parity knobs:
        add_special_tokens=False,      # TRL often False
        append_concat_token=True,      # TRL commonly True
        concat_token_id=None,          # default to eos_token_id if None
        pad_tail=False,                # TRL train: usually False; eval may be True
        return_attention_mask=False,   # match what your collator expected
        chars_per_token=None,          # only for __len__ approx
        shuffle_each_epoch=False,      # TRL usually relied on upstream shuffling
        seed=0,                        # deterministic shuffling
    ):
        self.tok = tokenizer
        self.ds = dataset
        self.field = dataset_text_field
        self.L = int(seq_length)
        self.infinite = bool(infinite)
        self.add_special = bool(add_special_tokens)
        self.concat_token_id = (
            concat_token_id if concat_token_id is not None else tokenizer.eos_token_id
        )
        self.append_concat = bool(append_concat_token)
        self.pad_tail = bool(pad_tail)
        self.return_mask = bool(return_attention_mask)
        self.cpt = chars_per_token
        self.shuffle_each_epoch = bool(shuffle_each_epoch)
        self.seed = int(seed)

        if self.L <= 0:
            raise ValueError("seq_length must be > 0")

    def _extract_text(self, ex):
        if isinstance(ex, dict):
            return ex.get(self.field, "")
        return str(ex)

    def _epoch_iterator(self):
        data = list(self.ds)
        if self.shuffle_each_epoch:
            # Seed per epoch + worker to match DDP dataloaders deterministically.
            wi = get_worker_info()
            wseed = (self.seed, 0 if wi is None else wi.id)
            rng = random.Random(hash(wseed) & 0xFFFFFFFF)
            rng.shuffle(data)
        return itertools.cycle(data) if self.infinite else iter(data)

    def __iter__(self):
        # Per-worker independent buffer to avoid boundary bleed
        _ = get_worker_info()
        buf = []
        for ex in self._epoch_iterator():
            text = self._extract_text(ex)
            ids = self.tok(
                text,
                add_special_tokens=self.add_special,
                return_attention_mask=False,
                return_tensors=None,
            )["input_ids"]
            if self.append_concat and self.concat_token_id is not None:
                ids.append(self.concat_token_id)

            buf.extend(ids)

            while len(buf) >= self.L:
                chunk = buf[:self.L]
                del buf[:self.L]
                item = {"input_ids": chunk, "labels": chunk}
                if self.return_mask:
                    # All ones because we don't emit padded chunks here
                    item["attention_mask"] = [1] * self.L
                yield item

        # Tail
        if not self.infinite and self.pad_tail and len(buf) > 0:
            pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
            chunk = buf + [pad_id] * (self.L - len(buf))
            item = {"input_ids": chunk, "labels": chunk}
            if self.return_mask:
                item["attention_mask"] = [1] * len(buf) + [0] * (self.L - len(buf))
            yield item

    def __len__(self):
        # Approximate; safe for schedulers/progress bars
        try:
            if self.cpt and self.cpt > 0:
                total_chars = 0
                for ex in self.ds:
                    s = self._extract_text(ex)
                    # +1 approximates a concat token (EOS) per example if enabled
                    total_chars += len(s) + (1 if self.append_concat else 0)
                approx_tokens = total_chars / float(self.cpt)
                n = int(approx_tokens // self.L)
            else:
                # Slow exact estimate; tokenize each record
                total_tokens = 0
                for ex in self.ds:
                    s = self._extract_text(ex)
                    ids = self.tok(s, add_special_tokens=self.add_special)["input_ids"]
                    if self.append_concat and self.concat_token_id is not None:
                        total_tokens += len(ids) + 1
                    else:
                        total_tokens += len(ids)
                n = int(total_tokens // self.L)
            if self.pad_tail and not self.infinite:
                # account for remainder as one padded chunk
                return n + 1 if n * self.L < total_tokens else n
            return max(1, n)
        except Exception:
            return 0

@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Completion-only loss collator compatible with HF Trainer/SFTTrainer (packing=False).
    Masks labels before the completion so the loss is computed only on the assistant span.

    Args:
        tokenizer: HF tokenizer.
        response_template: String marker in the *tokenized* sequence that precedes the completion
                           (e.g., "### Assistant:" or "<|assistant|>\n").
        response_template_ids: Optional pre-tokenized ids for the template. If provided, these take precedence.
        max_length: Optional max length for truncation.
        pad_to_multiple_of: Pad sequence lengths to a multiple (e.g., 8 for Tensor Cores).
        label_pad_token_id: Usually -100 for LM loss masking.
        add_eos_token: Append EOS if missing.
    """
    tokenizer: Any
    response_template: Optional[str] = None
    response_template_ids: Optional[List[int]] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    add_eos_token: bool = True

    def _ensure_template_ids(self) -> List[int]:
        if self.response_template_ids is not None:
            return list(self.response_template_ids)
        if not self.response_template:
            raise ValueError("Provide either response_template_ids or response_template.")
        # We deliberately avoid adding special tokens for the template itself
        return self.tokenizer(self.response_template, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _find_subsequence(seq: List[int], pat: List[int]) -> int:
        # return start index of first occurrence of pat in seq, or -1
        if not pat or len(pat) > len(seq):
            return
