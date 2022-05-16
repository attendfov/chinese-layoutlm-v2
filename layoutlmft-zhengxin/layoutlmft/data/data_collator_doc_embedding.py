from dataclasses import dataclass
from typing import Optional, Union

import torch

from detectron2.structures import ImageList
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, _features):
        batches = {'batch1': None, 'batch2': None, 'labels': None}
        labels = [feature[2] for feature in _features]
        labels = torch.tensor(labels, dtype=torch.int64)
        batches['labels'] = labels
        for i in range(2):
            features = [feature[i] for feature in _features]

            image = ImageList.from_tensors([torch.tensor(feature["image"]) for feature in features], 32)
            # for feature in features:
                # if 'image' in feature:
                #     del feature["image"]

            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                # Conversion to tensors will fail if we have labels as they are not of the same length yet.
                return_tensors="pt" if labels is None else None,
            )

            sequence_length = torch.tensor(batch["input_ids"]).shape[1]
            padding_side = self.tokenizer.padding_side
            if padding_side == "right":
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            else:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]

            del batch['id']
            del batch['labels']
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

            batch["image"] = image
            if i == 0:
                batches['batch1'] = batch
            else:
                batches['batch2'] = batch

        return batches
