# Lint as: python3
import json
import logging
import os

import datasets

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer

_URL = os.path.join(os.getcwd(), "../data/gartner_data/data/")
print(_URL)

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "labels": datasets.Value('string'),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        test_files_for_many_langs = [downloaded_files["test"]]
        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):

        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r") as f:
                data = json.load(f)

            for doc_id, doc in enumerate(data["documents"]):
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])

                doc_label = doc['label']
                document = doc["document"]
                item = {"input_ids": [], "bbox": []}

                for line in document:
                    if len(line["text"]) == 0:
                        continue

                    tokenized_inputs = self.tokenizer(
                        line["text"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                    )

                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    last_box = None

                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        if token_id == 6:
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box

                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]

                    tokenized_inputs.update({"bbox": bbox})

                    for i in item:
                        item[i] = item[i] + tokenized_inputs[i]

                input_ids = item['input_ids']
                bbox = item['bbox']

                # tokenizer.cls_token对应<s>
                # cls_id = tokenizer.cls_token_id
                # 真正的[CLS]
                cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')

                input_ids.insert(0, cls_id)
                bbox.insert(0, [0, 0, 0, 0])

                chunk_size = 512
                item['input_ids'] = input_ids[:chunk_size]
                item['bbox'] = bbox[:chunk_size]
                item.update(
                    {
                        "id": f"{doc['id']}",
                        "image": image,
                        "labels": doc_label
                    }
                )
                yield f"{doc['id']}", item



