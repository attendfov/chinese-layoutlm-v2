# Lint as: python3
import json
import logging
import os
import copy
import random
import itertools
import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import collections

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer


_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


def sample_neg_item(item, other_labels, label2items, neg_pos_rate):
    label2items = collections.OrderedDict(label2items)
    length_list = [len(label2items[other_label]) for other_label in other_labels]
    p_list = np.log([v + 1 for v in length_list]) * 1000

    all_other_items = []
    all_other_p = []
    for p, other_label in zip(p_list, other_labels):
        _len = len(label2items[other_label])
        p_per = p / _len
        all_other_p.extend([p_per] * _len)
        all_other_items.extend(label2items[other_label])

    negs = set()
    while len(negs) <= neg_pos_rate:
        negs.add(random.choices(range(len(all_other_items)), weights=all_other_p, k=1)[0])
    return [(item, all_other_items[neg], 'neg') for neg in negs]


def gene_doc_embedding_train_data(label2items):
    labels = list(label2items.keys())
    neg_pos_rate = 10
    train_data = []
    for label, items in label2items.items():
        other_label = copy.deepcopy(labels)
        other_label.remove(label)

        if len(items) == 1:
            item = items[0]
            train_data.extend(sample_neg_item(item, other_label, label2items, neg_pos_rate))
        else:
            pos_items = []
            pos_pairs = itertools.permutations(range(len(items)), 2)
            for pos_pair in pos_pairs:
                pos_items.append((items[pos_pair[0]], items[pos_pair[1]], 'pos'))
            train_data.extend(pos_items)

            for item in items:
                train_data.extend(sample_neg_item(item, other_label, label2items, neg_pos_rate))
    return train_data


def gene_doc_embedding_predict_data(items):
    train_data = []
    for item in items:
        train_data.append((item, item, 'pos'))
    return train_data


def _generate_examples(filepaths):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    for filepath in filepaths:
        logger.info("Generating examples from = %s", filepath)
        with open(filepath[0], "r") as f:
            data = json.load(f)

        # 将文档分组
        label2doc = {}
        all_items = []
        label2id = {label: i for i, label in enumerate(['pos', 'neg'])}

        doc_num = 0
        for _, doc in enumerate(data["documents"]):
            doc_num += 1
            doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
            image, size = load_image(doc["img"]["fpath"])

            doc_label = doc['label']
            document = doc["document"]
            item = {"input_ids": [], "bbox": []}

            text = []

            for line in document:
                if len(line["text"]) == 0:
                    continue

                text.append(line['text'])

                tokenized_inputs = tokenizer(
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
                            tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
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
            cls_id = tokenizer.convert_tokens_to_ids('[CLS]')

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

            all_items.append(item)
            if doc_label in label2doc.keys():
                label2doc[doc_label] += [item]
            else:
                label2doc[doc_label] = [item]
        print('length of documents is %d' % doc_num)

        train_data = gene_doc_embedding_train_data(label2doc)
        train_data = [copy.deepcopy((item[0], item[1], label2id[item[2]])) for item in train_data]
        train, dev = train_test_split(train_data, test_size=0.1, random_state=42, shuffle=True, )
        test = gene_doc_embedding_predict_data(all_items)
        test = [(item[0], item[1], label2id[item[2]]) for item in test]
        return train, dev, test


if __name__ == '__main__':
    _generate_examples()
