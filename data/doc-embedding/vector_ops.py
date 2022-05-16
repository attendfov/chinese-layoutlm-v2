import random
import time
import json

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


class milvusServer:
    def __init__(self, dim, host='localhost', collection_name='defaultName'):
        connections.connect("default", host=host, port="19530")

        self.dim = dim
        self.collection_name = collection_name
        has = utility.has_collection(self.collection_name)

        drop_collection = True
        if has and drop_collection:
            utility.drop_collection(self.collection_name)

        self.milvus_collection = self.create_collection()

    def create_collection(self):
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="label", dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]

        schema = CollectionSchema(fields, "")

        return Collection(self.collection_name, schema, consistency_level="Strong")

    def save_vector(self, entities):
        self.milvus_collection.insert(entities)

        ################################################################################
        # 4. create index
        # We are going to create an IVF_FLAT index for hello_milvus collection.
        # create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }

        self.milvus_collection.create_index("embeddings", index)

        connections.disconnect('default')


def read_entities():
    with open('embeddings/predictions_embedding.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    emb = data['pred']
    labels = data['label']
    print('entities num is %d' % len(labels))

    entities = [
        [i for i in range(len(labels))],
        [int(label) for label in labels],
        emb
    ]
    return entities


def read_entities_2():
    with open('embeddings/predictions_embedding_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    emb = data['reps']
    labels = data['doc-labels']
    print('entities num is %d' % len(labels))

    entities = [
        [i for i in range(len(labels))],
        [int(label) for label in labels],
        emb
    ]
    return entities


if __name__ == '__main__':
    entities = read_entities_2()

    milvus_server = milvusServer(dim=768, host='localhost', collection_name='Gartner_train')

    milvus_server.save_vector(entities)
