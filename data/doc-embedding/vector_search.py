import random
import time

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


class vectorSearch:
    def __init__(self, host, collection_name):
        connections.connect("default", host=host, port="19530")

        self.milvus_collection = Collection(collection_name)

        self.milvus_collection.load()

        self.search_params = {
            "metric_type": "l2"
        }

    def search(self, vectors):

        result = self.milvus_collection.search(vectors, "embeddings", self.search_params, limit=5,
                                               output_fields=["pk", "label"])
        return result

    def search2(self, vectors, label):
        result = self.milvus_collection.search(vectors, "embeddings", self.search_params, limit=3, expr="label > 3",
                                               output_fields=["pk"])
        return result

    def show(self, result):
        for hits in result:
            for hit in hits:
                print(f"hit: {hit}, label field: {hit.entity.get('label')}")

    def query(self):
        result = self.milvus_collection.query(expr="pk < 100", output_fields=["label", "embeddings"])
        return result

    def delete(self):

        expr = f"pk in [2, 3]"
        self.milvus_collection.delete(expr)

        result = self.milvus_collection.query(expr=expr, output_fields=["label", "embeddings"])


if __name__ == '__main__':
    vector_search = vectorSearch(host='localhost', collection_name='Gartner_train')
    print(utility.list_collections())

    vectors = []
    labels = []
    for v in vector_search.query():
        vectors.append(v['embeddings'])
        labels.append(v['label'])
    print('all labels:')
    print(labels)

    f = open('vector-metrics/layoutxlm-doc-embedding-512-train.txt',  'w', encoding='utf-8')

    f.write('所有的向量：\n')
    f.write(' '.join([str(l) for l in labels]) + '\n')

    results = vector_search.search(vectors)
    for index, hits in enumerate(results):
        search_vec_index = labels[index]
        f.write('向量label:%d \n'%search_vec_index)
        for hit in hits:
            f.write('最近的向量:\n')
            f.write(f"距离: {hit.distance}, label field: {hit.entity.get('label')}\n")
        f.write('--' * 40 + '\n')




