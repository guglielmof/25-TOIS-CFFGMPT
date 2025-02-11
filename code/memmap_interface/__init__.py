import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class AbsractMemmapEncoding:

    def __init__(self, datapath, mappingpath, embedding_size=768, index_name="id", sep=","):
        self.data = np.memmap(datapath, dtype=np.float32, mode="r").reshape(-1, embedding_size)
        self.id2int = {}

        with open(mappingpath, "r") as fp:
            for l in fp.readlines():
                idx, offset = l.strip().split(sep)
                if offset.isnumeric():
                    self.id2int[idx] = int(offset)

        self.int2id = {v: k for k, v in self.id2int.items()}

        self.shape = self.get_shape()

    def get_position(self, idx):
        if type(idx) == list or type(idx) == np.array:
            return [self.id2int[i] for i in idx]
        else:
            return self.id2int[idx]

    def get_inverse_position(self, offset):
        if type(offset) == list or type(offset) == np.array:
            return [self.int2id[i] for i in offset]
        else:
            return self.int2id[offset]

    def get_encoding(self, idx):
        # print(self.mapping.loc[idx, "offset"])
        return self.data[self.get_position(idx)]

    def get_centroid(self):
        if not hasattr(self, "centroid"):
            self.centroid = np.mean(self.data, axis=0)
        return self.centroid

    def normalize_data(self):
        if not hasattr(self, "normalized_data"):
            self.normalized_data = normalize(self.data)

    def get_normalized_encoding(self, idx):
        self.normalize_data()
        return self.normalized_data[self.get_position(idx)]

    def get_data(self, normalized=False):
        if normalized:
            self.normalize_data()
            return self.normalized_data
        else:
            return self.data

    def get_shape(self):

        return self.data.shape

    def get_ids(self):
        return list(self.id2int.keys())

    def dot(self, vector):
        return np.dot(vector, self.data.T)

    def get_mapper(self):
        return self.int2id


class MemmapCorpusEncoding(AbsractMemmapEncoding):

    def __init__(self, datapath, mappingpath, embedding_size=768):
        super().__init__(datapath, mappingpath, embedding_size, index_name="doc_id")


class MemmapQueriesEncoding(AbsractMemmapEncoding):
    def __init__(self, datapath, mappingpath, embedding_size=768):
        super().__init__(datapath, mappingpath, embedding_size, index_name="qid", sep="\t")
        # self.data = self.data[self.mapping.offset, :]
        # self.mapping.offset = np.arange(len(self.mapping.index))
