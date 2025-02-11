from ..AbstractPredictor import AbstractPredictor
import numpy as np
import numpy.random as npr
import pandas as pd
import rbo
import sys
from typing import *


def _compute_volume(dots):
    l = np.abs(np.max(dots, axis=0) - np.min(dots, axis=0))

    return 1 / np.sum(np.log(l))


class Hypervolume(AbstractPredictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams_identifiers = ["k"]

    def _local_predict(self, query: pd.Series) -> float:
        top_k_docs = list(self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k), "doc_id"].values)
        top_k_docs_embs = self.docs_encoder.get_encoding(top_k_docs)

        embs = np.append(top_k_docs_embs, query.representation[np.newaxis, :], axis=0)

        return _compute_volume(embs)
