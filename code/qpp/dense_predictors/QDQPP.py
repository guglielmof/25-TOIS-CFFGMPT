from ..AbstractPredictor import AbstractPredictor
import numpy as np
import numpy.random as npr
import pandas as pd
import rbo
import sys
from typing import *
from PDQPP import PDQPP, projection, _compute_local_predictor


class QDQPP(PDQPP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams_identifiers = ["k1", "k2", "k3"]

    def _local_predict(self, query: pd.Series) -> float:

        local_run = self.run.loc[(self.run.query_id == query.query_id)]

        pseudo_relevant = list(local_run.loc[local_run["rank"] <= self.k1, "doc_id"].values)
        top_k_docs = list(local_run.loc[local_run["rank"] <= self.k3, "doc_id"].values)
        den = local_run.loc[local_run["rank"] <= self.k2, "score"].std()

        pseudo_relevant_embs = self.docs_encoder.get_encoding(pseudo_relevant)
        top_k_docs_embs = self.docs_encoder.get_encoding(top_k_docs)

        local_pred = np.array([_compute_local_predictor(query.representation, top_k_docs_embs, query.representation+npr.normal(scale=self.sigma, size=top_k_docs_embs.shape[1])) for d in pseudo_relevant_embs])

        prediction = - np.mean(local_pred) / den

        return prediction
