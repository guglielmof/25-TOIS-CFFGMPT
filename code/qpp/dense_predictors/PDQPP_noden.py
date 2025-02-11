from ..AbstractPredictor import AbstractPredictor
import numpy as np
import numpy.random as npr
import pandas as pd
import rbo
import sys
from typing import *


def projection(a, b):
    scale = (np.dot(a, b) / (np.linalg.norm(b) ** 2)).reshape(-1, 1)
    proj = np.multiply(b, scale)
    if len(a.shape) == 1 or a.shape[1] == 1:
        proj = proj[0]
    return proj


def _compute_local_predictor(qemb, demb, vemb):
    qproj = projection(qemb, vemb)
    dproj = projection(demb, vemb)

    original_scores = np.dot(qemb, demb.T)
    projected_scores = np.dot(qproj, dproj.T)

    return np.std(original_scores - projected_scores)


class PDQPP_noden(AbstractPredictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams_identifiers = ["k1", "k3"]

    def _local_predict(self, query: pd.Series) -> float:

        local_run = self.run.loc[(self.run.query_id == query.query_id)]

        pseudo_relevant = list(local_run.loc[local_run["rank"] <= self.k1, "doc_id"].values)
        top_k_docs = list(local_run.loc[local_run["rank"] <= self.k3, "doc_id"].values)

        pseudo_relevant_embs = self.docs_encoder.get_encoding(pseudo_relevant)
        top_k_docs_embs = self.docs_encoder.get_encoding(top_k_docs)

        local_pred = np.array([_compute_local_predictor(query.representation, top_k_docs_embs, d) for d in pseudo_relevant_embs])

        prediction = - np.mean(local_pred)

        return prediction
