from ..AbstractPredictor import AbstractPredictor
import numpy as np
import numpy.random as npr
import pandas as pd
import rbo
import sys
from typing import *


def retrieve_faiss(qembs: np.array, indexWrapper, k: Optional[int] = 1000) -> pd.DataFrame:
    """
    :param qembs: np.array matrix containing the embeddings of the queries
    :param indexWrapper: a structure with two fields: index (a faiss index) and mapper (a list to map back integers to doc ids)
    :param k: the number of documents retrieved
    :return:
    """
    ip, idx = indexWrapper.index.search(qembs, k)

    run = pd.DataFrame({"query_id": np.arange(qembs.shape[0]), "doc_id": list(idx), "score": list(ip)}).explode(["doc_id", "score"])
    run.doc_id = run.doc_id.map(lambda x: indexWrapper.mapper[x])
    run.score = run.score.astype(float)

    return run


class Nsigma(AbstractPredictor):

    def __init__(self, *args, **kwargs):
        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)
        self.hyperparams_identifiers = ["k", "sigma", "reps"]

        # sample the noise (reps time to avoid fluctuation). sigma is set to 0.25 for ANCE. reps is 30.
        self.noise = npr.normal(0, self.sigma, (self.reps, 768))

    def _local_predict(self, query: pd.Series) -> float:
        # take the first k docs of the run (and the query) for which you are trying to predict the performance
        local_run = self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)]
        local_docs = local_run.sort_values("rank")["doc_id"].to_list()

        # take the dense representation of the query (these are precomputed on the "query" pd.Series
        qvec = query.representation


        qvec_perturbed = qvec[np.newaxis, :] + self.noise

        # compute the run using the perturbed queries
        run_perturbed = retrieve_faiss(qvec_perturbed, self.indexWrapper, k=self.k)
        run_perturbed["rank"] = run_perturbed.groupby("query_id")["score"].rank(ascending=False, method="first").astype(int)

        def rbo_similarity(v: pd.DataFrame) -> float:
            return rbo.RankingSimilarity(v.sort_values("rank")["doc_id"].to_list(), local_docs).rbo(p=1)

        # compute the rbo similarity between the original and perturbed run for each perturbed query and average
        prediction = run_perturbed.groupby("query_id").apply(rbo_similarity).mean()

        return prediction
