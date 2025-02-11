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


class DenseQPP(AbstractPredictor):

    def __init__(self, *args, **kwargs):
        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)
        self.hyperparams_identifiers = ["k", "sigma", "reps"]

    def _local_predict(self, query: pd.Series) -> float:
        """
        this function is applied (through pd.DataFrame.apply) individually to each query.
        :param query: query is pd.Series with three fields <query_id, text, representation>
        :return: the prediction, according to DenseQPP of the performance of the query
        """

        # take the first k (validated from 100 to 1000) docs of the run (and the query) for which you are trying to predict the performance
        local_run = self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)]
        local_docs = local_run.sort_values("rank")["doc_id"].to_list()

        # take the dense representation of the query (these are available as precomputed on the "query" pd.Series)
        qvec = query.representation

        # sample the noise ("reps" time to avoid fluctuation). "sigma" is set to 0.25 for ANCE. reps is 30.
        self.noise = npr.normal(0, self.sigma, (self.reps, len(qvec)))

        # perturb the query
        qvec_perturbed = qvec[np.newaxis, :] + self.noise

        # compute the run using the perturbed queries (and add the "rank" column)
        run_perturbed = retrieve_faiss(qvec_perturbed, self.indexWrapper, k=self.k)
        run_perturbed["rank"] = run_perturbed.groupby("query_id")["score"].rank(ascending=False, method="first").astype(int)

        # compute the rbo similarity between the original and perturbed run for each perturbed query and average
        def rbo_similarity(v: pd.DataFrame) -> float:
            return rbo.RankingSimilarity(local_docs, v.sort_values("rank")["doc_id"].to_list()).rbo()

        prediction = run_perturbed.groupby("query_id").apply(rbo_similarity).mean()

        return prediction
