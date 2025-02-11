from .AbstracyPTPredictor import AbstractPTPredictor
import numpy as np
import pandas as pd
import sys


class Clarity(AbstractPTPredictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams_identifiers = ["k"]
        self.max_freq = 0.1
        self.top_m = 100

    def _local_predict(self, query: pd.Series) -> float:
        # select k elements
        local_run = self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)]

        scores = local_run.score.to_list()
        normalized_scores = scores / np.sum(scores)

        rm = self.index_wrapper.construct_relevance_model(local_run.doc_id.to_list(), normalized_scores, self.top_m, self.max_freq)

        rm["CF"] = rm.term.apply(self.index_wrapper.getNormalizedCF)
        rm["weight"] = rm["freq"] * np.log(rm["freq"] / rm["CF"])

        pred_score = rm.weight.sum()
        return pred_score
