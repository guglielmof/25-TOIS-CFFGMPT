from .AbstracyPTPredictor import AbstractPTPredictor
import numpy as np
import pandas as pd
import sys
from scipy.stats import pearsonr
from .Clarity import Clarity
from .WIG import WIG
from ..score_predictors import NQCscores, SMVscores


class AbstractUEF(AbstractPTPredictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_predictor = None
        self.hyperparams_identifiers = ["k"]
        self.max_freq = 0.1
        self.top_m = 100
        self.considered_docs = 150

    def _local_predict(self, query):
        # select k elements

        p = self.base_predictor._local_predict(query)

        local_run = self.run.loc[(self.run["rank"] <= self.k) & (self.run.query_id == query.query_id)]

        scores = local_run.score.to_list()
        docsid = local_run.doc_id.to_list()
        # print(normalized_scores)
        # create the relevance model
        normalized_scores = scores / np.sum(scores)
        rm = self.index_wrapper.construct_relevance_model(docsid, normalized_scores, self.top_m, self.max_freq)

        # score the documents according to the rm
        lmscores = [self.index_wrapper.score_with_rm(d, rm) for d in docsid]
        pred_score = p * pearsonr(scores, lmscores)[0]
        return pred_score


class UEFClarity(AbstractUEF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_predictor = Clarity(**kwargs)


class UEFWIG(AbstractUEF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_predictor = WIG(**kwargs)


class UEFNQC(AbstractUEF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_predictor = NQCscores(**kwargs)


class UEFSMV(AbstractUEF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_predictor = SMVscores(**kwargs)
