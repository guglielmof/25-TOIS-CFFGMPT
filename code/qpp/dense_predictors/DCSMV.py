

from ..AbstractPredictor import AbstractPredictor
import numpy as np
import numpy.random as npr
import pandas as pd
import rbo
import sys
from typing import *


class DCSMV(AbstractPredictor):

    def __init__(self, *args, **kwargs):
        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)
        self.hyperparams_identifiers = ["k"]

    def _local_predict(self, query: pd.Series) -> float:

        local_run = self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)]
        scores = local_run.score.to_list()
        corpusScore = np.dot(query.representation, self.docs_encoder.get_centroid())
        prediction = np.mean(scores * np.abs(np.log(scores / np.mean(scores)))) / corpusScore

        return prediction
