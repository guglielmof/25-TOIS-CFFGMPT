from ..AbstractPredictor import AbstractPredictor
import pandas as pd
import numpy as np
class SMVscores(AbstractPredictor):

    def __init__(self, *args, **kwargs):
        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)
        self.hyperparams_identifiers = ["k"]

    def _local_predict(self, query: pd.Series) -> float:

        corpus_score = np.mean(self.run.loc[(self.run.query_id == query.query_id), "score"])

        # take the first k docs of the run (and the query) for which you are trying to predict the performance
        local_run = self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)]
        score_list = local_run.score

        mu = np.mean(score_list)
        prediction = np.mean(np.array(score_list) * abs(np.log(score_list / mu))) / corpus_score

        return prediction
