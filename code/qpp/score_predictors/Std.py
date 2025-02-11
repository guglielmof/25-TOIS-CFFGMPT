from ..AbstractPredictor import AbstractPredictor
import pandas as pd

class Std(AbstractPredictor):

    def __init__(self, *args, **kwargs):
        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)
        self.hyperparams_identifiers = ["k"]

    def _local_predict(self, query: pd.Series) -> float:
        # take the first k docs of the run (and the query) for which you are trying to predict the performance
        local_run = self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)]

        return local_run.score.std()
