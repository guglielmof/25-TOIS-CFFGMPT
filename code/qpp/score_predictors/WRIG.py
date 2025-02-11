import pandas as pd
import numpy as np
import rbo
from ..AbstractPredictor import AbstractPredictor


class WRIG(AbstractPredictor):

    def __init__(self, *args, **kwargs):
        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)
        self.hyperparams_identifiers = ["k"]
        self.variants_run[["topic_id", "var_id"]] = self.variants_run.query_id.str.split("_", expand=True)
        self.variants_run = self.variants_run.loc[self.variants_run["var_id"] != "0"]

    def _internal_heuristics(self, run):
        return run.score.std() ** 2

    def _local_predict(self, query: pd.Series) -> float:
        local_run = self.run.loc[(self.run["rank"] <= self.k) & (self.run["query_id"] == query.query_id)]
        doc_ids = local_run.doc_id.to_list()
        var_runs = self.variants_run.loc[(self.variants_run["rank"] <= self.k) & (self.variants_run.topic_id == query.query_id)]

        var_sims = var_runs.groupby("query_id")["doc_id"].apply(lambda x: rbo.RankingSimilarity(doc_ids, x.to_list()).rbo(p=1))
        var_pred = var_runs.groupby("query_id")["score"].std() ** 2

        wrig_scores = pd.DataFrame({"sims": var_sims, "pred": var_pred})

        den = local_run.score.std() ** 2 * wrig_scores.sims.sum()

        if den == 0:
            prediction = 0
        else:
            prediction = 1 - 1 / den * np.sum(wrig_scores.sims * wrig_scores.pred)
        return prediction

    '''
    def predict(self):

        qrun = self.run.loc[(self.run["rank"] <= self.k) & (self.run.qid.isin(self.queries.qid))]

        vrun = self.vars_run
        vrun = vrun.loc[(vrun["rank"] <= self.k) & (vrun["vid"] != "0") & (vrun.tid.isin(self.queries.qid))]

        out = qrun.groupby("qid").apply(lambda x: self._compute_prediction(x, vrun.loc[vrun.tid == x.iloc[0].qid]))

        return pd.DataFrame(out).reset_index().rename({0: "prediction"}, axis=1)



    def _compute_prediction(self, qrun, vruns):
        # predictions using a local predictor (default variance of the scores)
        local_pred = np.array(vruns.groupby("vid").apply(self._compute_internal_predictor))

        # similarity between runs
        sims = np.array(vruns.groupby("vid").apply(lambda x: self.similiarity(qrun, x, simtype="RBO")))

        den = self._compute_internal_predictor(qrun) * np.sum(sims)

        if den == 0:
            prediction = 0
        else:
            prediction = 1 - 1 / den * np.dot(local_pred, sims)

        return prediction
    '''
