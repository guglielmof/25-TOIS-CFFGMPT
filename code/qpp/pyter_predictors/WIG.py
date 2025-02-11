from .AbstracyPTPredictor import AbstractPTPredictor
import numpy as np
import pandas as pd
import sys


class WIG(AbstractPTPredictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams_identifiers = ["k"]

    def _local_predict(self, query: pd.Series) -> float:
        top_k_docs = list(self.run.loc[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k), "doc_id"])
        internal_ids = set([self.index_wrapper.getDocumentInternalId(d) for d in top_k_docs])

        query_processed = self.index_wrapper.analyze(query.text)

        pred_score = 0

        for term in query_processed:
            if self.index_wrapper.inIndex(term):
                # take the collection frequency of the term
                coll_frequency = self.index_wrapper.getNormalizedCF(term)

                nfound = 0
                for posting in self.index_wrapper.getTermPostings(term):
                    # for each doc id in the posting list among those retrieved
                    if posting.getId() in internal_ids:
                        # take the term frequency
                        term_frequency = posting.getFrequency() / self.index_wrapper.getDocumentLength(posting.getId())
                        pred_score += np.log(term_frequency / coll_frequency)
                        nfound += 1
                        if nfound == len(internal_ids):
                            break

        pred_score = pred_score / (self.k * np.sqrt(len(query_processed)))

        return pred_score
