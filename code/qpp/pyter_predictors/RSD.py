from .AbstracyPTPredictor import AbstractPTPredictor
import numpy as np
import pandas as pd
import sys
from .WIG import WIG


class RSD(AbstractPTPredictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams_identifiers = ["k"]

    def _compute_wig(self, query_processed, run):
        top_k_docs = run.doc_id.to_list()
        internal_ids = set([self.index_wrapper.getDocumentInternalId(d) for d in top_k_docs])

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

        pred_score = pred_score / (len(top_k_docs) * np.sqrt(len(query_processed)))

        return pred_score

    def _local_predict(self, query: pd.Series) -> float:

        local_run = self.run[self.run.query_id == query.query_id]
        query_processed = self.index_wrapper.analyze(query.text)
        wigs = []
        stds = []
        for l in range(2, self.k):
            short_run = local_run.loc[local_run["rank"]<=l]
            # original wig is pred_score/sqrt(q) while they did explicitly state they do not devide by sqrt(q)
            wigs.append(self._compute_wig(query_processed, short_run) * np.sqrt(len(query_processed)))
            stds.append(np.std(short_run.score))

        wigs, stds = np.array(wigs), np.array(stds)
        pred_score = np.sqrt(np.sum(wigs * stds))
        return pred_score
