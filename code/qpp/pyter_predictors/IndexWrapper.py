from typing import Optional, List

import pandas as pd
from jnius import autoclass
import numpy as np


class IndexWrapper:

    def __init__(self, index, **kwargs):
        self.index = index
        self.di = index.getDirectIndex()
        self.doi = index.getDocumentIndex()
        self.lex = index.getLexicon()
        self.meta = index.getMetaIndex()
        self.inv = index.getInvertedIndex()

        if "stemmer" in kwargs:
            self.stemmer = autoclass("org.terrier.terms.PorterStemmer")()  # kwargs["stemmer"]

        if "stopwords" in kwargs:
            self.stopwords = kwargs["stopwords"]
        else:
            self.stopwords = set([])

        if "tokeniser" in kwargs:
            self.tokeniser = kwargs["tokeniser"]
        else:
            self.tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

    def getNormalizedCF(self, term: str) -> float:
        """
        :param term: term for which we are interested in getting the normalized collection frequency
        :return: normalized Collection frequency (Total Term Frequency/Number of Tokens)
        """
        return self.lex[term].getFrequency() / self.getNumberOfTokens()

    def getTTF(self, term: str) -> int:
        """
        :param term: (processed) term for which we want to know the Total Term Frequency (TTF) (i.e., the total number of occurrences of the term)
        :return: Total Term Frequency of a the term
        """
        return self.lex[term].getFrequency()

    def getNumberOfTokens(self) -> int:
        """
        :return: total number of tokens in the index
        """

        return self.index.getCollectionStatistics().getNumberOfTokens()

    def getDocumentInternalId(self, docno: object, internal_meta: str = "docno") -> int:
        """
        :param docno: the alphanumeric id of the document
        :param internal_meta: the name of the internal attribute
        :return: the internal document id for the document externally identified by docno
        """
        return self.meta.getDocument(internal_meta, docno)

    def getTerm(self, term_id: int) -> str:
        """
        :param term_id: internal id of a term
        :return: human-readable term corresponding to the internal term_id
        """
        return self.lex.getLexiconEntry(term_id).getKey()

    def getDocumentPostings(self, docno: object, internal_meta: str = "docno"):
        doc_entry = self.doi.getDocumentEntry(self.getDocumentInternalId(docno, internal_meta))
        if doc_entry is None:
            return None
        return self.di.getPostings(doc_entry)

    def getTermPostings(self, term: str):
        return self.inv.getPostings(self.lex.getLexiconEntry(term))

    def getNumberOfDocuments(self):
        return self.index.getCollectionStatistics().getNumberOfDocuments()

    def getDocumentFrequency(self, term: str) -> int:
        return self.lex[term].getDocumentFrequency()

    def construct_relevance_model(self, doc_ids: List[object],
                                  scales: List[float],
                                  top_m: Optional[int],
                                  max_freq: Optional[float]) -> pd.DataFrame:

        rm_df = []
        for e, d in enumerate(doc_ids):
            # get the posting lists for the documents in doc_ids
            postings = self.getDocumentPostings(d)

            if postings is not None:
                # construct the relevance model using each posting lists (term_id -> frequency)
                local_rm = pd.DataFrame([[d, p.getId(), p.getFrequency()] for p in postings], columns=["doc_id", "term_id", "freq"])

                # normalize the relevance model and scale it
                local_rm["freq"] = self.normalize_rm(local_rm)["freq"] * scales[e]

                rm_df.append(local_rm)

        rm_df = pd.concat(rm_df)
        # filter too frequent terms (i.e., terms that appear more in more than "max_freq" documents)
        unique_terms = pd.DataFrame({'term_id': rm_df.term_id.unique()})
        unique_terms["term"] = unique_terms['term_id'].apply(self.getTerm)
        tot_ndocs = self.getNumberOfDocuments()
        unique_terms["freq"] = unique_terms["term"].apply(lambda x: self.getDocumentFrequency(x) / tot_ndocs)

        unique_terms = unique_terms.loc[unique_terms["freq"] <= max_freq]
        rm_df = rm_df.merge(unique_terms[["term_id", "term"]])[["term", "freq"]]

        # sum over the terms
        rm_df = rm_df.groupby("term").sum().reset_index()

        # keep only the first top m terms and discard the rest
        if top_m is not None:
            rm_df = rm_df.nlargest(top_m, "freq")

        # normalize the final relevance model
        rm_df = self.normalize_rm(rm_df)

        return rm_df

    def score_with_rm(self, docno, rm, mu=2000):
        rmterms = set(rm.term)
        doclen = 0

        rmweights = []
        termsfreq = []
        termscolf = []
        for posting in self.di.getPostings(self.doi.getDocumentEntry(self.getDocumentInternalId(docno))):
            termid = posting.getId()
            lee = self.lex.getLexiconEntry(termid)
            term = lee.getKey()
            doclen += 1

            if term in rmterms:
                rmweights.append(rm.loc[rm["term"] == term, 'freq'].values[0])
                termsfreq.append(posting.getFrequency())
                termscolf.append(self.getNormalizedCF(term))

        if len(rmweights) == 0:
            return 0

        termsfreq = np.array(termsfreq)
        rmweights = np.array(rmweights)
        termscolf = np.array(termscolf)

        left = termsfreq / (mu + doclen)
        right = (1 - doclen / (mu + doclen)) * termscolf
        qlscore = np.log(1 / (left + right))
        rmscore = np.sum(rmweights * qlscore)

        return rmscore

    def normalize_rm(self, rm: pd.DataFrame) -> pd.DataFrame:
        """
        :param rm: dataframe containing a relevance model ["term", "frequency"]
        :return: normalized relevance model such that the frequency column adds up to one
        """
        rm.freq = rm.freq / rm.freq.sum()
        return rm

    def analyze(self, query_text):
        tokens = [self.stemmer.stem(t) for t in self.tokeniser.getTokens(query_text) if t not in self.stopwords]
        return tokens

    def inIndex(self, term):
        """
        :param term: term to be searched on the index
        :return: boolean describing if the term is present in the index
        """
        return term in self.lex

    def getDocumentLength(self, docid):
        """
        :param docid: internal docid
        :return:
        """
        return self.doi.getDocumentLength(docid)
        # return np.sum([1 for _ in self.di.getPostings(self.doi.getDocumentEntry(docid))])
