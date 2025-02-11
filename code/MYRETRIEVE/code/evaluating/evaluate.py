from ir_measures import *
import pandas as pd
import ir_measures


def compute_measure(run, qrels, measures, only_available=False):
    """

    :param run: a pd.DataFrame with (at least) three columns: query_id, doc_id, score
    :param qrels: a pd.DataFrame with (at least) three columns: query_id, doc_id, relevancee
    :param measures: a list of either ir_measures measures or strings parsable according to ir_measures
    :param only_available: if True, consider only queries available in the runs

    :return:  a pd.DataFrame with three columns query_id, measure, value.

    This function takes in input a run an the qrels under the form of pd.DataFrame and a list of measures and computes the performance
    query wise. with the only_available paramer, it is possible to enforce that considered queries are only those available in the run.

    """
    if only_available:
        qrels = qrels.loc[qrels.query_id.isin(run.query_id)]

    # parse measures where they are strings
    measures = [ir_measures.parse_measure(m) if type(m) == str else m for m in measures]
    # compute the performance via iter_calc
    performance = pd.DataFrame(ir_measures.iter_calc(measures, qrels, run))
    # cast the measures in the performance dataset into strings
    performance['measure'] = performance['measure'].astype(str)

    return performance
