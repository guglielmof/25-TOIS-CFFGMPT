import configparser

import ir_datasets
import pandas as pd


def read_uqv100_queries(*args):
    if type(args[0]) == str:
        path = args[0]
    elif type(args[0]) == configparser.ConfigParser:
        path = args[0]["Collections"]["uqv100.queries.path"]
    else:
        raise ValueError("unrecognized input type")
    queries = pd.read_csv(path, header=None, names=["full"])
    queries[["query_id", "text"]] = queries.full.str.split(" ", n=1, expand=True)
    queries.drop("full", axis=1, inplace=True)

    return queries


def read_uqv100_qrels(*args):
    if type(args[0]) == str:
        path = args[0]
    elif type(args[0]) == configparser.ConfigParser:
        path = args[0]["Collections"]["uqv100.qrels.path"]
    else:
        raise ValueError("unrecognized input type")

    return pd.read_csv(path)


def read_trecdl2019_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["trec-dl-2019.datasetid"])


def read_trecdl2019_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-dl-2019.datasetid"])


def read_trecdl2020_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["trec-dl-2020.datasetid"])


def read_trecdl2020_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-dl-2020.datasetid"])


def read_trecdlhard_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["trec-dl-hard.datasetid"])


def read_trecdlhard_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-dl-hard.datasetid"])


def read_vaswani_queries(*args):
    return _read_irdataset_queries(args[0]["Collections"]["vaswani.datasetid"])


def read_vaswani_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["vaswani.datasetid"])


def read_treccastv12020_queries(*args):
    postprocessing = lambda x: x.rename({"manual_rewritten_utterance": "text"}, axis=1)[["query_id", "text"]]
    return _read_irdataset_queries(args[0]["Collections"]["trec-cast-v1-2020.datasetid"], postprocessing=postprocessing)


def read_treccastv12020_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-cast-v1-2020.datasetid"])


def _read_irdataset_queries(dataset_id, postprocessing=None):
    if postprocessing is None:
        postprocessing = lambda x: x
    # print(postprocessing(pd.DataFrame(ir_datasets.load(dataset_id).queries_iter())))
    return postprocessing(pd.DataFrame(ir_datasets.load(dataset_id).queries_iter()))


def _read_irdataset_qrels(dataset_id, postprocessing=None):
    if postprocessing is None:
        postprocessing = lambda x: x

    return postprocessing(pd.DataFrame(ir_datasets.load(dataset_id).qrels_iter()))


def read_trecrobust2004_queries(*args):
    postprocessing = lambda x: x.rename({"title": "text"}, axis=1)[["query_id", "text"]]
    return _read_irdataset_queries(args[0]["Collections"]["trec-robust-2004.datasetid"], postprocessing=postprocessing)


def read_trecrobust2004_qrels(*args):
    return _read_irdataset_qrels(args[0]["Collections"]["trec-robust-2004.datasetid"])
