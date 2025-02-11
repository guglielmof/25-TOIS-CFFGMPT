import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import faiss
import sys

sys.path += ["code", "..", "/mnt"]

import argparse
import configparser
import importlib
import pandas as pd
import local_utils
from local_utils import flprint
from MYRETRIEVE.code.indexes.FaissIndex import FaissIndex
from memmap_interface import MemmapCorpusEncoding
from multiprocessing.dummy import Pool
import numpy as np
from time import time
import pyterrier as pt

from MYRETRIEVE.code.evaluating.evaluate import compute_measure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection")
    parser.add_argument("-e", "--encoder")
    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    queries = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.query_reader"])(config)
    qrels = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.qrels_reader"])(config)

    encoder = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), args.encoder.capitalize())()
    queries["representation"] = list(encoder.encode_queries(queries.text.to_list()))
    corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv",
                                        embedding_size=encoder.get_embedding_dim())
    indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())

    qembs = np.array(queries.representation.to_list())
    r2q = {e: q for e, q in enumerate(queries.query_id.to_list())}

    ip, idx = indexWrapper.index.search(qembs, 1000)

    run = pd.DataFrame({"query_id": np.arange(qembs.shape[0]), "doc_id": list(idx), "score": list(ip)}).explode(["doc_id", "score"])
    run.query_id = run.query_id.map(r2q)
    run.doc_id = run.doc_id.map(lambda x: indexWrapper.mapper[x])
    run.score = run.score.astype(float)

    run["rank"] = run.groupby("query_id")["score"].rank(ascending=False, method="first").astype(int)

    run["Q0"] = "Q0"
    run["rid"] = args.encoder

    run = run[["query_id", "Q0", "doc_id", "rank", "score", "rid"]]

    run.to_csv(f"data/runs/{args.collection}_{args.encoder}.tsv", header=False, index=False, sep="\t")
    measure = compute_measure(run, qrels, ["nDCG@10"]).drop("measure", axis=1)
    print(f"avg perf: {measure.value.mean()}")