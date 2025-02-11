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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictor")
    parser.add_argument("-c", "--collection")
    parser.add_argument("-e", "--encoder")
    parser.add_argument("-w", "--workers", default=40, type=int)
    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    queries = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{args.collection}.query_reader"])(config)

    run = pd.read_csv(f"data/runs/{args.collection}_{args.encoder}.tsv", names=["query_id", "doc_id", "rank", "score"], usecols=[0, 2, 3, 4], sep="\t",
                      dtype={"query_id": str, "doc_id": str})

    common_params = {"predictor_name": args.predictor}

    if args.predictor in ["DenseQPP"]:
        encoder = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), args.encoder.capitalize())()
        queries["representation"] = list(encoder.encode_queries(queries.text.to_list()))
        corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"
        if args.encoder != "minilml12":
            docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv")
        else:
            docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv", embedding_size=384)
        indexWrapper = FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())
        common_params = common_params | {"indexWrapper": indexWrapper, "run": run}

    elif args.predictor in ["DCNQC", "DCWIG", "DCSMV", "PDQPP", "Hypervolume", "DPDQPP", "QPDQPP", "RPDQPP", "PDQPP_noden"]:
        encoder = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), args.encoder.capitalize())()
        queries["representation"] = list(encoder.encode_queries(queries.text.to_list()))
        corpora_memmapsdir = f"{config['DEFAULT']['indexes_dir']}/memmap/{config['Collections'][f'{args.collection}.corpus']}/{args.encoder}"

        if args.encoder != "minilml12":
            docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv")
        else:
            docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.encoder}.dat", f"{corpora_memmapsdir}/{args.encoder}_map.csv", embedding_size=384)

        common_params = common_params | {"run": run, "docs_encoder": docs_encoder}


    elif args.predictor in ["Nsigma", "NQCscores", "SMVscores", "Std"]:
        common_params = common_params | {"run": run}

    elif args.predictor in ["WIG", "Clarity", "UEFNQC", "UEFSMV", "UEFWIG", "UEFClarity", "RSD"]:
        index_path = f"{config['DEFAULT']['indexes_dir']}/pyterrier/{config['Collections'][f'{args.collection}.corpus']}/data.properties"
        index = pt.IndexFactory.of(index_path)
        stoplist = set([l.strip() for l in open("data/pyterrier_util_data/stopword-list.txt", "r").readlines()])
        common_params = common_params | {"index": index, "stoplist": stoplist, "run": run}

    elif args.predictor in ["WRIG"]:
        variants_run = pd.read_csv(f"data/w2v_runs/{args.collection}_{args.encoder}_w2v.tsv",
                                   names=["query_id", "doc_id", "rank", "score"], usecols=[0, 2, 3, 4], sep="\t", dtype={"query_id": str, "doc_id": str})

        common_params = common_params | {"variants_run": variants_run, "run": run}


    else:
        raise ValueError("unrecognized predictor")

    params = [(common_params | hyper_params) for hyper_params in local_utils.parse_qpp_params(config, args.predictor)]
    args.workers = min(args.workers, len(params))


    def _parallel_instantiate(params_set):
        predictions = []
        for hyper_params in params_set:
            predictor = getattr(importlib.import_module("qpp"), hyper_params["predictor_name"])(**hyper_params)
            predictions.append(predictor.predict(queries))
        return pd.concat(predictions)


    flprint(f"working on {args.collection}, {args.encoder} using {args.predictor}")
    flprint("started predictions ... ", end="")
    start_time = time()
    with Pool() as pool:
        predictions = pd.concat(pool.map(_parallel_instantiate, np.array_split(params, args.workers)))

    predictions.to_csv(f"data/predictions/{args.collection}_{args.encoder}_{args.predictor}.csv", index=False)
    print(f"done in {time() - start_time:.2f}")
