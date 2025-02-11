import sys
import os

sys.path += [".", "code"]
from pathlib import Path
import argparse
import numpy as np
import importlib
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import re


def preprocess_medline(ds):
    ds["text"] = ds.title + " " + ds.abstract
    return ds

def preprocess_robust(ds):
    ds["text"] = ds.title + " " + ds.abstract
    return ds

preprocessing = {"medline_2004": preprocess_medline, "tipster": preprocess_robust}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection")
    parser.add_argument("--corpus_name", default=None)
    parser.add_argument("-e", "--encoder")
    parser.add_argument("-b", "--batch_size", default=5000, type=int)
    parser.add_argument("--collections_path", default="collections")

    args = parser.parse_args()

    if args.corpus_name is None:
        args.corpus_name = args.collection

    print(f"indexing {args.collection} ({args.corpus_name}) with {args.encoder}", flush=True)
    print(f"batch size: {args.batch_size}", flush=True)
    print(f"the results will be stored in {args.collections_path}/INDEXES/memmap/{args.corpus_name}/{args.encoder}", flush=True)
    sys.stdout.flush()
    os.environ["IR_DATASETS_HOME"] = f"{args.collections_path}/ir_datasets"

    model = getattr(importlib.import_module("irmodels.dense"), args.encoder.capitalize())()

    import ir_datasets

    ir_datasets.util.home_path()

    dataset = ir_datasets.load(args.collection)

    n_docs = dataset.metadata()["docs"]["count"]
    print(f"The collection has been loaded correctly and {n_docs} documents will be indexed", flush=True)

    print(f"initializing memmap ... ", end="", flush=True)

    memmap_path = f"{args.collections_path}/INDEXES/memmap/{args.corpus_name}/{args.encoder}"
    Path(memmap_path).mkdir(parents=True, exist_ok=True)
    fp = np.memmap(f"{memmap_path}/{args.encoder}.dat", dtype='float32', mode='w+', shape=(n_docs, model.embeddings_dim))

    print(f"memmap initialized", flush=True)

    n_batches = int(np.ceil(n_docs / args.batch_size))
    offset = []
    start_idx = 0
    di = dataset.docs_iter()
    pool = model.start_multi_process_pool()

    for b in tqdm(range(n_batches)):
        #print(f"{b + 1}/{n_batches}", flush=True)
        len_batch = min(args.batch_size, n_docs - b * args.batch_size)
        batch = pd.DataFrame([di.__next__() for i in range(len_batch)])
        if args.corpus_name in preprocessing:
            batch = preprocessing[args.corpus_name](batch)

        representations = model.get_model().encode_multi_process(batch.text.to_list(), pool)
        fp[start_idx: start_idx + len_batch] = representations
        start_idx += len_batch
        offset += batch.doc_id.to_list()

    offset = pd.DataFrame({"doc_id": offset, "offset": np.arange(n_docs)})
    offset.to_csv(f"{memmap_path}/{args.encoder}_map.csv", index=False)