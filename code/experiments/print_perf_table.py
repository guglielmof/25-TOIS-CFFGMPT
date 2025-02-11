import sys
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path += [".", "code", "..", "/mnt"]
import pandas as pd
import numpy as np
import qpp.utils
from MYRETRIEVE.code.evaluating.evaluate import compute_measure
import argparse
import configparser
import importlib
from glob import glob
from multiprocessing.dummy import Pool
from local_utils.bioinfokit.analys import stat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collections", nargs="+", default=["trec-dl-2019", "trec-dl-2020", "trec-dl-hard", "trec-robust-2004"])
    parser.add_argument("-e", "--irmodels", nargs="+", default=["contriever", "tasb", "ance", "minilml12"])
    parser.add_argument("-m", "--measure", default="nDCG@10")
    parser.add_argument("-p", "--qpp_perf_measure", nargs="+", default=["pearson", "kendall", "smare"])
    parser.add_argument("--config_path", default="properties/properties.ini")
    args = parser.parse_args()

    print_order = ["Std", "Nsigma", "Clarity", "NQCscores", "SMVscores", "RSD", "WIG", "UEFClarity", "UEFNQC", "UEFSMV", "UEFWIG", "WRIG", "DCNQC", "DCSMV", "DCWIG", "Hypervolume", "DenseQPP", 'bertqppbi', 'bertqppce', "PDQPP"]

    config = configparser.ConfigParser()
    config.read(args.config_path)

    real = []
    pred = []
    for collection in args.collections:
        qrels = getattr(importlib.import_module("local_utils.data_readers"), config["Collections"][f"{collection}.qrels_reader"])(config)
        for irmodel in args.irmodels:
            if os.path.exists(f"data/runs/{collection}_{irmodel}.tsv"):
                run = pd.read_csv(f"data/runs/{collection}_{irmodel}.tsv", names=["query_id", "doc_id", "score"], usecols=[0, 2, 4], sep="\t",
                                  dtype={"query_id": str, "doc_id": str})
                run = run.loc[run["query_id"].isin(qrels.query_id.unique())]

                real.append(compute_measure(run, qrels, [args.measure]).drop("measure", axis=1))
                real[-1]["collection"] = collection
                real[-1]["irmodel"] = irmodel

                pred.append(pd.concat([pd.read_csv(f, dtype={"query_id": str}) for f in glob(f"data/predictions/{collection}_{irmodel}_*.csv")]))
                pred[-1][["collection", "irmodel"]] = collection, irmodel

    real = pd.concat(real)
    print(real.groupby(["collection", "irmodel"])["value"].mean().reset_index())
    pred = pd.concat(pred)
    pred = pred.fillna(0)
    pred = pred.loc[pred.predictor.isin(print_order)]
    full_ds = pred.merge(real)

    qpp_measures = []
    if "pearson" in args.qpp_perf_measure:
        qpp_measures.append(lambda x: qpp.utils.twofold_validation(x))
    if "kendall" in args.qpp_perf_measure:
        qpp_measures.append(lambda x: qpp.utils.twofold_validation(x, perf_measure="kendall"))
    if "smare" in args.qpp_perf_measure:
        qpp_measures.append(lambda x: qpp.utils.twofold_validation(x, perf_measure="smare"))


    def anova(ds):
        means = ds.groupby("predictor").mean().reset_index()
        print(means)
        bp = means.loc[means["value"].idxmax(), "predictor"]

        res = stat()
        res.tukey_hsd(df=ds, res_var='value', xfac_var='predictor', anova_model='value ~ C(predictor)')
        tukey_res = res.tukey_summary
        tukey_res = tukey_res.loc[(tukey_res["p-value"] >= 0.05) & ((tukey_res['group1'] == bp) | (tukey_res['group2'] == bp))]
        eq_systems = set(tukey_res.group2.values).union(tukey_res.group1.values).union({bp})
        return list(eq_systems)


    parallel_input = [(full_ds.loc[full_ds.collection == c].copy(), f) for c in full_ds.collection.unique() for f in qpp_measures]

    with Pool(processes=len(parallel_input)) as pool:
        def _parallel_instantiate(data):
            ds, f = data
            sys.stdout.flush()
            validated_perf = ds.groupby(["predictor", "collection", "irmodel"]).apply(f).reset_index().drop("level_3", axis=1)
            return validated_perf


        qpp_performance = pd.concat(pool.map(_parallel_instantiate, parallel_input))
    qpp_performance.to_csv("data/performance/full_performance.csv", index=False)

    full_perf = qpp_performance.groupby(["predictor", "collection", "irmodel", "measure"])["value"].mean().reset_index()
    full_perf["value"] = full_perf.value.apply(lambda x: round(x, 3))
    top_group = qpp_performance.groupby(["collection", "irmodel", "measure"]).apply(anova).reset_index().rename({0: "predictor"}, axis=1).explode("predictor")
    top_group["top_group"] = True
    best_pred = full_perf.groupby(["collection", "irmodel", "measure"])["value"].max().reset_index()
    best_pred["best_pred"] = True

    full_perf = full_perf.merge(top_group, how="left").merge(best_pred, how="left").fillna(False)


    def stringify(row):
        out_string = f"{row['value']:.3f}"
        if row["best_pred"]:
            out_string = "\\bf{" + out_string + "}"
        if row["top_group"]:
            out_string += "\\tg"
        return out_string


    full_perf["string_format"] = full_perf.apply(stringify, axis=1)
    full_perf.drop(["value", "top_group"], axis=1, inplace=True)

    full_perf["predictor"] = pd.Categorical(full_perf["predictor"], categories=print_order, ordered=True)

    def printing(ds):
        s = ds.pivot_table(index="predictor", columns=["collection", "measure"], values="string_format", aggfunc=lambda x: x).to_string()

        print(f"-------------------------- {ds.irmodel.values[0]} --------------------------  ")
        print(s)
        with open(f"data/performance/{ds.irmodel.values[0]}.txt", "w") as f:
            f.write(s)
            pass

    full_perf.groupby("irmodel").apply(printing)

    print(top_group.groupby(["irmodel", "predictor"])["top_group"].count().reset_index().pivot_table(index="predictor", columns="irmodel",
                                                                                                     values="top_group").to_string())
    print(
        full_perf[full_perf["best_pred"]].groupby(["irmodel", "predictor"])["best_pred"].count().reset_index().pivot_table(index="predictor", columns="irmodel",
                                                                                                                           values="best_pred").to_string())
