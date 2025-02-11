import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as sts
import sys


def flprint(string, *args, **kwargs):
    print(string, *args, **kwargs)
    sys.stdout.flush()


def pearson(x):
    return x[["prediction", "value"]].corr().loc["prediction", "value"]


def kendall(x):
    return x[["prediction", "value"]].corr(method="kendall").loc["prediction", "value"]


def sMARE(x, y):
    rx = sts.rankdata(x)
    ry = sts.rankdata(y)

    sARE = 1 - np.abs(rx - ry) / len(rx)

    return np.mean(sARE)


def smare(x):
    return x[["prediction", "value"]].corr(method=sMARE).loc["prediction", "value"]


n2f = {"pearson": pearson, "kendall": kendall, "smare": smare}


def twofold_validation(ds, perf_measure="pearson", reps=30, seed=12345):
    rs = npr.RandomState(seed)
    queries = ds.query_id.unique()
    nqueries = len(queries)

    res = []
    for r in range(reps):
        selectorsA = rs.choice(np.arange(nqueries), replace=False, size=int(round(nqueries / 2)))

        selectorsB = [i for i in np.arange(nqueries) if i not in selectorsA]
        foldA = ds[ds.query_id.isin(queries[selectorsA])]
        foldB = ds[ds.query_id.isin(queries[selectorsB])]

        foldA_performance = foldA.groupby(["params"]).apply(n2f[perf_measure]).reset_index().rename({0: "perf"}, axis=1)
        foldB_performance = foldB.groupby(["params"]).apply(n2f[perf_measure]).reset_index().rename({0: "perf"}, axis=1)

        bestB = foldB_performance.loc[foldB_performance["perf"].idxmax(), "params"]
        bestA = foldA_performance.loc[foldA_performance["perf"].idxmax(), "params"]

        perfA = foldA_performance.loc[foldA_performance["params"] == bestB, "perf"].values[0]
        perfB = foldB_performance.loc[foldB_performance["params"] == bestA, "perf"].values[0]
        res.append((perfA + perfB) / 2)

    return pd.DataFrame({"fold": np.arange(reps), "measure": [perf_measure] * reps, "value": res})
