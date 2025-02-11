import numpy as np
import itertools
import sys

def flprint(string, *args, **kwargs):
    print(string, *args, **kwargs)
    sys.stdout.flush()


def parse_qpp_params(config, qpp_model):
    types = {"int": int, "float": float}

    params = []
    if f"{qpp_model}.params" in config["QPP"]:
        params_names = config["QPP"][f"{qpp_model}.params"].split(",")
        params_values = {p: config["QPP"][f"{qpp_model}.{p}"].split(",") for p in params_names}
        params_values = {p: list(map(types[config["QPP"][f"{qpp_model}.{p}.type"]], pset)) for p, pset in params_values.items()}
        params = [dict(zip(params_values, x)) for x in itertools.product(*params_values.values())]

    return params
