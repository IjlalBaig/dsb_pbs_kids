import pandas as pd
import numpy as np
import json
import os


def delta_time(t0, t1):
    dt = pd.Timestamp(t1) - pd.Timestamp(t0)
    if dt.days >= 0:
        return pd.Timedelta(dt).seconds
    else:
        return pd.Timedelta(dt).seconds - 24*60*60


def time_of_day(ts):
    pd_timestamp = pd.Timestamp(ts)
    return pd_timestamp.hour * 60 * 60 + pd_timestamp.minute * 60 + pd_timestamp.second


def day_of_week(ts):
    return pd.Timestamp(ts).dayofweek


def one_hot_encode(val, max_val):
    encode = np.zeros(max_val)
    encode[val] = 1.0
    return encode


def read_json(fpath):
    data = None
    try:
        with open(fpath, "r") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        pass
    return data


def write_json(fpath, data):
    dpath = os.path.dirname(fpath)
    os.makedirs(name=dpath, exist_ok=True)
    with open(fpath, "w") as file:
        json.dump(data, file)
