import pandas as pd
import numpy as np


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
