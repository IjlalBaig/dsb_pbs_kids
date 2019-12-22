import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import timedelta
import src.utils as utils
import itertools
import torch
from PIL import Image

import src.enums as ENUMS
ASSESSMENT_TITLES = [title for title in ENUMS.TITLES if "Assessment" in title]


class PBSKidsDataset(Dataset):
    def __init__(self, fpath, **kwargs):
        super().__init__()
        self._fpath = None
        self._kwargs = kwargs
        self._raw_df = None
        self._data = []
        self._data_mean = []
        self._data_std = []

        # init data
        self.fpath = fpath

    @property
    def fpath(self):
        return self._fpath

    @fpath.setter
    def fpath(self, path=""):
        self._raw_df = pd.read_csv(path)
        self._fpath = path

        cache = dict()
        cache_fpath = self._kwargs.get("cache_fpath", "./data/cache.json")
        if self._kwargs.get("use_cached", True):
            cache = utils.read_json(cache_fpath)
            if not cache:
                cache = dict()
            self._data = cache.get("processed_data", [])
        if not self._data:
            self.process_raw_df()
            cache["processed_data"] = self._data
            utils.write_json(cache_fpath, cache)
        self.find_data_standardization(use_cached=self._kwargs.get("use_cached", True),
                                       cache_fpath=self._kwargs.get("cache_fpath", "./data/cache.json"))

    @property
    def data(self):
        return self._data

    @staticmethod
    def process_sampledata(frame):
        title_data = frame["title"].value_counts().to_frame().transpose()
        type_data = frame["type"].value_counts().to_frame().transpose()
        world_data = frame["world"].value_counts().to_frame().transpose()
        event_data = frame["event_code"].value_counts().to_frame().transpose()
        gametime_data = frame["game_time"].loc[frame["game_session"].drop_duplicates(keep="last").index].sum()

        gameplay_data = pd.DataFrame(0, index=np.zeros(1),
                                     columns=ENUMS.TITLES + ENUMS.TYPES + ENUMS.WORLDS +
                                             ENUMS.EVENT_CODES + ["game_time"])
        gameplay_data[title_data.columns] = title_data.values
        gameplay_data[type_data.columns] = type_data.values
        gameplay_data[world_data.columns] = world_data.values
        gameplay_data[event_data.columns] = event_data.values
        gameplay_data["game_time"] = gametime_data
        return gameplay_data

    def process_raw_df(self):
        df = self._raw_df.sort_values(by=["installation_id", "timestamp", "game_session"])
        install_ids = df.installation_id.unique()

        for install_id in install_ids:
            user_data = df[df["installation_id"] == install_id]
            if 'Assessment' in user_data["type"].unique():
                assessment_data = user_data[user_data["type"] == "Assessment"]
                accuracies = self.get_accuracy(assessment_data)

                start_idx = 0
                end_idxs = assessment_data[assessment_data["event_data"].str.contains("correct")]\
                    .drop_duplicates("game_session", keep="last").index
                for i, end_idx in enumerate(end_idxs):

                    # process frames
                    new_frame = user_data.loc[start_idx:end_idx]
                    processed_new_frame = self.process_sampledata(new_frame)

                    cum_frame = user_data.loc[:end_idx]
                    processed_cum_frame = self.process_sampledata(cum_frame)
                    sample = np.concatenate([processed_new_frame.values,
                                             processed_cum_frame.values / (i + 1)]).reshape(-1).tolist()

                    out_titles = accuracies.iloc[:i+1].groupby("title").nth(-1).reset_index()["title"].values.tolist()
                    out_groups = accuracies.iloc[:i+1].groupby("title").nth(-1).reset_index()["accuracy_group"].values.tolist()
                    out_idxs = [ASSESSMENT_TITLES.index(title) for title in out_titles]
                    out = np.zeros(len(ASSESSMENT_TITLES)) - 1
                    for j, idx in enumerate(out_idxs):
                        out[idx] = out_groups[j]
                    self.data.append({"install_id": install_id,
                                      "gameplay_data": sample,
                                      "accuracy_data": out.tolist()})
                    start_idx = end_idx

    @staticmethod
    def get_accuracy(df, filter_events=None):
        # pd.Series((assessment_title.timestamp.apply(pd.Timestamp)
        #            - assessment_title.timestamp.shift(fill_value=assessment_title.timestamp.iloc[0]).apply(
        #             pd.Timestamp)).apply(pd.to_timedelta)).apply(pd.Timedelta.total_seconds)
        # Also break at start of new session
        # assessment_title[(assessment_title.index.to_series() - assessment_title.index.to_series().shift(
            # fill_value=assessment_title.index.to_series().iloc[0])) > 1]

        assessments = df[(df["type"]) == "Assessment"]
        if filter_events is not None:
            assessments = assessments[(assessments["event_data"].str.contains("correct")) &
                                      (assessments["event_code"].isin(filter_events))]
        else:
            assessments = assessments[(assessments["event_data"].str.contains("correct"))]

        assessments["correct"] = False
        assessments.loc[(assessments["event_data"].str.contains("true")), "correct"] = True
        assessment_groups = assessments.groupby("game_session")
        session_ids = assessment_groups.nth(0).reset_index()["game_session"]
        assessment_titles = assessment_groups.nth(0).reset_index()["title"]
        accuracy_group = np.zeros_like(session_ids, dtype=int)

        for i, session_id in enumerate(session_ids):
            trues_count = assessments[(assessments["game_session"] == session_id)]["correct"].sum()
            false_count = (~assessments[(assessments["game_session"] == session_id)]["correct"]).sum()
            accuracy = trues_count / (trues_count + false_count)

            if accuracy == 0:
                accuracy_group[i] = 0
            elif 0 < accuracy < 0.5:
                accuracy_group[i] = 1
            elif 0.5 <= accuracy < 1:
                accuracy_group[i] = 2
            elif accuracy == 1:
                accuracy_group[i] = 3
        return pd.DataFrame({"game_session": session_ids, "title": assessment_titles, "accuracy_group": accuracy_group})

    def find_data_standardization(self, use_cached, cache_fpath):
        self._data_mean = np.zeros_like(self._data[0].get("gameplay_data")).tolist()
        self._data_std = np.ones_like(self._data[0].get("gameplay_data")).tolist()
        return
        cache = None
        if use_cached:
            # load from cache
            cache = utils.read_json(cache_fpath)
            if isinstance(cache, dict) and all([key in cache.keys() for key in ["mean", "std"]]):
                self._data_mean = cache.get("mean")
                self._data_std = cache.get("std")
                return
        # compute and cache
        data_array = None
        for i, item in enumerate(self._data):
            if i == 0:
                data_array = item.get("gameplay_data")
            else:
                data_array = np.vstack([data_array, item.get("gameplay_data")])
        self._data_mean = np.mean(data_array, axis=0).tolist()
        self._data_std = np.std(data_array, axis=0).tolist()
        if cache is None:
            cache = dict()
        cache["mean"] = self._data_mean
        cache["std"] = self._data_std
        utils.write_json(cache_fpath, cache)

    def preprocess_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        install_id = item.get("install_id")
        gameplay_data = item.get("gameplay_data")
        accuracy_data = item.get("accuracy_data")

        # standardize
        mu_tensor = torch.tensor(self._data_mean, dtype=torch.float)
        std_tensor = torch.tensor(self._data_std, dtype=torch.float)
        x_tensor = torch.tensor(gameplay_data, dtype=torch.float).sub(mu_tensor).div(std_tensor)

        sample = {"install_id": install_id,
                  "gameplay_data": x_tensor,
                  "accuracy_data": torch.tensor(accuracy_data, dtype=torch.float)}
        return sample

