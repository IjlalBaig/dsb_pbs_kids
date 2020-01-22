import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import src.utils as utils
import torch
from tqdm import tqdm
import timeit
from numba import jitclass, jit

import src.enums as ENUMS
ASSESSMENT_TITLES = [title for title in ENUMS.TITLES if "Assessment" in title]


class PBSKidsDataset(Dataset):
    def __init__(self, fpath, **kwargs):
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
        # self._raw_df = None
        self._fpath = path
        self.preprocess_data(cache_dpath=self._kwargs.get("cache_dpath", "./data/preprocessed"),
                             refresh=self._kwargs.get("cache_refresh", False),
                             mode=self._kwargs.get("mode", "train"))

    @property
    def data(self):
        return self._data

    @staticmethod
    def process_frame_slice(slice_):
        title_data = slice_["title"].value_counts().to_frame().transpose()
        type_data = slice_["type"].value_counts().to_frame().transpose()
        world_data = slice_["world"].value_counts().to_frame().transpose()
        event_data = slice_["event_code"].value_counts().to_frame().transpose()
        gametime_data = slice_["game_time"].loc[slice_["game_session"].drop_duplicates(keep="last").index].sum()
        gameplay_elements = ENUMS.TITLES + ENUMS.TYPES + ENUMS.WORLDS + ENUMS.EVENT_CODES + ["game_time"]
        gameplay_data = pd.DataFrame(0, index=np.zeros(1), columns=gameplay_elements)
        gameplay_data[title_data.columns] = title_data.values
        gameplay_data[type_data.columns] = type_data.values
        gameplay_data[world_data.columns] = world_data.values
        gameplay_data[event_data.columns] = event_data.values
        gameplay_data["game_time"] = gametime_data
        return gameplay_data

    def cache_data(self, cache_dpath):
        train_cache_fpath = os.path.join(cache_dpath, "train.json")
        std_cache_fpath = os.path.join(cache_dpath, "std.json")

        train_cache = {"processed_data": self._data}
        std_cache = {"data_mean": self._data_mean,
                     "data_std": self._data_std}

        utils.write_json(train_cache_fpath, train_cache)
        utils.write_json(std_cache_fpath, std_cache)

    def preprocess_data(self, cache_dpath, mode="train", refresh=False):
        train_cache_fpath = os.path.join(cache_dpath, "train.json")
        std_cache_fpath = os.path.join(cache_dpath, "std.json")
        if mode == "train":
            if os.path.exists(train_cache_fpath) and os.path.exists(std_cache_fpath) and not refresh:
                # load old cache
                train_cache = utils.read_json(train_cache_fpath)
                std_cache = utils.read_json(std_cache_fpath)
                self._data = train_cache.get("processed_data", [])
                self._data_mean = std_cache.get("data_mean", [])
                self._data_std = std_cache.get("data_std", [])
            else:
                # process and cache
                self.process_raw_df()
                self.find_data_standardization()
                self.cache_data(cache_dpath)

        elif mode == "test":
            # process only
            std_cache_fpath = os.path.join(cache_dpath, "std.json")
            std_cache = utils.read_json(std_cache_fpath)
            self._data_mean = std_cache.get("data_mean", [])
            self._data_std = std_cache.get("data_std", [])
            self.process_raw_df()

    def process_userdata(self, user_data):
        install_id = user_data["installation_id"][0]
        if 'Assessment' in user_data["type"].unique():
            assessment_data = user_data[user_data["type"] == "Assessment"]
            accuracies = self.get_accuracy(assessment_data)

            start_idx = 0
            end_idxs = assessment_data[assessment_data["event_data"].str.contains("correct")] \
                .drop_duplicates("game_session", keep="last").index
            for i, end_idx in enumerate(end_idxs):

                # process frames
                new_frame = user_data.loc[start_idx:end_idx]
                processed_new_frame = self.process_frame_slice(new_frame)

                cum_frame = user_data.loc[:end_idx]
                processed_cum_frame = self.process_frame_slice(cum_frame)
                sample = np.concatenate([processed_new_frame.values,
                                         processed_cum_frame.values / (i + 1)]).reshape(-1).tolist()

                out_titles = accuracies.iloc[:i + 1].groupby("title").nth(-1).reset_index()["title"].values.tolist()
                out_groups = accuracies.iloc[:i + 1].groupby("title").nth(-1).reset_index()[
                    "accuracy_group"].values.tolist()
                out_idxs = [ASSESSMENT_TITLES.index(title) for title in out_titles]
                out = np.zeros(len(ASSESSMENT_TITLES)) - 1
                for j, idx in enumerate(out_idxs):
                    out[idx] = out_groups[j]
                self.data.append({"install_id": install_id,
                                  "gameplay_data": sample,
                                  "accuracy_data": out.tolist()})
                start_idx = end_idx

    def process_raw_df(self):
        df = self._raw_df.sort_values(by=["installation_id", "timestamp", "game_session"])
        install_ids = df.installation_id.unique()
        install_ids = install_ids[:min(17000, len(install_ids))]

        for install_id in tqdm(install_ids, desc="pre-processing data"):
            user_data = df[df["installation_id"] == install_id]
            if 'Assessment' in user_data["type"].unique():
                assessment_data = user_data[user_data["type"] == "Assessment"]
                accuracies = self.get_accuracy(assessment_data)
                if len(accuracies) > 0:
                    start_idx = 0
                    end_idxs = assessment_data[(assessment_data["event_data"].str.contains("correct"))].drop_duplicates("game_session", keep="last").index
                    if accuracies.iloc[-1].accuracy_group is None:
                        end_idxs = end_idxs.append(pd.Index([assessment_data.iloc[-1].name]))
                        if len(accuracies) > 1:
                            start_idx = end_idxs[-2]

                    for i, end_idx in enumerate(end_idxs):
                        if accuracies.iloc[-1].accuracy_group is None:
                            if i < len(end_idxs) - 1:
                                continue
                        # process frames
                        new_frame = user_data.loc[start_idx:end_idx]
                        processed_new_frame = self.process_frame_slice(new_frame)

                        cum_frame = user_data.loc[:end_idx]
                        processed_cum_frame = self.process_frame_slice(cum_frame)
                        sample = np.concatenate([processed_new_frame.values,
                                                 processed_cum_frame.values / (i + 1)]).reshape(-1).tolist()

                        out_titles = accuracies.iloc[:i + 1].groupby("title").nth(-1).reset_index()["title"].values.tolist()
                        out_groups = accuracies.iloc[:i + 1].groupby("title").nth(-1).reset_index()["accuracy_group"].values.tolist()
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
        assessments = df[(df["type"]) == "Assessment"]
        probe_session_id = None
        if len(assessments) == 1 or \
                len(assessments) > 1 and assessments.iloc[-1].game_session != assessments.iloc[-2].game_session:
            probe_session_id = assessments.iloc[-1].game_session

        # Filter assessments
        if filter_events is not None:
            assessments = assessments[(assessments["event_data"].str.contains("correct")) &
                                      (assessments["event_code"].isin(filter_events)) |
                                      (assessments["game_session"] == probe_session_id)]
        else:
            assessments = assessments[(assessments["event_data"].str.contains("correct")) |
                                      (assessments["game_session"] == probe_session_id)]

        # Compute accuracy group
        assessments["correct"] = False
        assessments.loc[(assessments["event_data"].str.contains("true")), "correct"] = True
        session_ids = assessments.drop_duplicates("game_session")["game_session"]
        assessment_titles = assessments.drop_duplicates("game_session")["title"]
        accuracy_group = np.zeros_like(session_ids, dtype=object)

        for i, session_id in enumerate(session_ids):
            trues_count = assessments[(assessments["game_session"] == session_id)]["correct"].sum()
            false_count = (~assessments[(assessments["game_session"] == session_id)]["correct"]).sum()
            accuracy = trues_count / (trues_count + false_count)

            if session_id == probe_session_id:
                accuracy_group[i] = None
            elif accuracy == 0:
                accuracy_group[i] = 0
            elif 0 < accuracy < 0.5:
                accuracy_group[i] = 1
            elif 0.5 <= accuracy < 1:
                accuracy_group[i] = 2
            elif accuracy == 1:
                accuracy_group[i] = 3

        return pd.DataFrame({"game_session": session_ids,
                             "title": assessment_titles,
                             "accuracy_group": accuracy_group})

    def find_data_standardization(self):
        data_array = None
        for i, item in enumerate(self._data):
            if i == 0:
                data_array = item.get("gameplay_data")
            else:
                data_array = np.vstack([data_array, item.get("gameplay_data")])
        self._data_mean = np.mean(data_array, axis=0).tolist()
        self._data_std = np.std(data_array, axis=0).tolist()

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
        std_tensor[std_tensor == 0.0] = 1.0
        x_tensor = torch.tensor(gameplay_data, dtype=torch.float).sub(mu_tensor).div(std_tensor)

        sample = {"install_id": install_id,
                  "gameplay_data": x_tensor,
                  "accuracy_data": torch.tensor(accuracy_data, dtype=torch.float)}
        return sample

