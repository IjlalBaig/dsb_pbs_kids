import os
import pandas as pd
import numpy as np
import src.utils as utils
import itertools
import torch
from PIL import Image

import src.enums as ENUMS


class PBSKidsDataset():
    def __init__(self, fpath="", **kwargs):
        super().__init__()
        self._fpath = None
        self._kwargs = kwargs
        self._raw_df = None
        self._data = None

        # init data
        self.fpath = fpath

    @property
    def fpath(self):
        return self._fpath

    @fpath.setter
    def fpath(self, path=""):
        self._raw_df = pd.read_csv(path)
        self._fpath = path
        self._data = self.get_processed_df()

    @property
    def data(self):
        return self._data

    def get_processed_df(self):
        df = self._raw_df.sort_values(by=["installation_id", "timestamp", "game_session"])
        install_ids = df.installation_id.unique()

        for install_id in install_ids:
            user_data = df

            session_break_idxs = np.array(user_data[(user_data["game_time"] == 0)].index.append(user_data.tail(1).index))

            gameplay, accuracy_groups = self.process_gameplay(user_data, session_break_idxs)
            if gameplay:
                self.data.append({"install_id": install_id,
                                  "gameplay": gameplay,
                                  "accuracy_groups": accuracy_groups})
        return []

    @staticmethod
    def get_accuracy(df, filter_events=None):

        assessments = df[(df["type"]) == "Assessment"]
        if filter_events is not None:
            assessments = assessments[(assessments["event_data"].str.contains("correct")) &
                                    (assessments["event_code"].isin(filter_events))]
        else:
            assessments = assessments[(assessments["event_data"].str.contains("correct"))]

        assessments["correct"] = False
        assessments.loc[(assessments["event_data"].str.contains("true")), "correct"] = True
        session_ids = assessments["game_session"].unique()
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
        return pd.DataFrame({"game_session": session_ids, "accuracy_group": accuracy_group})

    def encode_event(self, event_data):
        # todo: implement
        day_of_week = utils.time_of_day(event_data["timestamp"])
        time_of_day = utils.day_of_week(event_data["timestamp"])
        title_encode = utils.one_hot_encode(ENUMS.TITLES.index(event_data["title"]), ENUMS.TITLES.__len__())
        type_encode = utils.one_hot_encode(ENUMS.TYPES.index(event_data["type"]), ENUMS.TYPES.__len__())
        world_encode = utils.one_hot_encode(ENUMS.WORLDS.index(event_data["world"]), ENUMS.WORLDS.__len__())
        code_encode = utils.one_hot_encode(ENUMS.EVENT_CODES.index(event_data["event_code"]), ENUMS.EVENT_CODES.__len__())
        event_count = event_data["event_count"]
        game_time = event_data["game_time"]

        encoding = 0
        return encoding

    @staticmethod


    def get_accuracy_groups(self, event_data):
        # todo: implement
        return 0

    def process_gameplay(self, data, break_idxs):
        gameplay = []
        for _, event_data in data.iterrows():
            encoding = self.encode_event(event_data)
            gameplay.append(encoding)
        accuracy_groups = self.get_accuracy_groups(data)
        return gameplay, accuracy_groups



    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def read_df():
        pass


# a = PBSKidsDataset()
# print(a.fpath)