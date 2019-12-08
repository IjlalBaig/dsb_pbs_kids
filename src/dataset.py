import os
import pandas as pd
import numpy as np
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
            data_filter = {"min_session_events": self._kwargs.get("min_session_events", 1)}
            user_data = self.filter_userdata(df, install_id, data_filter)

            session_break_idxs = np.array(user_data[(user_data["game_time"] == 0)].index.append(user_data.tail(1).index))

            gameplay, accuracy_groups = self.process_gameplay(user_data, session_break_idxs)
            if gameplay:
                self._users_data.append({"install_id": install_id,
                                         "gameplay": gameplay,
                                         "accuracy_groups": accuracy_groups})
        return []

    @staticmethod
    def filter_userdata(df, install_id, data_filters={}):
        user_data = df[(df["installation_id"] == install_id)]
        # clear misclicked events
        user_data = user_data[(user_data["event_count"] != user_data["event_count"].shift(-1))]

        for key, val in data_filters.items():
            if key == "min_session_events" and val > 1:
                session_break_idxs = np.array(user_data[(user_data["game_time"] == 0)].index.append(user_data.tail(1).index))
                session_lengths = np.roll(session_break_idxs, -1) - session_break_idxs + 1
                invalid_ranges = list(zip(session_break_idxs[np.flatnonzero(session_lengths[:-1] < val)],
                                          session_break_idxs[np.flatnonzero(session_lengths[:-1] < val) + 1] + 1))
                invalid_rows = list(itertools.chain(*[list(range(*range_)) for range_ in invalid_ranges]))
                user_data = user_data.drop(invalid_rows)

        return user_data

    def encode_event(self, event_data):
        # todo: implement
        pd_timestamp = pd.Timestamp(event_data["timestamp"])
        day_of_week = pd_timestamp.dayofweek
        time_of_day = pd_timestamp.hour * 60 * 60 + pd_timestamp.minute * 60 + pd_timestamp.second
        title_encode = self.one_hot_encode(ENUMS.TITLES.index(event_data["title"]), ENUMS.TITLES.__len__())
        type_encode = self.one_hot_encode(ENUMS.TYPES.index(event_data["type"]), ENUMS.TYPES.__len__())
        world_encode = self.one_hot_encode(ENUMS.WORLDS.index(event_data["world"]), ENUMS.WORLDS.__len__())
        code_encode = self.one_hot_encode(ENUMS.EVENT_CODES.index(event_data["event_code"]), ENUMS.EVENT_CODES.__len__())
        event_count = event_data["event_count"]
        game_time = event_data["game_time"]

        encoding = 0
        return encoding

    @staticmethod
    def one_hot_encode(val, max_val):
        encode = np.zeros(max_val)
        encode[val] = 1.0
        return encode

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