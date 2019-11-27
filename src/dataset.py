import os
import pandas as pd
import torch
from PIL import Image

class PBSKidsDataset():
    def __init__(self, fpath="", **kwargs):
        super().__init__()
        self._fpath = None
        self._df = None
        self._kwargs = kwargs
        self._users_data = []

        # init data
        self.fpath = fpath
    
    @property
    def df(self):
        return self._df

    @property
    def fpath(self):
        return self._fpath

    @fpath.setter
    def fpath(self, path=""):
        self._df = pd.read_csv(path)
        self._fpath = path
        self._users_data = []

    def encode_event(self, event_data):
        # todo: implement
        encoding = 0
        return encoding

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

    def compose_userdata(self):
        df = self._df.sort_values(by=["installation_id", "timestamp", "game_session"])
        install_ids = df.installation_id.unique()

        for install_id in install_ids:
            user_data = df[(df["installation_id"] == install_id)]
            user_data = user_data[(user_data["event_count"] != user_data["event_count"].shift(-1))]
            session_end_idxs = user_data[(user_data["event_count"] - user_data["event_count"].shift(-1) > 1)].index
            gameplay, accuracy_groups = self.process_gameplay(user_data, session_end_idxs)
            if gameplay:
                self._users_data.append({"install_id": install_id,
                                         "gameplay": gameplay,
                                         "accuracy_groups": accuracy_groups})



    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


    def read_df():
        pass


# a = PBSKidsDataset()
# print(a.fpath)