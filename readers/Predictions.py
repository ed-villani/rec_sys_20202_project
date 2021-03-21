from readers.BaseReader import BaseReader
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
import re
from tqdm import tqdm
import os


class CBUser:
    def __init__(self):
        self.items = []
        self.ratings = []
        self.mean = 0
        self.sum = 0
        self.count = 0

    def normalize(self):
        self.mean = self.sum / self.count
        self.ratings = [rating - self.mean for rating in self.ratings]


class CBItem:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def get_mean(self):
        return self.sum / self.count


class PredictionsReader(BaseReader):
    def __init__(
        self,
        file_name,
        user_id_column="user_id",
        item_id_column="recipe_id",
        rating_column="rating",
        double_lineskip=True,
    ):
        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.rating_column = rating_column
        super().__init__(file_name, "Reading Predictions", double_lineskip)

    def read_cb(self, desc="Reading content-based data"):
        users = {}
        items = {}
        for line in super().read_lines(desc):
            user_id = int(line[self.user_id_column])
            item_id = int(line[self.item_id_column])
            rating = float(line[self.rating_column])
            if user_id not in users:
                users[user_id] = CBUser()
            if item_id not in items:
                items[item_id] = CBItem()
            users[user_id].items.append(item_id)
            users[user_id].ratings.append(rating)
            users[user_id].sum += rating
            users[user_id].count += 1
            items[item_id].sum += rating
            items[item_id].count += 1

        for _, user in tqdm(users.items(), desc="Normalizing user ratings"):
            user.normalize()

        id2mean = {}
        for id, item in tqdm(items.items(), desc="Calculating item averages"):
            id2mean[id] = item.get_mean()

        return users, id2mean

    def read_lines(self, desc="Reading file"):
        for line in super().read_lines(desc):
            user_id = int(line[self.user_id_column])
            item_id = int(line[self.item_id_column])
            rating = float(line[self.rating_column])
            yield user_id, item_id, rating