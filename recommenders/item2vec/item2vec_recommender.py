import numpy as np
from typing import Union

from sklearn.metrics import mean_squared_error

from recommenders.item2vec.item2vec import Item2Vec
from pandas import DataFrame
from tqdm import tqdm


class Item2VecRecommender:
    def __init__(self, data: DataFrame, load_model: Union[None, str] = None, save_path: Union[None, str] = None, text_type: str = 'recipe'):
        self._data = data
        self._solution = None
        self._text_type = text_type
        if load_model is not None:
            from gensim.models import Word2Vec
            self._model = Word2Vec.load(load_model)
            self._word_vectors = self._model.wv
            item2vec = Item2Vec(data)
        else:
            item2vec = Item2Vec(data)
            item2vec.train(text_type=text_type)
            if save_path is not None:
                item2vec.save(save_path)
            self._model = item2vec.model
            self._word_vectors = self._model.wv

        self._user2index = item2vec.user2index
        self._index2user = item2vec.index2user

        self._recipe2index = item2vec.recipe2index
        self._index2recipe = item2vec.index2recipe

        self._user_recipe_dict = item2vec.user_recipe
        self._recipe_user_dict = item2vec.recipe_user
        self._user_mean_rate = item2vec.user_mean_rate
        self._item_mean_rate = item2vec.item_mean_rate

    def evaluate(self, test):
        def similarities(key, subkey, main_dict):
            similarities_list = []
            rates = []
            for slave_key in main_dict[key]:
                try:
                    similarities_list.append(self.word_vector.similarity(str(subkey), str(slave_key[0])))
                    rates.append(slave_key[1])
                except KeyError:
                    continue
            rates = np.array(rates, dtype=np.float32)
            similarities_list = np.array(similarities_list, dtype=np.float32)
            return np.average(rates, weights=similarities_list)
        self._solution = np.zeros(len(test))
        global_mean = np.mean(self._data['label'])
        for index, row in tqdm(enumerate(np.array(test))):
            user = row[0]
            recipe = row[1]
            data_dict = self._user_recipe_dict if self._text_type == 'recipe' else self._recipe_user_dict
            if user in self._user2index and recipe in self._recipe2index:
                key = recipe if self._text_type == 'user' else user
                subkey = user if self._text_type == 'user' else recipe
                try:
                    self._solution[index] = similarities(key, subkey, data_dict)
                except ZeroDivisionError:
                    self._solution[index] = self._item_mean_rate[recipe]
            elif user in self._user2index and recipe not in self._recipe2index:
                self._solution[index] = self._user_mean_rate[user]
            elif user not in self._user2index and recipe in self._recipe2index:
                self._solution[index] = self._item_mean_rate[recipe]
            else:
                self._solution[index] = global_mean

    def scores(self, y_test):
        y_test = np.array(y_test)
        y_pred = self.y_pred
        print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
        print(f"MSE: {mean_squared_error(y_test, y_pred, squared=True)}")

    @property
    def model(self):
        return self._model

    @property
    def word_vector(self):
        return self._word_vectors

    @property
    def y_pred(self):
        return self._solution

