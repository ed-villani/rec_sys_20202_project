import os
from copy import deepcopy
from datetime import datetime
from typing import Union

import numpy as np
from gensim.models import Word2Vec
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from LibRecommender.libreco.data import random_split
except Exception:
    pass
from main import RawDataInteraction


class Item2Vec:
    def __init__(self, data):
        self._data = deepcopy(data)
        self._texts, self._user_recipe_dict, self._index2user, self._user2index, self._index2recipe, self._recipe2index = self._get_train_data()
        del self._data

        self._model = None

    @property
    def recipe2index(self):
        return self._recipe2index

    @property
    def index2recipe(self):
        return self._index2recipe

    @property
    def user2index(self):
        return self._user2index

    @property
    def index2user(self):
        return self._index2user

    @property
    def model(self):
        return self._model

    @property
    def texts(self):
        return self._texts

    @property
    def user_recipe(self):
        return self._user_recipe_dict

    def _get_train_data(self):
        texts = []
        user_recipe_dict = {}

        aux_data = deepcopy(self._data).sort_values(by='time', ascending=True)
        aux_data = aux_data[aux_data['label'] >= 4]
        index2user = {}
        user2index = {}
        index2recipe = {}
        recipe2index = {}
        i = 0
        j = 0
        for row in tqdm(np.array(aux_data)):
            user = row[0]
            recipe = row[1]
            rate = row[2]

            if user not in user2index:
                index2user[i] = user
                user2index[user] = i
                i = i + 1

            if recipe not in recipe2index:
                index2recipe[j] = recipe
                recipe2index[recipe] = j
                j = j + 1

            if user not in user_recipe_dict:
                user_recipe_dict[user] = []
                texts.append([])
            user_recipe_dict[user].append([str(recipe), rate])
            texts[user2index[user]].append(str(recipe))

        del aux_data
        return texts, user_recipe_dict, index2user, user2index, index2recipe, recipe2index

    def train(self,
              iter=5,
              min_count=1,
              size=200,
              workers=os.cpu_count() - 1,
              sg=1,
              hs=0,
              negative=5,
              window=9999999):
        # window = max(len(seq) for seq in tqdm(self._texts)) + 5
        model = Word2Vec(
            sentences=self._texts,
            iter=iter,
            min_count=min_count,
            size=size,
            workers=workers,
            sg=sg,
            hs=hs,
            negative=negative,
            window=window
        )

        self._model = model

    def save(self, path: str):
        self._model.save(f"{path}")


class TrainTestAndValidation:
    def __init__(self, df, test_size, valid_size: float = 0.0, random_state: int = 4):
        TEST_SIZE = test_size
        train, test = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=random_state
        )
        if valid_size:
            VALID_SIZE = valid_size
            train, validation = train_test_split(
                df,
                test_size=VALID_SIZE,
                random_state=random_state
            )
        else:
            validation = None
        print(f'{datetime.now()} - TRAINING DATA')
        print('Shape of input sequences: {}'.format(train.shape))
        print("-" * 50)
        if valid_size:
            print(f'{datetime.now()} - VALIDATION DATA')
            print('Shape of input sequences: {}'.format(validation.shape))
            print("-" * 50)
        print(f'{datetime.now()} - TESTING DATA')
        print('Shape of input sequences: {}'.format(test.shape))

        self._train, self._test, = train, test
        self._validation = validation

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def validation(self):
        return self._validation


class Item2VecRecommender:
    def __init__(self, data: DataFrame, load_model: Union[None, str] = None, save_path: Union[None, str] = None):
        self._data = data
        self._solution = None
        if load_model is not None:
            from gensim.models import Word2Vec
            self._model = Word2Vec.load(load_model)
            self._word_vectors = self._model.wv
            item2vec = Item2Vec(data)
        else:
            item2vec = Item2Vec(data)
            item2vec.train()
            if save_path is not None:
                item2vec.save(save_path)
            self._model = item2vec.model
            self._word_vectors = self._model.wv

        self._user2index = item2vec.user2index
        self._index2user = item2vec.index2user

        self._recipe2index = item2vec.recipe2index
        self._index2recipe = item2vec.index2recipe

        self._user_recipe_dict = item2vec.user_recipe

    def evaluate(self, test):
        self._solution = np.zeros(len(test))
        global_mean = np.mean(self._data['label'])
        for index, row in tqdm(enumerate(np.array(test))):
            user = row[0]
            recipe = row[1]

            if user in self._user2index and recipe in self._recipe2index:
                user_item_similarities = np.array(
                    [self.word_vector.similarity(str(recipe), str(recipe_user[0])) for recipe_user in
                     self._user_recipe_dict[user]])
                user_item_rates = np.array(
                    [recipe_user[1] for recipe_user in
                     self._user_recipe_dict[user]])
                self._solution[index] = np.average(user_item_rates, weights=user_item_similarities)
            elif user in self._user2index and recipe not in self._recipe2index:
                self._solution[index] = np.mean(self._data[self._data['user'] == user]['label'])
            elif user not in self._user2index and recipe in self._recipe2index:
                self._solution[index] = np.mean(self._data[self._data['item'] == recipe]['label'])
            else:
                self._solution[index] = global_mean

    @property
    def model(self):
        return self._model

    @property
    def word_vector(self):
        return self._word_vectors

    @property
    def y_pred(self):
        return self._solution


def main():
    rdi = RawDataInteraction()
    train_test_and_validation = TrainTestAndValidation(rdi(), 0.2)
    item2vec_recommender = Item2VecRecommender(
        train_test_and_validation.train,
        load_model='models/item2vec_embeddings_rate_ge_4'
    )
    item2vec_recommender.evaluate(train_test_and_validation.test)
    y_test = np.array(train_test_and_validation.test['label'])
    y_pred = item2vec_recommender.y_pred
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred, squared=True)}")


if __name__ == '__main__':
    main()
