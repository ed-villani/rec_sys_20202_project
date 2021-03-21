import os
import numpy as np
from copy import deepcopy

from gensim.models import Word2Vec
from tqdm import tqdm


class Item2Vec:
    def __init__(self, data):
        self._data = deepcopy(data)
        self._texts, self._user_recipe_dict, self._index2user, self._user2index, self._index2recipe, self._recipe2index, self._recipe_user_dict = self._get_train_data()
        self._user_avg = self._get_avgs('user')
        self._item_avg = self._get_avgs('item')
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
    def recipe2vec(self):
        return self._texts[0]

    @property
    def user2vec(self):
        return self._texts[1]

    @property
    def user_recipe(self):
        return self._user_recipe_dict

    @property
    def recipe_user(self):
        return self._recipe_user_dict

    @property
    def user_mean_rate(self):
        return self._user_avg

    @property
    def item_mean_rate(self):
        return self._item_avg

    def _get_avgs(self, key):
        data = np.array(self._data.groupby([key]).mean().reset_index()[[key, 'label']])
        data_dict = {}
        for row in tqdm(data):
            data_dict[int(row[0])] = row[1]
        return data_dict

    def _get_train_data(self):
        texts = []
        texts_recipe = []
        user_recipe_dict = {}
        recipe_user_dict = {}

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

            if recipe not in recipe_user_dict:
                recipe_user_dict[recipe] = []
                texts_recipe.append([])
            recipe_user_dict[recipe].append([str(user), rate])
            texts_recipe[recipe2index[recipe]].append(str(user))

            if user not in user_recipe_dict:
                user_recipe_dict[user] = []
                texts.append([])
            user_recipe_dict[user].append([str(recipe), rate])
            texts[user2index[user]].append(str(recipe))

        del aux_data
        return (texts, texts_recipe), user_recipe_dict, index2user, user2index, index2recipe, recipe2index, recipe_user_dict

    def train(self,
              text_type='recipe',
              iter=5,
              min_count=1,
              size=200,
              workers=os.cpu_count() - 1,
              sg=1,
              hs=0,
              negative=5,
              window=9999999):
        # window = max(len(seq) for seq in tqdm(self._texts)) + 5
        if text_type == 'user':
            texts = self.user2vec
        if text_type == 'recipe':
            texts = self.recipe2vec
        model = Word2Vec(
            sentences=texts,
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

