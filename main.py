import warnings
from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from recommenders.item2vec.item2vec_recommender import Item2VecRecommender
from recommenders.utils.train_test_and_validation import TrainTestAndValidation

try:
    from LibRecommender.libreco.algorithms import SVDpp
    from LibRecommender.libreco.evaluation import evaluate
    from LibRecommender.libreco.data import random_split, DatasetPure
except Exception:
    pass

warnings.warn("deprecated", DeprecationWarning)


class DataReaderBase:
    def __init__(self, file_path: str, names: Union[None, List[str]] = None, nrows: Union[None, int] = None):
        self._df = self._df = pd.read_csv(
            file_path,
            header=0,
            quotechar='"',
            names=names,
            nrows=nrows
        )

    def __call__(self):
        return self._df


class RawDataRecipe(DataReaderBase):
    def __init__(self):
        super().__init__("inputs/original/raw-data_recipe.csv")


class RawDataInteraction(DataReaderBase):
    def __init__(self):
        super().__init__("inputs/original/raw-data_interaction.csv", ["user", "item", "label", "time"])


class CoreDataRecipe(DataReaderBase):
    def __init__(self, nrows: Union[None, int] = None):
        super().__init__("inputs/original/core-data_recipe.csv", nrows=nrows)


def main():
    # rdi = RawDataInteraction()
    #
    # train_data, eval_data, test_data = random_split(rdi(), multi_ratios=[0.64, 0.16, 0.2])
    # train_data, data_info = DatasetPure.build_trainset(train_data)
    # eval_data = DatasetPure.build_evalset(eval_data)
    # test_data = DatasetPure.build_testset(test_data)
    # print(data_info)
    #
    # svdpp = SVDpp(
    #     task="rating",
    #     data_info=data_info,
    #     embed_size=16,
    #     n_epochs=5,
    #     lr=0.001,
    #     reg=None,
    #     batch_size=256
    # )
    #
    # svdpp.fit(train_data, verbose=2, eval_data=eval_data, metrics=["rmse", "mae", "r2"])
    # svdpp.save(
    #     path='models/rec_sys_project_cf',
    #     model_name='rec_sys_project_cf')
    # # do final evaluation on test data
    # print("evaluate_result: ", evaluate(model=svdpp, data=test_data, metrics=["rmse", "mae"]))

    rdi = RawDataInteraction()
    train_test_and_validation = TrainTestAndValidation(rdi(), 0.2)
    item2vec_recommender = Item2VecRecommender(
        train_test_and_validation.train,
        load_model='models/item2vec_embeddings_rate_ge_4',
        text_type='recipe'
    )
    item2vec_recommender.evaluate(train_test_and_validation.test)
    y_test = np.array(train_test_and_validation.test['label'])
    y_pred = item2vec_recommender.y_pred
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred, squared=True)}")


if __name__ == '__main__':
    main()
