import warnings

from readers.Predictions import PredictionsReader
from readers.Recipes import RecipesCorpusReader
from readers.generic_readers import RawDataInteraction
from recomenders.ContentBased import ContentBasedRecommender
from recommenders.collaborative_filtering.collaborative_recommender import CollaborativeRecommender
from recommenders.item2vec.item2vec_recommender import Item2VecRecommender
from recommenders.utils.train_test_and_validation import TrainTestAndValidation
import dask.dataframe as dd
import json
from recomenders.ContentBased import ContentBasedRecommender
from readers.Recipes import RecipesCorpusReader
from readers.Predictions import PredictionsReader
from tqdm import tqdm

warnings.warn("deprecated", DeprecationWarning)


def main():
    rdi = RawDataInteraction()
    # try:
    #     cf = CollaborativeRecommender(rdi())
    #     cf.fit()
    #     cf.evaluate()
    # except Exception:
    #     pass
    train_test_and_validation = TrainTestAndValidation(rdi(), 0.2)
    item2vec_recommender = Item2VecRecommender(
        train_test_and_validation.train,
        load_model='models/item2vec_embeddings_rate_ge_4',
        text_type='user'
    )
    item2vec_recommender.evaluate(train_test_and_validation.test)
    item2vec_recommender.scores(train_test_and_validation.test['label'])

    # reader = RecipesCorpusReader("inputs/core-data_recipe.csv")
    # recommender = ContentBasedRecommender(reader, 100, 2, 25)
    # recommender.train_or_load()
    #
    # train_preds = PredictionsReader("inputs/core-data-train_rating.csv")
    # users, items = train_preds.read_cb()
    #
    # test_preds = PredictionsReader("inputs/core-data-test_rating.csv")
    # recommender.score(users, test_preds, items)


if __name__ == "__main__":
    main()
