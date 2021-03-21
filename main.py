import pandas as pd
import dask.dataframe as dd
import json
from recomenders.ContentBased import ContentBasedRecommender
from readers.Recipes import RecipesCorpusReader
from readers.Predictions import PredictionsReader
from tqdm import tqdm


def read_data(file_path: str):
    data = []
    with open(file=file_path, mode="r") as f:
        for line in f.readlines():
            if line == '"\n':
                continue
            line = line.replace("\n", "").replace('"', "")
            data.append(line.split(","))
    f.close()
    return data
    # return pd.DataFrame(data=data[1:], columns=data[0])


def write_fixed_data(output_path: str, data: list):
    import csv

    with open(output_path, "w", newline="") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(data)
    f.close()


def read_and_fix_data(input_path: str, output_path: str):
    data = read_data(input_path)
    write_fixed_data(output_path, data)


def main():
    read_and_fix_data(
        "inputs/original/raw-data_interaction.csv",
        "inputs/fixed/raw-data_interaction_fixed.csv",
    )
    # dd.read_csv("inputs/fixed/raw-data_interaction_fixed.csv")


if __name__ == "__main__":

    reader = RecipesCorpusReader("inputs/core-data_recipe.csv")
    recommender = ContentBasedRecommender(reader, 100, 2, 25)
    recommender.train_or_load()

    train_preds = PredictionsReader("inputs/core-data-train_rating.csv")
    users, items = train_preds.read_cb()

    test_preds = PredictionsReader("inputs/core-data-test_rating.csv")
    recommender.score(users, test_preds, items)
