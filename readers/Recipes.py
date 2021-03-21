from readers.BaseReader import BaseReader
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
import re
import json


class RecipesCorpusReader(BaseReader):
    def __init__(
        self,
        file_name,
        id_column="recipe_id",
        ignore_columns=set(["image_url", "nutritions"]),
    ):
        self.id_column = id_column
        self.ignore_columns = ignore_columns
        super().__init__(file_name, "Reading Corpus")

    def read(self):
        doc_corpus = []
        id2index = {}
        for index, line in enumerate(super().read_lines()):
            id = int(line[self.id_column])
            whole_text = "\n".join(
                [
                    str(col)
                    for key, col in line.items()
                    if key != self.id_column and key not in self.ignore_columns
                ]
            )
            tokens = simple_preprocess(whole_text)
            doc_corpus.append(TaggedDocument(tokens, [id]))
            id2index[id] = index
        return doc_corpus, id2index


class NutrientsReader(BaseReader):
    def __init__(
        self,
        file_name,
        id_col="recipe_id",
        nutrition_col="nutritions",
    ):
        self.file_name = file_name
        self.nutrition_col = nutrition_col
        self.id_col = id_col
        super().__init__(file_name, "Reading Nutrients")

    def read(self):
        for line in super().read_lines():
            nutrient_raw = (
                line[self.nutrition_col]
                .replace("u'", '"')
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("None", "null")
                .replace("< 1", "0.05")
            )
            nutrients = json.loads(nutrient_raw)
            yield int(line[self.id_col]), line["recipe_name"], line[
                "ingredients"
            ], line["cooking_directions"], nutrients
