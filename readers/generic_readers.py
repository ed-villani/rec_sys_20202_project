from typing import Union, List

import pandas as pd


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
