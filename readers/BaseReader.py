import re
from tqdm import tqdm
import os
import csv


matcher = '(?:^|,)(?=[^"]|(")?)"?((?(1)[^"]*|[^,"]*))"?(?=,|$)'


class BaseReader:
    def __init__(
        self,
        file_name,
        description,
        double_lineskip=False,
    ):
        self.file_name = file_name
        self.double_lineskip = double_lineskip
        self.description = description
        self.corpus_name = re.search("(?:.*\\/)*(.+)\\..*", file_name).group(1)

    def read_lines(self, desc=None):
        headers = []
        if desc is None:
            desc = self.description
        with tqdm(total=os.path.getsize(self.file_name), desc=desc) as pbar:
            with open(self.file_name, "r") as file:
                csv_reader = csv.reader(file, delimiter=",", quotechar='"')
                for i, columns in enumerate(csv_reader):
                    if i == 0:
                        for i, column in enumerate(columns):
                            headers.append(column)
                    else:
                        data = {}
                        for i, value in enumerate(columns):
                            data[headers[i]] = value
                        yield data
                    pbar.update(len(",".join(columns) + "\n"))
