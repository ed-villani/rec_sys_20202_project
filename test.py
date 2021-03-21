import os

import pandas as pd
from gensim.models import Word2Vec

from main import CoreDataRecipe


def main():
    df3 = CoreDataRecipe(nrows=12000)
    df3 = df3()
    print(df3.head())
    ingredient_list = [f'{row[3].lower().replace("^", ",")},{row[0]},{row[1].lower().replace(",", "")}' for row in df3.values]
    # df = pd.read_csv('data.csv')
    # df['Maker_Model'] = df['Make'] + " " + df['Model']
    # df1 = df[
    #     ['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style',
    #      'Maker_Model']]
    # df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)
    # df_clean = pd.DataFrame({'clean': df2})
    # sent = [row.split(',') for row in df_clean['clean']]
    # print(sent[:2])
    sent = [ingredient.split(',') for ingredient in ingredient_list]
    model = Word2Vec(sent, min_count=1, size=30, workers=os.cpu_count() - 1, window=3, sg=1)
    print(model.similarity('240488', '16881'))


if __name__ == '__main__':
    main()
