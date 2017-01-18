"""
Create all features and split the data into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sys


if __name__ == "__main__":

    hexagons = pd.read_csv('data/hexagons.csv', header=None)[0].tolist()

    num_rows = int(sys.argv[1])
    if num_rows == -1:
        num_rows = None

    all_data = pd.read_csv('data/model_data.csv', nrows=num_rows)
    all_data = all_data[all_data['age'] <= 85]
    all_data = all_data[all_data['hexagon_id'].notnull()]
    all_data = all_data.drop('id', axis=1)

    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    train_top = int(np.floor(0.8 * len(all_data)))
    train_idx, test_idx = indices[:train_top], indices[train_top:]
    all_data.iloc[train_idx].to_csv('output/train.csv')
    all_data.iloc[test_idx].to_csv('output/test.csv')

    for f in ['output/train.csv', 'output/test.csv']:
        all_data = pd.read_csv(f)
        all_data['age_bucket'] = pd.cut(all_data['age'], bins=(
            16, 20, 30, 40, 50, 60, 70, 80, 85), include_lowest=True, retbins=False)
        all_data = all_data.drop('age', axis=1)
        all_data[['month', 'hexagon_id', 'start_hour']] = all_data[['month', 'hexagon_id', 'start_hour']].astype(np.dtype('i4'))

        for i in range(1, 13):
            all_data['month_{}'.format(i)] = all_data['month'].apply(lambda x: x == i)
        all_data = all_data.drop('month', axis=1)

        for i in [0] + [i for i in range(5, 24)]:
            all_data['hour_{}'.format(i)] = all_data['start_hour'].apply(lambda x: x == i)

        all_data = all_data.drop('start_hour', axis=1)

        for b in ['[16, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 70]', '(70, 80]', '(80, 85]']:
            all_data['bucket_{}'.format(b)] = all_data['age_bucket'].apply(lambda x: x == b)
        all_data = all_data.drop('age_bucket', axis=1)

        for h in hexagons:
            all_data['hex_{}'.format(h)] = all_data['hexagon_id'].apply(lambda x: x == h)
        all_data = all_data.drop('hexagon_id', axis=1)

        all_data.to_csv(f)