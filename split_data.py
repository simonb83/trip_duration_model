"""
Create all features and split the data into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

def process_data(df):
    df[['month', 'hexagon_id', 'start_hour']] = df[['month', 'hexagon_id', 'start_hour']].astype(np.dtype('i4'))

    for i in range(1, 13):
        df['month_{}'.format(i)] = df['month'].apply(lambda x: x == i)
    df = df.drop('month', axis=1)

    for i in [0] + [i for i in range(5, 24)]:
        df['hour_{}'.format(i)] = df['start_hour'].apply(lambda x: x == i)
    df = df.drop('start_hour', axis=1)

    for h in hexagons:
        df['hex_{}'.format(h)] = df['hexagon_id'].apply(lambda x: x == h)
    df = df.drop('hexagon_id', axis=1)

    for d in range(7):
        df['dow_'.format(d)] = df['dow'].apply(lambda x: x == d)
    df = df.drop('dow', axis=1)

    return df


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

    # Process the training data
    all_data = pd.read_csv('output/train.csv')

    all_data = process_data(all_data)

    age_scaler = StandardScaler()
    all_data['age'] = all_data['age'].astype(np.float)
    age_scaler.fit(np.array(all_data['age']).reshape(-1, 1))
    all_data['age'] = age_scaler.transform(np.array(all_data['age']).reshape(-1, 1))

    temp_scaler = StandardScaler()
    all_data['temp'] = all_data['temp'].astype(np.float)
    temp_scaler.fit(np.array(all_data['temp']).reshape(-1, 1))
    all_data['temp'] = temp_scaler.transform(np.array(all_data['temp']).reshape(-1, 1))

    new_cols = all_data.columns.tolist()
    new_cols.remove('duration')
    np.random.shuffle(new_cols)
    new_cols = ['duration'] + new_cols

    all_data = all_data[new_cols]

    all_data.to_hdf('output/train.h5', 'train', format='t')

    # Now process the test data
    all_data = pd.read_csv('output/test.csv')

    all_data = process_data(all_data)

    all_data['age'] = all_data['age'].astype(np.float)
    all_data['age'] = age_scaler.transform(np.array(all_data['age']).reshape(-1, 1))
    all_data['temp'] = all_data['temp'].astype(np.float)
    all_data['temp'] = temp_scaler.transform(np.array(all_data['temp']).reshape(-1, 1))

    all_data = all_data[new_cols]

    all_data.to_hdf('output/test.h5', 'test', format='t')