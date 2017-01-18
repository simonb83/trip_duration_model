"""
Create all features and split the data into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sys


if __name__ == "__main__":

    num_rows = int(sys.argv[1])
    if num_rows == -1:
        num_rows = None

    all_data = pd.read_csv('data/model_data.csv', nrows=num_rows)

    # Need these because did not filter correctly in SQL query and to avoid
    # outliers and data errors
    all_data = all_data[all_data['age'] <= 85]
    all_data = all_data[all_data['hexagon_id'].notnull()]

    all_data['age_bucket'] = pd.cut(all_data['age'], bins=(
        16, 20, 30, 40, 50, 60, 70, 80, 85), include_lowest=True, retbins=False)
    all_data = all_data.drop('age', axis=1)
    all_data = all_data.drop('id', axis=1)

    all_data = pd.get_dummies(
        all_data, columns=['start_hour', 'month', 'hexagon_id', 'age_bucket'], sparse=True)

    train, test = train_test_split(all_data, test_size=0.2, random_state=0)

    train.to_csv('output/train_data.csv', index=None)
    test.to_csv('output/test_data.csv', index=None)
