"""
Train a machine learning model on Ecobici trips data for trips between 2014-01-01 AND 2016-07-31

The aim is to predict bicycle trip duration in seconds based on the following features:

- Gender
- Age
- Weekday vs. Weekend
- Hour of Day
- Month
- Start location based on hexagonal grid

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import logging
import argparse


def duration_classes(x):
    if x <= 60 * 15:
        return 1
    if x <= 60 * 30:
        return 2
    if x <= 60 * 45:
        return 3
    if x <= 60 * 60:
        return 4


def pre_process_data(data):
    data['duration'] = data['duration'].apply(lambda x: duration_classes(x))
    data.reset_index(inplace=True, drop=True)
    return data


def parse_data(data):
    y = np.ravel(data['duration'])
    data = data.drop('duration', axis=1)
    return data, y

if __name__ == "__main__":

    logging.basicConfig(
        filename="output/fit_classifier_model.log", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--estimators", help="Number of estimators")
    parser.add_argument(
        "-f", "--features", help="Max features")
    parser.add_argument(
        "-s", "--sample", help="Training sample size")
    parser.add_argument(
        "-t", "--test", help="Test sample size")
    args = parser.parse_args()

    n_estimators = int(args.estimators)
    features = str(args.features)
    train_sample = int(args.sample)
    test_sample = int(args.test)

    class_weights = {1: 0.708, 2: 0.237, 3: 0.049, 4: 0.006}

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_features=features, class_weight=class_weights)

    store = pd.HDFStore('output/train.h5')
    nrows = store.get_storer('train').nrows
    r = np.random.randint(0, nrows, size=train_sample)
    data = pd.read_hdf('output/train.h5', 'train', where=pd.Index(r))
    data = pre_process_data(data)

    train_X, train_y = parse_data(train_data)
    clf.fit(train_X, train_y)

    store = pd.HDFStore('output/test.h5')
    nrows = store.get_storer('test').nrows
    r = np.random.randint(0, nrows, size=test_sample)
    data = pd.read_hdf('output/test.h5', 'test', where=pd.Index(r))
    data = pre_process_data(data)

    test_X, test_y = parse_data(train_data)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)

    df = pd.DataFrame(np.array([test_y, y_pred]).T,
                      columns=['True', 'Predicted'])
    df.to_hdf('output/predicted_class.h5',
              'predicted_class', append=True, format='t')

    logging.info("Detailed classification report:\n")
    df = pd.read_hdf('output/predicted_class.h5')
    logging.info(classification_report(df['True'], df[
                 'Predicted']), target_names=all_classes)
