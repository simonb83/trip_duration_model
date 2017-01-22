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
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
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
        filename="output/tune_classifier_params.log", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--sample", help="Sample size")

    args = parser.parse_args()
    sample = int(args.sample)

    cross_validation_results = {}

    n_estimators = [10, 100, 500]
    max_features = ['sqrt', 'log2', 10, 0.4]
    random_state = 1

    all_classes = [1, 2, 3, 4]
    class_weights = {1: 0.708, 2: 0.237, 3: 0.049, 4: 0.006}

    store = pd.HDFStore('output/train.h5')
    nrows = store.get_storer('train').nrows
    r = np.random.randint(0, nrows, size=sample)
    data = pd.read_hdf('output/train.h5', 'train', where=pd.Index(r))
    data = pre_process_data(data)

    for n in n_estimators:
        for m in max_features:
            parameters = "N_estimators: {}, Max_features: {}".format(
                n, m)

            clf = RandomForestClassifier(
                n_estimators=n, max_features=m, class_weight=class_weights)
            logging.info("Starting cross-fold validation")

            kf = KFold(sample, n_folds=3, shuffle=True)
            scores = []
            for train_index, test_index in kf:
                X_train, y_train = parse_data(data.loc[train_index])
                clf.fit(X_train, y_train)
                X_test, y_test = parse_data(data.loc[test_index])
                score = clf.score(X_test, y_test)
                scores.append(score)
            avg_score = np.mean(scores)
            cross_validation_results[parameters] = avg_score
            logging.info("{}: {}".format(parameters, avg_score))

    logging.info("\n\nFINAL RESULTS\n\n")

    for k, v in sorted(cross_validation_results.items(), key=lambda x: -x[1]):
        logging.info("{}: {}".format(k, v))
