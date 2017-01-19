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
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import KFold
import logging
import argparse


def regression_report(y_true, y_pred):
    report = ""
    report += "R2 Score:\t\t{}".format(r2_score(y_true, y_pred))
    report += "\n\n"
    report += "median_absolute_error:\t\t{}".format(
        median_absolute_error(y_true, y_pred))
    report += "\n\n"
    report += "mean_squared_error:\t\t{}".format(
        mean_squared_error(y_true, y_pred))
    report += "\n\n"
    report += "mean_absolute_error:\t\t{}".format(
        mean_absolute_error(y_true, y_pred))
    report += "\n\n"
    return report


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def duration_classes(x):
    if x <= 120:
        return 1
    elif x <= 300:
        return 2
    elif x <= 600:
        return 3
    elif x <= 900:
        return 4
    elif x <= 1200:
        return 5
    elif x <= 1500:
        return 6
    elif x <= 1800:
        return 7
    elif x <= 2700:
        return 8
    else:
        return 9


if __name__ == "__main__":

    logging.basicConfig(filename="output/tune_classifier_params.log", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--chunksize", help="Chunksize for processing training data")

    args = parser.parse_args()
    chunksize = int(args.chunksize)

    cross_validation_results = {}

    loss_functions = ['hinge', 'log']
    penalty = ['l2', 'l1']
    epsilon = 60
    num_iterations = 1
    random_state = 1
    eta0 = [0.01, 0.001, 0.0001]

    store = pd.HDFStore('output/train.h5')
    nrows = store.get_storer('train').nrows

    all_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_weights = {
        1: 0.01, 2: 0.14, 3: 0.34, 4: 0.22, 5: 0.12,
        6: 0.07, 7: 0.04, 8: 0.05, 9: 0.01 }

    for l in loss_functions:
        for p in penalty:
            for e in eta0:
                parameters = "Loss: {}, Penalty: {}, LR: {}".format(l, p, e)
                logging.info("Starting run with params: {}".format(parameters))

                # Initialize the model
                clf = SGDClassifier(
                    loss=l, penalty=p, n_iter=num_iterations, epsilon=epsilon, random_state=random_state, eta0=e, class_weight=class_weights)

                logging.info("Starting cross-fold validation")

                kf = KFold(nrows, n_folds=3, shuffle=True)

                scores = []
                for train_index, test_index in kf:

                    logging.info("Training")
                    # Train using chunks
                    for chunk in chunks(train_index, chunksize):
                        logging.info(chunk)
                        X = pd.read_hdf('output/train.h5',
                                        'train', where=pd.Index(chunk))
                        X = X.drop('Unnamed: 0', axis=1)
                        cols = X.columns.tolist()
                        cols.remove('duration')
                        X['duration'] = X['duration'].apply(lambda x: duration_classes(x))
                        y = np.ravel(X['duration'])
                        X = X.drop('duration', axis=1)
                        clf.partial_fit(X, y, classes=all_classes)

                    logging.info("Validatiing on holdout fold")
                    # Test on the holdout set
                    for chunk in chunks(test_index, chunksize):
                        X = pd.read_hdf('output/train.h5',
                                        'train', where=pd.Index(chunk))
                        X = X.drop('Unnamed: 0', axis=1)
                        cols = X.columns.tolist()
                        cols.remove('duration')
                        X['duration'] = X['duration'].apply(lambda x: duration_classes(x))
                        y_true = np.ravel(X['duration'])
                        X = X.drop('duration', axis=1)

                        score = clf.score(X, y_true)
                        scores.append(score)

                # Average scores and add to dictionary
                cross_validation_results[parameters] = np.mean(scores)

    for k, v in sorted(cross_validation_results.items(), key=lambda x: -x[1]):
        logging.info("{}: {}".format(k, v))
