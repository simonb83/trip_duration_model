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
from sklearn.linear_model import SGDRegressor
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


if __name__ == "__main__":

    logging.basicConfig(filename="output/tune_regressor_params.log", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--chunksize", help="Chunksize for processing training data")

    args = parser.parse_args()
    chunksize = int(args.chunksize)

    cross_validation_results = {}

    loss_functions = ['squared_loss', 'epsilon_insensitive']
    penalty = ['l2', 'l1']
    epsilon = 60
    num_iterations = 1
    random_state = 1
    eta0 = [0.01, 0.001, 0.0001]

    store = pd.HDFStore('output/train.h5')
    nrows = store.get_storer('train').nrows

    for l in loss_functions:
        for p in penalty:
            for e in eta0:
                parameters = "Loss: {}, Penalty: {}, LR: {}".format(l, p, e)
                logging.info("Starting run with params: {}".format(parameters))

                # Initialize the model
                clf = SGDRegressor(
                    loss=l, penalty=p, n_iter=num_iterations, epsilon=epsilon, random_state=random_state, eta0=e)

                logging.info("Starting cross_fold validation")

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
                        y = np.ravel(X['duration'])
                        X = X.drop('duration', axis=1)
                        clf.partial_fit(X, y)

                    logging.info("Validatiing on holdout fold")
                    # Test on the holdout set
                    for chunk in chunks(test_index, chunksize):
                        X = pd.read_hdf('output/train.h5',
                                        'train', where=pd.Index(chunk))
                        X = X.drop('Unnamed: 0', axis=1)
                        cols = X.columns.tolist()
                        cols.remove('duration')
                        y_true = np.ravel(X['duration'])
                        X = X.drop('duration', axis=1)

                        score = clf.score(X, y_true)
                        scores.append(score)

                # Average scores and add to dictionary
                cross_validation_results[parameters] = np.mean(scores)

    for k, v in sorted(cross_validation_results.items(), key=lambda x: -x[1]):
        logging.info("{}: {}".format(k, v))
