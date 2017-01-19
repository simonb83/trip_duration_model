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
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
import logging
import sys


def regression_report(y_true, y_pred):
    report = ""
    report += "R2 Score:\t\t{}".format(r2_score(y_true, y_pred))
    report += "\n\n"
    report += "median_absolute_error:\t\t{}".format(median_absolute_error(y_true, y_pred))
    report += "\n\n"
    report += "mean_squared_error:\t\t{}".format(mean_squared_error(y_true, y_pred))
    report += "\n\n"
    report += "mean_absolute_error:\t\t{}".format(mean_absolute_error(y_true, y_pred))
    report += "\n\n"
    return report


if __name__ == "__main__":

    logging.basicConfig(filename="output/fit_model.log", level=logging.INFO)

    #Initialize the model
    clf = SGDRegressor()

    logging.info("Starting training")
    i = 1
    # Iterate over the training data
    for chunk in pd.read_hdf('output/train.h5', chunksize=100000):
        logging.info("Processing chunk {}".format(i))
        cols = chunk.columns.tolist()
        cols.remove('duration')
        y = np.ravel(chunk['duration'])
        chunk = chunk.drop('duration', axis=1)

        clf.partial_fit(chunk, y)
        i += 1

    logging.info("Parameters used on training set:\n\n{}\n".format(clf.get_params))


    logging.info("Starting test run")
    i = 1
    # Make some predictions also in chunks
    for chunk in pd.read_hdf('output/test.h5', chunksize=100000):
        logging.info("Processing chunk {}".format(i))
        cols = chunk.columns.tolist()
        cols.remove('duration')
        y = np.ravel(chunk['duration'])
        chunk = chunk.drop('duration', axis=1)

        y_pred = clf.predict(chunk)
        df = pd.DataFrame(np.array([y, y_pred]).T, columns=['True', 'Predicted'])
        df.to_hdf('output/predicted.h5', 'predicted', append=True, format='t')
        i += 1

    logging.info("Detailed classification report:\n")
    df = pd.read_hdf('output/predicted.h5')
    logging.info(regression_report(df['True'], df['Predicted']))
