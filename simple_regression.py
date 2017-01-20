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
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
from sklearn.externals import joblib
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

def clean_chunk(df, hexagons):
    df = df.drop('Unnamed: 0', axis=1)
    cols = df.columns.tolist()
    cols.remove('duration')
    cols.remove('temp')
    cols.remove('dow_')
    for i in (1, 13):
        cols.remove("month_{}".format(i))
    for h in hexagons:
        cols.remove("hex_{}".format(h))
    y = np.ravel(df['duration'])
    return df[cols], y


    y = np.ravel(chunk['duration'])

if __name__ == "__main__":

    logging.basicConfig(filename="output/simple_model.log", level=logging.INFO)

    hexagons = pd.read_csv('data/hexagons.csv', header=None)[0].tolist()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--loss", help="Loss function")
    parser.add_argument(
        "-p", "--penalty", help="Penalty")
    parser.add_argument(
        "-e", "--eta", help="Eta0")
    parser.add_argument(
        "-c", "--chunksize", help="Chunksize for processing training data")
    args = parser.parse_args()

    loss = str(args.loss)
    penalty = str(args.penalty)
    eta = float(args.eta)
    chunksize = int(args.chunksize)

    epsilon = 60
    num_iterations = 1
    random_state = 1

    clf = SGDRegressor(
        loss=loss, penalty=penalty, n_iter=num_iterations, epsilon=epsilon, random_state=random_state, eta0=eta)

                # Iterate over the training data
    i = 1
    for chunk in pd.read_hdf('output/train.h5', chunksize=chunksize):
        logging.info("Training with chunk {}".format(i))
        X, y = clean_chunk(chunk)
        clf.partial_fit(X, y)
        i += 1

    i = 1
    # Make some predictions also in chunks
    for chunk in pd.read_hdf('output/test.h5', chunksize=chunksize):
        logging.info("Testing with chunk {}".format(i))
        X, y = clean_chunk(chunk)

        y_pred = clf.predict(chunk)
        df = pd.DataFrame(np.array([y, y_pred]).T,
                          columns=['True', 'Predicted'])
        df.to_hdf('output/predicted.h5', 'predicted', append=True, format='t')
        i += 1

    # logging.info("Save model to disk:\n")
    # joblib.dump(clf, 'models/regressor/regressor.pkl') 
    logging.info("Detailed Regression report:\n")
    df = pd.read_hdf('output/predicted.h5')
    logging.info(regression_report(df['True'], df['Predicted']))
    
