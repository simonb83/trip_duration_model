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
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import logging
import argparse


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

    logging.basicConfig(filename="output/fit_classifier_model.log", level=logging.INFO)

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

    all_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_weights = {
        1: 0.01, 2: 0.14, 3: 0.34, 4: 0.22, 5: 0.12,
        6: 0.07, 7: 0.04, 8: 0.05, 9: 0.01}

    clf = SGDClassifier(
        loss=loss, penalty=penalty, n_iter=num_iterations, epsilon=epsilon, random_state=random_state, eta0=eta, class_weight=class_weights)

    # Iterate over the training data
    i = 1
    for chunk in pd.read_hdf('output/train.h5', chunksize=chunksize):
        logging.info("Processing chunk {}".format(i))
        chunk = chunk.drop('Unnamed: 0', axis=1)
        cols = chunk.columns.tolist()
        cols.remove('duration')
        chunk['duration'] = chunk['duration'].apply(lambda x: duration_classes(x))
        y = np.ravel(chunk['duration'])
        chunk = chunk.drop('duration', axis=1)
        clf.partial_fit(chunk, y, classes=all_classes)
        i += 1

    logging.info("Starting test run")
    i = 1
    # Make some predictions also in chunks
    for chunk in pd.read_hdf('output/test.h5', chunksize=chunksize):
        logging.info("Processing chunk {}".format(i))
        chunk = chunk.drop('Unnamed: 0', axis=1)
        cols = chunk.columns.tolist()
        cols.remove('duration')
        chunk['duration'] = chunk['duration'].apply(lambda x: duration_classes(x))
        y = np.ravel(chunk['duration'])
        chunk = chunk.drop('duration', axis=1)

        y_pred = clf.predict(chunk)
        df = pd.DataFrame(np.array([y, y_pred]).T,
                          columns=['True', 'Predicted'])
        df.to_hdf('output/predicted_class.h5', 'predicted_class', append=True, format='t')
        i += 1

    logging.info("Save model to disk:\n")
    joblib.dump(clf, 'models/classifier/classifier.pkl')
    logging.info("Detailed classification report:\n")
    df = pd.read_hdf('output/predicted_class.h5')
    logging.info(classification_report(df['True'], df['Predicted']), target_names=all_classes)
