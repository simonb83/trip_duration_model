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
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
import logging


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
    logging.basicConfig(filename="output/paramater_tuning.log", level=logging.INFO)

    cols = pd.read_csv('output/train_data.csv', nrows=1).columns.tolist()
    cols.remove('duration')

    X_train = pd.read_csv('output/train_data.csv', usecols=cols)
    y_train = np.ravel(pd.read_csv(
        'output/train_data.csv', usecols=['duration']))

    params = {
        'n_estimators': [10, 50, 100, 250, 500, 1000],
        'max_features': ['auto', 'sqrt', 0.5],
        'min_samples_leaf': [1, 10, 50]
    }

    clf = GridSearchCV(RandomForestRegressor(random_state=1), params, cv=5, n_jobs=4)
    clf.fit(X_train, y_train)

    logging.info("Best parameters set found on training set:\n\n{}\n".format(clf.best_params_))

    X_test = pd.read_csv('output/test_data.csv', usecols=cols)
    y_test = np.ravel(pd.read_csv(
        'output/test_data.csv', usecols=['duration']))

    logging.info("Detailed classification report:\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    logging.info(regression_report(y_true, y_pred))

    np.save("output/predictions.npy", y_pred)
