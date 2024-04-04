import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA



'''
python3 weather_city.py monthly-data-labelled.csv monthly-data-unlabelled.csv predictions.csv
'''

def main():
    data = pd.read_csv(sys.argv[1])
    unlabelled_data = pd.read_csv(sys.argv[2])

    # ----- Predicting City ------
    train, test = train_test_split(data, test_size=0.3)

    X_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    X_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]

    X_predict = unlabelled_data.iloc[:, 1:]

    std_scalar = StandardScaler()
    X_train = std_scalar.fit_transform(X_train)
    X_test = std_scalar.fit_transform(X_test)
    X_predict = std_scalar.fit_transform(X_predict)

    nb = KNeighborsClassifier()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_predict)
    nb_score = nb.score(X_test, y_test)
    print(nb_score)

    pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)


if __name__ == '__main__':
    main()