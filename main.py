from KNN import KNN
from NaiveBayes import NaiveBayes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from time import time

def is_categorical(col):
    if set(col) == set(range(col.unique().shape[0])):
        return True
    if col.dtype in (object, bool):
        return True
    return False

def min_max_normalization(col):
    if not is_categorical(col):
        col_mean, col_stddev = np.mean(col), np.std(col)
        for i in range(col.shape[0]):
            col.iloc[i] = (col.iloc[i]-col_mean)/col_stddev
    return col

def z_score_normalization(col):
    if not is_categorical(col):
        col_min, col_max = col.min(), col.max()
        for i in range(col.shape[0]):
            col.iloc[i] = (col.iloc[i]-col_min)/(col_max-col_min)
    return col

def normalizer(col, method):
    if method.lower() == 'z-score':
        return z_score_normalization(col)
    if method.lower() == 'minmax':
        return min_max_normalization(col)

def main():
    while True:
        norm_method = input('Please select your desired normalization method:\n1. MinMax\n2. Z-Score\n>>> ')
        if norm_method in ['1', '2']:
            norm_method = 'minmax' if norm_method == '1' else 'z-score'
            break
        elif norm_method.lower() in ['minmax', 'z-score']:
            norm_method = norm_method.lower()
            break
        else:
            print('Invalid option!')

    df = pd.read_csv('heart.csv')
    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]
    pd.set_option('mode.chained_assignment', None)

    for col in X.columns:
        X[col] = normalizer(X[col], norm_method)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print("KNN:")
    knn_model = KNN()
    knn_model.fit(X_train, y_train)
    pred = knn_model.predict(X_test)
    stop = time()
    print('Accuracy: '+str(knn_model.score(y_test, pred))+'%')
    print('Time Elapsed: %.2fs' %(stop-knn_model.start_time))

    print("\nNaive Bayes:")
    start = time()
    naivebayes_model = NaiveBayes()
    naivebayes_model.fit(X_train, y_train)
    pred = naivebayes_model.predict(X_test)
    stop = time()
    print('Accuracy: '+str(naivebayes_model.score(y_test, pred))+'%')
    print('Time Elapsed: %.2fs' %(stop-start))

if __name__ == '__main__':
    main()
