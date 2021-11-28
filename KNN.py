import numpy as np
from sklearn.model_selection import train_test_split
from time import time

class KNN:
    def __init__(self, k=None):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.start_time = time()

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        if self.k is None:
            self._optimal_k()
            self.start_time = time()
    
    def _optimal_k(self):
        start = time()
        X_tr, X_valid, y_tr, y_valid = train_test_split(self.X_train, self.y_train, test_size=0.25)
        err = []
        for k in range(1, 16):
            print("Finding the optimal value of k: %d (out of 15)" %k, end='\r')
            knn = KNN(k)
            knn.fit(X_tr, y_tr)
            pred = knn.predict(X_valid)
            err.append((k, knn.score(y_valid, pred)))
        stop = time()
        self.k = sorted(err, key=lambda x: x[1], reverse=True)[0][0]
        print('Time elapsed for finding the optimal k is %.2fs' %(stop-start))
        print('The optimal k is %d' %self.k)
    
    def _distance(self, a, b):
        return np.sqrt(((a-b)**2).sum())

    def _decider(self, knn):
        votes = {}
        for n in knn:
            votes[n[1]] = votes.get(n[1], 0)+1
        return max(votes)

    def predict(self, X_test):
        pred = np.array([])
        for j in range(X_test.shape[0]):
            test_ins = X_test.iloc[j]
            distances = []
            for i in range(self.X_train.shape[0]):
                train_ins = self.X_train.iloc[i]
                d = self._distance(train_ins, test_ins)
                distances.append((d, self.y_train.iloc[i]))
            knn = sorted(distances, key=lambda x: x[0])[:self.k]
            pred = np.append(pred, self._decider(knn))
        return pred

    def score(self, y_true, y_test):
        return round(100.0 * sum(y_test == y_true)/len(y_true),3)
