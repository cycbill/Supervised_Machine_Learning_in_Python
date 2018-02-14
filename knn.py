## knn.py

# This is an example of a K-Nearest Neighbors classifier on MNIST data.
# We try k=1...5 to show how we might choose the best k.

import numpy as np
from sortedcontainers import SortedList
from util import get_data
from datetime import datatime

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):   # i is index, x is the value.
            sl = SortedList(load=self.k)
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))

            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.iteritems():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v 
            y[i] = max_votes_class
        return y

    def scores(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)



if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    for k in (1,2,3,4,5):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        print("Train accuracy:", knn.score(Xtrain, Ytrain))
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        print("Test accuracy:", knn.score(Xtest, Ytest))
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
