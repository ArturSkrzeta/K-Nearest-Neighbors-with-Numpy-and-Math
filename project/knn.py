import numpy as np
from collections import Counter

# function for each corresponding sets of features in both X_test and X_train.
def distance_function(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

                # sqrt (
                    # sum (
                        # (f-t1 - f1) ** 2
                        # (f-t2 - f2) ** 2
                        # (f-t3 - f3) ** 2
                        # (f-t4 - f4) ** 2
                        # (f-t5 - f5) ** 2
                    #   )
                #   )

class Knn:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train    # X_train = [[f1,f2,f3,f4],[f1,f2,f3,f4],...] - list of features list
        self.y_train = y_train    # y_train = [l1,l2,...] - list of labels

    def predict(self, X_test):
        y_test = [self._predict(x) for x in X_test] # X_test = [[f-t1,f-t2,f-t3,f-t4], [f-t1,f-t2,f-t3,f-t4],...]
        return np.array(y_test)

    def _predict(self, x):
                            #   x = [f-t1,f-t2,f-t3,f-t4]
                            #   x_train = [f1,f2,f3,f4]
        distances = [distance_function(x, x_train) for x_train in self.X_train] # taking one set of features from X_test and one set of features from X_train into distance_function
        k_indicies = np.argsort(distances)[:self.k] # from 0 to k - indicies of the k nearest samples - sorted ascending - taking the smallest distances
        k_nearest_labels = [self.y_train[i] for i in k_indicies] # taking labels for the smallest distances
        most_common_labels = Counter(k_nearest_labels).most_common(1) # return list of tuples [(most_common, times_it_appears)]
        return most_common_labels[0][0] # retruning most_common as first item from first tuple in list
