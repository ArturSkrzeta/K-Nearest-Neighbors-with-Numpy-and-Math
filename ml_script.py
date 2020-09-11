import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_iris_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

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

def calculate_accuracy(predictions, labels):
    acc = np.sum(predictions == labels) / len(labels)
    print('Accuracy: ' + str(acc))

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
        distances = [distance_function(x, x_train) for x_train in self.X_train]
        k_indicies = np.argsort(distances)[:self.k] # from 0 to k - indicies of the k nearest samples - sorted ascending
        k_nearest_labes = [self.y_train[i] for i in k_indicies] # taking labels for the smallest distances
        most_common_labels = Counter(k_nearest_labes).most_common(1) # return list of tuples [(most_common, times_it_appears)]
        return most_common_labels[0][0] # retruning most_common as first item from first tuple in list


def main():

    X, y = get_iris_data()

    # numpy sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = Knn(k=3)
    clf.fit(X_train, y_train) # fitting model with features and corresponding labels
    predictions = clf.predict(X_test)

    print('Test samples size: ' + str(X_test.size))          # 120 features
    print(X_test)
    print('')
    print('Predictions size: ' + str(predictions.size))     # 30 lables
    print(predictions)

    print('')
    calculate_accuracy(predictions, y_test) # comparing predicitons outcome with y_test

    new_features = np.asarray([[6.2,2.8,5.7,1.8]])
    predicted_label = clf.predict(np.asarray(new_features))

    print('')
    print('New Features: ' + str(new_features))
    print('Predicted lable: ' + str(predicted_label))

if __name__ == "__main__":
    main()
