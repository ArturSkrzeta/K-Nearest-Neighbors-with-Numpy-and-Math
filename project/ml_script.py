import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn import Knn

species = ['I. setosa', 'II. versicolor', 'III. virginica']

def plot_chart(df):

    colours = ['red', 'green', 'blue']

    for i in range(len(species)):
        species_df = df[df['species'] == i]
        plt.scatter(
            species_df['sepal length (cm)'],
            species_df['petal length (cm)'],
            color=colours[i],
            alpha=0.5,
            label=species[i]
        )

    plt.xlabel('feature: petal length')
    plt.ylabel('feature: sepal length')
    plt.title('Iris dataset: petal length vs sepal length')
    plt.legend(loc='lower right')
    plt.show()

def calculate_accuracy(predictions, labels):
    acc = np.sum(predictions == labels) / len(labels)
    print('Accuracy: ' + str(acc))

def main():

    # getting data
    # returning set of features and set of labels
    # for each 4-elements set of features there is one label assigned
    # label is assgined based on characteristic resulting from features
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    X = iris_df.to_numpy()
    y = iris['target']
    iris_df['species'] = iris['target']
    # print(X)  # [[5.9 3.  4.2 1.5],...,[6.  2.2 4.  1. ],...,[6.1 2.9 4.7 1.4]]
    # print(y)  # [0,0,0,....,1,1,1,....,2,2,2,...]
    plot_chart(iris_df)

    # splitting data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = Knn(k=3)
    clf.fit(X_train, y_train) # fitting model with features and corresponding labels
    predictions = clf.predict(X_test)

    print('Test samples shape: ' + str(X_test.shape))          # 120 features
    print(X_test)
    print('')
    print('Predictions shape: ' + str(predictions.shape))     # 30 lables
    print(predictions)

    print('')
    calculate_accuracy(predictions, y_test) # comparing predicitons outcome with y_test

    new_features = np.asarray([[6.2,2.8,5.7,1.8]])
    predicted_label = clf.predict(np.asarray(new_features))

    print('')
    print('New Features: ' + str(new_features))
    print('Predicted label: ' + str(predicted_label))
    print('Predicted speices: ' + str(species[int(predicted_label[0])]))

if __name__ == "__main__":
    main()
