
__author__ = 'Debjit Paul'
"""
This class uses logistic regression on Iris Dataset and Visiualize the data before and after prediction 
"""
#!/usr/bin/env python3
# Import data and modules
import pandas as pd
import numpy as np
from sklearn import datasets
import pylab
import matplotlib.pyplot as plt 
pylab.rcParams['figure.figsize'] = (10, 6)
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets

import warnings

def main():
    ## Load the iris data 
    iris = datasets.load_iris()
    ##
    X_train, X_test, y_train, y_test,iris_df, X,y=get_data(iris)
    X_train_std,X_test_std=scale_data(X_train,X_test,iris_df)
    show_data(y_test,X,y)
    classification=Logistic(X_train_std,y_train, X_test_std, y_test)
    classification.perform_svm(X_train_std, y_train, X_test_std, y_test)
    
def get_data(iris):
# Only petal length and petal width considered
    X = iris.data[:, [2, 3]]
    y = iris.target
    
# Place the iris data into a pandas dataframe
    iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])

# View the data
    print(iris_df.head())

# Print the classes of the dataset
    print('\n' + 'The classes in this data are ' + str(np.unique(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

    print('Training set are {} samples  and Test set are {} samples'.format(
    X_train.shape[0], X_test.shape[0]))
    print()
    return(X_train, X_test, y_train, y_test,iris_df, X,y)

def scale_data(X_train,X_test,iris_df):
     sc = StandardScaler()
     sc.fit(X_train)
     X_train_std = sc.transform(X_train)
     X_test_std = sc.transform(X_test)
     print('After standardizing our features,data looks like as follows:\n')
     print(pd.DataFrame(X_train_std, columns=iris_df.columns).head())
     return(X_train_std,X_test_std)
    
def show_data(y_test,X,y):
    ##There are 3 classes
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y_test))])
    for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   c=cmap(idx), marker=markers[idx], label=cl)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()

class Logistic(object):
   def __init__(self,X_train_std,y_train,X_test_std, y_test):
     self.X_train_std=X_train_std
     self.y_train=y_train
     self.X_test_std=X_test_std
     self.y_test=y_test
   def perform_svm(self,X_train_std,y_train,X_test_std, y_test):
      h = .02  # step size in the mesh

      logreg = linear_model.LogisticRegression(C=1e5)

      # we create an instance of Neighbours Classifier and fit the data.
      logreg.fit(X_train_std, y_train)
      print('The training accuracy is {:.2f}%'.format(logreg.score(X_train_std, y_train)*100))
      print('The test accuracy is {:.2f}%'.format(logreg.score(X_test_std, y_test)*100))
      X=X_test_std
      y=y_test
      resolution=0.01
      x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
      x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
      xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

      #Z = svm.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
      markers = ('s', 'x', 'o', '^', 'v')
      colors = ('red', 'blue', 'green', 'gray', 'cyan')
      cmap = ListedColormap(colors[:len(np.unique(y_test))])
      X=X_test_std
      y=y_test    
    # plot the decision surface
      x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
      x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
      xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

      Z = logreg.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
      Z = Z.reshape(xx1.shape)
      plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
      plt.xlim(xx1.min(), xx1.max())
      plt.ylim(xx2.min(), xx2.max())

      for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.5, c=cmap(idx),
                    marker=markers[idx], label=cl)
      plt.show()

if __name__=="__main__":
   main()

