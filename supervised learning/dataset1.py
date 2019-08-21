import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

data = pd.read_csv("./letter-recognition.data")

#The data could also be referred from following line
#data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data")

y=data.iloc[:,[0]]
x=data.iloc[:,[i for i in range(1,16)]]

le = preprocessing.LabelEncoder()
y=le.fit_transform(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.1, random_state=0)

# Decision tree for dataSet1
#gridSearchCV for DT's max_depth
n_maxDepth = [i for i in range(1,17)]
param_grid = [{'max_depth':n_maxDepth}]
clf = tree.DecisionTreeClassifier()
grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(x,y)
scores = grid.cv_results_
plt.plot(n_maxDepth, scores['mean_test_score'])
plt.ylabel('accuracy')
plt.xlabel('depth of DT')
plt.plot(grid.cv_results_['mean_train_score'])
plt.plot(n_maxDepth, scores['std_test_score'])
plt.ylabel('std of accuracy')
plt.xlabel('depth of DT')

#learning curve for DT(depth=15)
train=[]
test=[]
for i in range(1,95):
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1-i/100)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    train.append(accuracy_score(y_train, train_predict))
    test.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(train)),train, label="train data")
plt.plot(range(len(test)),test, label="test data")
plt.ylabel('accuracy')
plt.xlabel('fractional trainning data size')
plt.legend(loc="best")
plt.show()

clf = tree.DecisionTreeClassifier(max_depth = 15)
plot_learning_curve(clf, "DT with depth of 15", x, y, ylim=[0,1])

#Boosted DT classifier
#gridSearchCV for Boosting's n_estimators
n_estimator = [i*50 for i in range(1,6)]
param_grid = [{'n_estimators':n_estimator}]
clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1)) # change the max_depth = 1, 5, 10, 15 for parameter tuning
grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(x,y)
scores = grid.cv_results_

plt.plot(n_estimator,scores['mean_test_score'])
plt.ylabel('mean of accuracy')
plt.xlabel('number of estimator')
plt.title('DT with depth of 1')

#learning curve for boosting
train=[]
test=[]
for i in range(1,95):
    clf = clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10), n_estimators=50)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    train.append(accuracy_score(y_train, train_predict))
    test.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(train)),train, label="train data")
plt.plot(range(len(test)),test, label="test data")
plt.ylabel('accuracy')
plt.xlabel('fractional trainning data size')
plt.legend(loc="best")
plt.show()

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10), n_estimators=50)
plot_learning_curve(clf, "Adaboost with n_estimators 50", x, y)

#Neural network
train=[]
test=[]
fraction = [i/10 for i in range(1,10)]
for i in range(1,10):
    clf = MLPClassifier(hidden_layer_sizes=(26))   # change the hidden_layer_sizes = (26) or (16, 8) for parameter tuning
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1-i*10/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    train.append(accuracy_score(y_train, train_predict))
    test.append(accuracy_score(y_test, test_predict))
plt.plot(fraction,train, label="train data")
plt.plot(fraction,test, label="test data")
plt.ylabel('accuracy')
plt.xlabel('fractional trainning data size')
plt.legend(loc="best")
plt.show()

#SVC
train=[]
test=[]
fraction = [i/10 for i in range(1,10)]
for i in range(1,10):
    clf = svm.SVC(kernel="sigmoid") # change the kernel = ‘linear’, ‘rbf’ and ‘sigmoid’ for parameter tuning
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1-i*10/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    train.append(accuracy_score(y_train, train_predict))
    test.append(accuracy_score(y_test, test_predict))
plt.plot(fraction,train, label="train data")
plt.plot(fraction,test, label="test data")
plt.ylabel('accuracy')
plt.xlabel('fractional trainning data size')
plt.legend(loc="best")
plt.show()


#KNN
#gridSearchCV for KNN's k in range [1, 20]
k_range = [i for i in range(1,21)]
param_grid = [{'n_neighbors' : k_range}]
clf = KNeighborsClassifier()
grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(x,y)
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
plt.plot(k_range, means)
plt.ylabel('mean accuracy')
plt.xlabel('value of k')
plt.plot(k_range, stds)
plt.ylabel('std of accuracy')
plt.xlabel('value of k')

#learning carve for KNN
train=[]
test=[]
fraction = [i/10 for i in range(1,10)]
for i in range(1,10):
    clf = clf = KNeighborsClassifier(1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=1-i*10/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    train.append(accuracy_score(y_train, train_predict))
    test.append(accuracy_score(y_test, test_predict))
plt.plot(fraction,train, label="train data")
plt.plot(fraction,test, label="test data")
plt.ylabel('accuracy')
plt.xlabel('fractional trainning data size')
plt.legend(loc="best")
plt.show()

clf = clf = KNeighborsClassifier(1)
plot_learning_curve(clf, "KNN with k=1", x, y)


#learning curve function from sklearn tutorial

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt