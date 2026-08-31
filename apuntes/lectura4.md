# Evaluación de modelos
> Introduction to Machine Learning with Python [Muller, 2019]

## Cross Validation
Cross-validation is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set. In crossvalidation, the data is instead split repeatedly and multiple models are trained. 

The most commonly used version of cross-validation is k-fold cross-validation, where k is a user-specified number, usually 5 or 10. When performing five-fold cross-validation, the data is first partitioned into five parts of (approximately) equal size, called folds.

Next, a sequence of models is trained. The first model is trained using the first fold as the test set, and the remaining folds (2–5) are used as the training set. The model is built using the data in folds 2–5, and then the accuracy is evaluated on fold 1. Then another model is built, this time using fold 2 as the test set and the data in folds 1,3, 4, and 5 as the training set. This process is repeated using folds 3, 4, and 5 as test sets.

For each of these five splits of the data into training and test sets, we compute the accuracy. In the end, we have collected five accuracy values. 

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)

print("Cross-validation scores: {}".format(scores))
# Cross-validation scores: [ 1. 0.967 0.933 0.9 1. ]

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("Cross-validation scores:\n{}".format(
cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
```

### Leave-one-out cross-validation
K=N.
~~~python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
~~~

### Shuffle-split cross-validation
Another, very flexible strategy for cross-validation is shuffle-split cross-validation. In
shuffle-split cross-validation, each split samples train_size many points for the
training set and test_size many (disjoint) point for the test set. This splitting is
repeated n_iter times. Figure 5-3 illustrates running four iterations of splitting a
dataset consisting of 10 points, with a training set of 5 points and test sets of 2 points
each (you can use integers for train_size and test_size to use absolute sizes for
these sets, or floating-point numbers to use fractions of the whole dataset).
~~~python
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)

print("Cross-validation scores:\n{}".format(scores))
~~~

### GroupKFold
To achieve this, we can use GroupKFold, which takes an array of groups as argument that we can use to indicate which person is in the image. The groups array here indicates groups in the data that should not be split when creating the training and test sets, and should not be confused with the class label.
This example of groups in the data is common in medical applications, where you
might have multiple samples from the same patient, but are interested in generalizing
to new patients. Similarly, in speech recognition, you might have multiple recordings
of the same speaker in your dataset, but are interested in recognizing speech of new
speakers

~~~python
from sklearn.model_selection import GroupKFold
# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)
# assume the first three samples belong to the same group,
# then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("Cross-validation scores:\n{}".format(scores))
~~~

## Searching Best Hyperparameters

### The danget of overfitting
There is a danger of overfitting the parameters: we tried many different parameters and selected the one with best accuracy on the test set, but this accuracy won’t necessarily carry over to new data. Because we used the test data to adjust the parameters, we can no longer use it to assess how good the model is. 
Therefore we use Training Set | Validation Set | Test Set

### Grid search
Test every configuration.

#### GridSearch-CrossVal
~~~python
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
 for C in [0.001, 0.01, 0.1, 1, 10, 100]:
 # for each combination of parameters,
 # train an SVC
 svm = SVC(gamma=gamma, C=C)
 # perform cross-validation
 scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
 # compute mean cross-validation accuracy
 score = np.mean(scores)
 # if we got a better score, store the score and parameters
 if score > best_score:
 best_score = score
 best_parameters = {'C': C, 'gamma': gamma}

# rebuild a model on the combined training and validation set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)

## o tambien
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# or a list of grids
param_grid = [{'kernel': ['rbf'],
 'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
 {'kernel': ['linear'],
 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)
~~~

Fitting the GridSearchCV object not only searches for the best parameters, but also automatically fits a new model on the whole training dataset with the parameters that yielded the best cross-validation performance.

Best cross-val score != Test set score for winning configuration

> You can access the model with the best parameters trained on the whole training set using the best_estimator_attribute.
> The results of a grid search can be found in the cv_results_ attribute, which is a dictionary storing all aspects of the search. 
