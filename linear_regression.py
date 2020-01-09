#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:58:55 2019

@author: desir
"""

import pandas as pd
import itertools
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#designate input file
input_file = "MLpain_old_vol_icvcontrol.csv"

#pandas read input csv
dataset = pd.read_csv(input_file, header = 0,  sep=',')

#select data
#X = dataset.iloc[:, 103:]  #select column through end, predictors
X = dataset.iloc[:, 29:]  #select column through end, predictors
y = dataset.iloc[:, 17]   #select column, target

#shuffle the data and split the sample into training and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=.9, test_size=.1, stratify = y, shuffle=True)

#standarize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
#          probability=False, tol=0.001, class_weight=None, verbose=False, 
#          max_iter=-1, decision_function_shape='ovo', random_state=None)

clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', 
                fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, 
                max_iter=1000)


clf.fit(X_train, y_train)

#train model
clf.fit(X_train, y_train)
acc = clf.score(X_train, y_train)
print("Accuracy: %.4f" % acc)

# predict the training data based on the model
y_pred = clf.predict(X_train)

#print classification report
report = classification_report(y_train, y_pred)
print(report)

# get a table to help us break down these scores
cm = confusion_matrix(y_true=y_train, y_pred = y_pred) 
print(cm)

# cross-validation
y_pred = cross_val_predict(clf, X_train, y_train, 
                           groups=y_train, cv=10)

# Evaluate a score for each cross-validation fold
acc = cross_val_score(clf, X_train, y_train, 
                     groups=y_train, cv=10)

for i in range(10):
    print('Fold %s -- Acc = %s'%(i, acc[i]))

# get scores
overall_acc = accuracy_score(y_pred = y_pred, y_true = y_train)
overall_cr = classification_report(y_pred = y_pred, y_true = y_train)
overall_cm = confusion_matrix(y_pred = y_pred, y_true = y_train)
print('Accuracy:',overall_acc)
print(overall_cr)

print('Confusion matrix:')
print(overall_cm)

# plot
thresh = overall_cm.max() / 2
cmdf = DataFrame(overall_cm, index = ['NoPain','Pain'], columns = ['NoPain','Pain'])
sns.heatmap(cmdf, cmap='copper')
plt.xlabel('Predicted')
plt.ylabel('Observed')
for i, j in itertools.product(range(overall_cm.shape[0]), range(overall_cm.shape[1])):
        plt.text(j+0.5, i+0.5, format(overall_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white")


#test model
clf.fit(X_train, y_train) # fit to training data

y_pred = clf.predict(X_test) # classify pain group using testing data
acc = clf.score(X_test, y_test) # get accuracy
cr = classification_report(y_pred=y_pred, y_true=y_test) # get prec., recall & f1
cm = confusion_matrix(y_pred=y_pred, y_true=y_test) # get confusion matrix

# print results
print('accuracy =', acc)
print(cr)

print('confusion matrix:')
print(cm)

## plot results
thresh = cm.max() / 2
cmdf = DataFrame(cm, index = ['NoPain','Pain'], columns = ['NoPain','Pain'])
sns.heatmap(cmdf, cmap='RdBu_r')
plt.xlabel('Predicted')
plt.ylabel('Observed')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j+0.5, i+0.5, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white")
