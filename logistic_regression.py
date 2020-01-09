#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:25:31 2019

@author: lussier
"""

import pandas as pd
import itertools
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFE

#designate input file
input_file = "MLpain_old_vol_icvcontrol.csv"

#pandas read input csv
dataset = pd.read_csv(input_file, header = 0,  sep=',')

#select data
#X = dataset.iloc[:, 103:]  #select column through end, predictors
X = dataset.iloc[:, 29:]  #select column through end, predictors
y = dataset.iloc[:, 17]   #select column, target

#shuffle the data and split the sample into training and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=.8, test_size=.2, stratify = y, shuffle=True)

#standarize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression(penalty='l1', C=1e4, solver='liblinear', multi_class='auto')

#train model
logreg.fit(X_train, y_train)

#acc = logreg.score(X_test, y_test)
acc = logreg.score(X_train, y_train)
print("Accuracy: %.4f" % acc)

# predict the training data based on the model
y_pred = logreg.predict(X_train)

#print classification report
report = classification_report(y_train, y_pred)
print(report)

# get a table to help us break down these scores
cm = confusion_matrix(y_true=y_train, y_pred = y_pred) 
print(cm)

#To retrieve the intercept:
print(logreg.intercept_)

#For retrieving the slope:
print(logreg.coef_)


# cross-validation
y_pred = cross_val_predict(logreg, X_train, y_train, 
                           groups=y_train, cv=10)

# Evaluate a score for each cross-validation fold
acc = cross_val_score(logreg, X_train, y_train, 
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
logreg.fit(X_train, y_train) # fit to training data

y_pred = logreg.predict(X_test) # classify pain group using testing data
acc = logreg.score(X_test, y_test) # get accuracy
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
        


model = SelectFromModel(logreg, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

selector = RFE(logreg, 1)
selector = selector.fit(X_train, y_train)
selector.support_ 
order = selector.ranking_
order
print(order)
