#!/bin/env python
#-*- coding=utf-8 -*-
#

import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

#change class label to integer by assifninf automatically
print('Class labels', np.unique(df_wine['Class label']))
#print first five lines of dataframe
print(df_wine.head())

for columns in df_wine.columns:
    print(columns)



#require the second to all column as X, and the first column as y
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

#split the train data and test data by 70%, 30% radomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#Decision tree below
#tree=DecisionTree(criterion='entropy', max_depth=6, random_state=None)
#tree.fit(X_train, y_train, df_wine.columns[1:])
#y_pred=tree.predict(X_test)

#Random Forest below
forest=RandomForest(criterion='gini', n_estimators=20, max_features='auto', max_depth=3, min_samples_split = 2,random_state=None)
forest.fit(X_train, y_train, df_wine.columns[1:])
y_pred=forest.predict(X_test)

#mis-classified number
print ("Misclassified samples/total test samples: %d/%d" %((y_test != y_pred).sum(), len(y_test)))
#print (y_test)
#print (y_pred)
#print "The first sample probability is ", tree.predict_proba(X_test[0,:])
