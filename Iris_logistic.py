#!/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from LinearRegression import LinearRegression


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris\
.data', header=None)
    
    print(df.head(5))
    #print (df.shape)

    #require the fourth column value (target)                                             
    y=df.iloc[0:100, 4].values
    #assign the label of sentosa 0, and virginica is 1                                   
    y=np.where(y=='Iris-setosa', 0, 1)
    #require first 100 row of column 1,3
    X=df.iloc[0:100, [0,2]].values

    X_std=np.zeros(X.shape)
    #standarize
    X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

    LR=LinearRegression(epochs=800, eta=0.05, shuffle=True, random_state=None)
    err_hist=LR.fit(X_std,y)

    plt.plot(err_hist,color='red',linestyle='-',label="LinearRegression")
    plt.xlabel('epoch')
    plt.ylabel('Ein')
    plt.legend(loc='upper right')
    #plt.ylim(0.3, 0.71)
    plt.show()


    plot_decision_regions(X_std, y, classifier=LR)
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    
    return

def plot_decision_regions(X,y, classifier,test_idx=None, resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z= classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z= Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl,1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx],label=cl)

    if test_idx:
        X_test, y_test= X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1],c='',
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test set')
    

if __name__=="__main__":
    main()
