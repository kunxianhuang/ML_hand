#!/bin/env python3
#-*- coding=utf-8 -*-

import numpy as np
from scipy.special import expit


#Linear Regression class
class LinearRegression(object):
    #initialize
    def __init__(self,epochs=500,eta=0.001, shuffle=True,random_state=None):
        np.random.seed(random_state)
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.w_initialized=False

    def fit(self, X, y):
        #X:array-like shape=[n_samples, n_features], trainging vectors
        #y:array-like shape=[n_samples], means true label value
        self.n_features=X.shape[1]
        if (self.w_initialized==False):
            self.w_=self._initialize_weights()
        self.error_=[]
        #loop for training
        for i_trai in range(self.epochs):
            #data shuffle
            if self.shuffle:
                X,y = self._shuffle(X,y)

            self.error_.append(self._get_error(X,y))
            self._update_weights(X,y)

            if i_trai % 100 ==0:
                print("Epopch {}, E_in is {}".format(i_trai+1,self.error_[-1]))

        return self.error_

    #initialize the weighting vector with random number (-1,1)
    #dimension of weighting vector is n_features+1 (w0....wd)
    def _initialize_weights(self):
        return np.random.uniform(-1.0,1.0, size=self.n_features+1)
    """
    def _sigmoid(self,z):
        return expit(z) #sigmoid function
    """
    #shuffle the data
    def _shuffle(self, X, y):
        r=np.random.permutation(len(y))
        return X[r],y[r]
    #update the weighting vector
    def _update_weights(self,X,y):
        n_samples=X.shape[0]
        wx_=self.net_input(X)
        zi=[]
        for wxi, yi in zip(wx_,y):
            zi.append(yi-wxi) #(y_true-y_pred)
        #change to array
        net_sig=np.array(zi)
        net_yx=[]
        net_y=[]
        for Xi, yi, sigi in zip(X,y, net_sig):
            net_yx.append(-2.0*Xi*sigi)
            net_y.append(-2.0*sigi)

        net_yx = np.array(net_yx)
        net_y  = np.array(net_y)
        self.w_[1:] += self.eta*(-1.0/n_samples)*np.sum(net_yx,axis=0)
        self.w_[0] += self.eta*(-1.0/n_samples)*np.sum(net_y)
        
    
    #obtain the Ein for each step
    def _get_error(self, X, y):
        n_samples=X.shape[0]
        wx_=self.net_input(X)
        Ein_p=[]
        for wxi, yi in zip(wx_,y):
            Ein_p.append(np.square(wxi-y))
            
        return (1.0/n_samples)*np.sum(np.array(Ein_p))

    #return wx
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    #return the prediction
    def predict(self, X):
        #return sigmoid function of wx
        z=self.net_input(X)
        return z
