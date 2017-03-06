#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:17:40 2017

@author: Nick
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/Nick/Git/Machinelearning/')

# Importing the dataset
dataset = pd.read_csv('first1500.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
                
#Fill in the gaps of the data so fields that have a zero value are normalized
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Muiltiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backwards Elimination
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]q
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()