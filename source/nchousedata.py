import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/Nick/Git/Machinelearning')

learningrate = 0.01
trainingepochs = 100

# Define our dataset
dataset = pd.read_csv('first1500.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Clean up the data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Categorize city names
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Split the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

'''#Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

print(X_train).size
print(y_train).size
'''

#placeholder for tensor objects
X = tf.placeholder("float")
y = tf.placeholder("float")

# define cost function
costfunc = (tf.square(y-y_train))

#train the machine
train_op = tf.train.GradientDescentOptimizer(learningrate).minimize(costfunc)

#Create Tensorflow session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# execute
for epoch in range(trainingEpochs):
    for (x, y) in zip(xTrain, yTrain):
        sess.run(train_op, feed_dict={X: x, Y: y})
    w_val = sess.run(w)
    
#plot
plt.scatter(X_train, y_train)
y_learned = X_train*w_val
plt.plot(X_train, y_learned, 'r')
plt.show()

'''# Plot
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()'''
