import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/Nick/Git/Machinelearning')

# Define our dataset
dataset = pd.read_csv('data/first1500.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, 3].values

                
#Fill in the gaps of the data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Categorize city names
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
X = pd.get_dummies(X, columns='PHYSICAL_CITY')

#Split the dataset into our train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print("onto post scaling results")
#y_train = y_train.reshape((y_train.shape[0], 1))


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


# Set our learning rate and training epoch
learningrate = 0.01
trainingepochs = 100
display_step = 50

# Generate a random number
rng = numpy.random

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
n_samples = X_train.shape[0]
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch a tensorflow session to
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Execute everything
for epoch in range(trainingepochs):
    for (x, y) in zip(X_train, y_train):
        sess.run(optimizer, feed_dict={X: x, Y: y})
    w_val = sess.run(W)

sess.close()

#plot the data
plt.scatter(X_test[:-1], y_test, color='red')
plt.plot(X_train, optimizer.predict(X_train), color = 'blue')
plt.show()