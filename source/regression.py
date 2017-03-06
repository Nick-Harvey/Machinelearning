import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learningRate = 0.01
trainingEpochs = 100

# Return evenly spaced numbers over a specified interval
xTrain = np.linspace(-2, 1, 200)

#Return a random matrix with data from the standard normal distribution.
yTrain = 2 * xTrain + np.random.randn(*xTrain.shape) * 0.33

#Create a placeholder for a tensor that will be always fed.
X = tf.placeholder("float")
Y = tf.placeholder("float")

#define model and construct a linear model
def model (X, w):
    return tf.mul(X, w)

#Set model weights
w = tf.Variable(0.0, name="weights")

y_model = model(X, w)

#Define our cost function
costfunc = (tf.square(Y-y_model))

#Use gradient decent to fit line to the data
train_op = tf.train.GradientDescentOptimizer(learningRate).minimize(costfunc)

# Launch a tensorflow session to
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Execute everything
for epoch in range(trainingEpochs):
    for (x, y) in zip(xTrain, yTrain):
        sess.run(train_op, feed_dict={X: x, Y: y})
    w_val = sess.run(w)

sess.close()

#Plot the data
plt.scatter(xTrain, yTrain)
y_learned = xTrain*w_val
plt.plot(xTrain, y_learned, 'r')
plt.show()
