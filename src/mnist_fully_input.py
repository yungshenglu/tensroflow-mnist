'''
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron) implementation with TensorFlow.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/).

LINKS       : MNIST Dataset (http://yann.lecun.com/exdb/mnist/)
PROJECT     : https://github.com/aymericdamien/TensorFlow-Examples
AUTHOR      : Aymeric Damien
CONTRIBUTOR : David Lu (https://github.com/yungshenglu)
'''
 
from __future__ import print_function
import tensorflow as tf
 
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
 
# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100
model_path = "/tmp/model.ckpt"
 
'''
Network Parameters
n_hidden_1  : 1st layer number of neurons
n_hidden_2  : 2nd layer number of neurons
num_input   : MNIST data input (image shape: 28*28)
num_classes : MNIST total classes (0-9 digits)
'''
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

# Insert the placeholder for a tensor that will be always fed
X = tf.placeholder("float", [None, num_input])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Evaluate model
# tf.argmax returns the index with the largest value across axes of a tensor
ans = tf.argmax(prediction, 1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
 
# Running a test dataset by loading the model saved earlier
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # Restore the model
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)
    # Calculate accuracy for MNIST test images
    print("Answer:", sess.run(ans, feed_dict={X: mnist.test.images}))