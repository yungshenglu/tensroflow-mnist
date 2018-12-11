#!/usr/bin/python
#
# Exmaple program for TensorFlow
# Author: David Lu (https://github.com/yungshenglu)

import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))