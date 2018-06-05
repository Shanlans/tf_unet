# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:10:25 2018

@author: 17622
"""
import tensorflow as tf
import numpy as np

np.random.seed(1)
y = np.random.randint(0,2,size=(1,3,3,1))
y1 = 1- y 

y = np.squeeze(np.stack((y,y1),axis = 3),axis =4)

y_sum = np.sum(y,axis=3).reshape((1,3,3,1))
y = np.divide(y,y_sum)
class_weights = [1,10]
flat_labels = tf.cast(tf.reshape(y, [-1, 2]),'float32')

print(flat_labels.get_shape())

class_weights = tf.constant(np.array(class_weights, dtype=np.float32).reshape((1,2)))          
weight_map = tf.multiply(flat_labels, class_weights)

with tf.Session() as sess:
      w_img = sess.run(weight_map) 
      img = sess.run(flat_labels)
      print(w_img)
      print(img)