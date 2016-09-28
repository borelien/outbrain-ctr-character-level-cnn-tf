from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
from utils import truncated_normal

class Network(object):
    def __init__(self, alphabet_size):
        self.inputs = tf.placeholder(tf.float32, [None, alphabet_size, None, 1], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None], name='labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        self.variables = []
        self.scores = None
        self.layer_key = 0
        self.loss = 0.
        self.weight_decay = 0.

    def convolution(self, input, nOut, kH, kW, strides, padding='VALID', boolDecay=True, non_linearity=tf.nn.relu):
        self.layer_key += 1
        with tf.name_scope('Convolution_{0}'.format(self.layer_key)) as scope:
            nIn = input.get_shape().dims[-1].value
            stddev = math.sqrt(2.0 / (kH * kW * nIn))
            filterInitializer = tf.constant_initializer(truncated_normal(shape=(kH, kW , nIn, nOut), stddev=stddev, mean=0.))                
            kernel = tf.get_variable(initializer=filterInitializer, shape=(kH, kW , nIn, nOut), trainable=True, name='filter_{0}'.format(self.layer_key))
            conv = tf.nn.conv2d(input=input, filter=kernel, strides=strides, padding=padding)
            output_shape = [dim.value for dim in conv.get_shape()]
            output_shape[0] = -1

            biases = tf.get_variable(initializer=tf.constant_initializer(0.01 + np.zeros(nOut)), shape=nOut, trainable=True, name='biases_{0}'.format(self.layer_key))
            conv_biases = conv + biases

            if non_linearity:
                nonlin = non_linearity(conv_biases)
            else:
                nonlin = conv_biases

            if boolDecay:
                self.variables += [kernel]
                self.variables += [biases]
            print(nonlin)
        return nonlin

    def pooling(self, input, ksize, strides, padding='VALID', pooling='max'):
        if pooling == 'max':
            pool = tf.nn.max_pool
        elif pooling == 'ave':
            pool = tf.nn.ave_pool
        else:
            raise ValueError('unknown pooling type')

        with tf.name_scope('Pooling_{0}'.format(self.layer_key)) as scope:
            pool = pool(input, ksize=ksize, strides=strides, padding=padding)
        print(pool)
        return pool

    def deploy(self):
        conv0 = self.convolution(input=self.inputs, nOut=128, kH=self.inputs.get_shape().dims[1].value, kW=5, strides=[1, 1, 1, 1])
        pool0 = self.pooling(input=conv0, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], pooling='max')

        conv1 = self.convolution(input=pool0, nOut=128, kH=1, kW=3, strides=[1, 1, 1, 1])
        pool1 = self.pooling(input=conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], pooling='max')
        pool1 = tf.nn.dropout(pool1, self.dropout_keep_prob, name='Dropout_{0}'.format(self.layer_key))

        previous_shape = [dim.value or 22 for dim in pool1.get_shape().dims]
        fc2 = self.convolution(input=pool1, nOut=256, kH=previous_shape[1], kW=previous_shape[2], strides=[1, 1, 1, 1])
        fc2 = tf.nn.dropout(fc2, self.dropout_keep_prob, name='Dropout_{0}'.format(self.layer_key))

        previous_shape = [dim.value or 1 for dim in fc2.get_shape().dims]
        fc3 = self.convolution(input=fc2, nOut=256, kH=previous_shape[1], kW=previous_shape[2], strides=[1, 1, 1, 1])
        
        previous_shape = [dim.value or 1 for dim in fc3.get_shape().dims]
        fc4 = self.convolution(input=fc3, nOut=1, kH=previous_shape[1], kW=previous_shape[2], strides=[1, 1, 1, 1], non_linearity=None)
        
        fc4_hist_summary = tf.histogram_summary("fc4", fc4)
        self.scores = tf.squeeze(tf.reduce_mean(tf.sigmoid(fc4), 2), name='scores')
        print(self.scores)

    def add_loss(self):
        with tf.name_scope('loss'):
            self.loss += tf.reduce_mean(tf.square(self.scores - self.labels))

    def add_weight_decay(self):
        with tf.name_scope('weight_decay'):
            self.weight_decay = sum([tf.reduce_sum(tf.square(var)) for var in self.variables])
            print(self.weight_decay)

    def forward_all(self):
        self.deploy()
        self.add_loss()
        self.add_weight_decay()