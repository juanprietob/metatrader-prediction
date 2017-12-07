from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())

def convolution(x, filter_shape, name, stride=1, relu=True, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            # in_time -> stride in time
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv1d( x, w_conv, stride=stride, padding="SAME", name='conv1_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(relu):
                conv_op = tf.nn.relu( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

def matmul(x, out_channels, name, relu=True, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        in_channels = x.get_shape().as_list()[-1]

    with tf.device(ps_device):
        w_matmul_name = 'w_' + name
        w_matmul = tf.get_variable(w_matmul_name, shape=[in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))

        print_tensor_shape( w_matmul, 'w_matmul shape')        

        b_matmul_name = 'b_' + name
        b_matmul = tf.get_variable(name=b_matmul_name, shape=[out_channels])        

    with tf.device(w_device):

        matmul_op = tf.nn.bias_add(tf.matmul(x, w_matmul), b_matmul)

        if(relu):
            matmul_op = tf.nn.relu(matmul_op)

        return matmul_op

def lstm(x, out_channels, name, keep_prob=1, activation=tf.tanh, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        in_channels = x.get_shape().as_list()[-1]

        with tf.device(ps_device):
            rnn_layers = [rnn.DropoutWrapper(rnn.LSTMCell(csize, activation=activation), output_keep_prob=keep_prob) for csize in out_channels]
            multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

            # initialize state cell
            # initial_state = tf.Variable(multi_rnn_cell.zero_state(shape[0], dtype=tf.float32))
            # initial_state = multi_rnn_cell.zero_state(x.get_shape()[0], dtype=tf.float32)

        with tf.device(w_device):
            outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=x,
                                               # initial_state=initial_state,
                                               dtype=tf.float32)

        return outputs, state


def inference_rnn(series, batch_size=1, keep_prob=1, training=False, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.name_scope('rnn'):

        shape = series.get_shape().as_list()

        print_tensor_shape(series, 'series')

        # conv1_op = convolution(series, [3, shape[-1], shape[-1]], "conv1_op", stride=1, ps_device=ps_device, w_device=w_device)

        # series = tf.concat([series, conv1_op], 2)

        # print_tensor_shape(series, 'series')

        # create  LSTMCells        
        outputs, state = lstm(series, [32], "lstm1", keep_prob=keep_prob, ps_device=ps_device, w_device=w_device)

        print_tensor_shape(outputs, "Out Rnn")

        #conv1_op = convolution(outputs, [shape[1], 32, shape[-1]], "conv1_op", stride=shape[1], ps_device=ps_device, w_device=w_device)
        matmul1_op = matmul(outputs[:,-1,:], shape[-1], "Matmul1", relu=False, ps_device=ps_device, w_device=w_device)
        print_tensor_shape( matmul1_op, 'Matmul1 shape')

        return matmul1_op

def evaluation(predictions, labels, name="accuracy"):
    #return tf.metrics.accuracy(predictions=predictions, labels=labels, name=name)
    #return tf.metrics.mean_absolute_error(predictions=predictions, labels=labels, name=name)
    return tf.metrics.root_mean_squared_error(predictions=predictions, labels=labels, name=name)
    

def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.summary.scalar('2learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    # optimizer = tf.train.RMSPropOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def loss(logits, labels):
    
    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')

    # loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    loss = tf.losses.mean_squared_error(predictions=logits, labels=labels)
    #loss = tf.losses.absolute_difference(labels, logits)



    return loss