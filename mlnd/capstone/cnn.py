from tensorflow.contrib.layers import flatten
import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import numpy.core.defchararray as npc
from IPython.display import display
import matplotlib
from matplotlib import pyplot as plt
from tensorflow import summary
from sklearn import preprocessing
from simple import build_training_feed
from simple import build_eval, build_loss, build_train, build_training_feed, build_training_set_placeholders


def build_cnn_net(CX, input_features, num_hidden1, num_hidden2):

    #### cnn1
    with tf.name_scope("cnn1"):
        #### read one sampple at a time, 8 rows (short term data) and 8 columns (half the features)
        #### map features 15. 15 convolutions of 1*8*8
        #print("input to cnn network: ", CX)
        cnnw1 = tf.Variable(tf.truncated_normal(shape=(1, 8, 15, 15),\
                                               stddev = 1.0 / math.sqrt(64.0))\
                            , name = "weights")
        #print("cnn1 weights: ", cnnw1)
        cnnb1 = tf.Variable(tf.zeros(15))
        #print("cnn1 biases: ", cnnb1)
        conv1 = tf.nn.conv2d(CX, cnnw1, strides=[1,1,1,1], padding='VALID') +  cnnb1
        #print("cnn with stride 1: ", conv1)
        conv1 = tf.nn.relu(conv1)
        #print("cnn after relu: ", conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #print("conv1 after pool:", conv1)
        ### 1, 8, 15, 15 will turn into 1, 4, 4, 15 due to 1, 2, 2, 1 pooling
        RX = tf.reshape(conv1, shape=[-1,15])
        #print("RX after flatten: ", RX)
        ### hidden1
        with tf.name_scope("hidden1"):
            h1w = tf.Variable(tf.truncated_normal([input_features, num_hidden1],\
                          stddev = 1.0 / math.sqrt(float(input_features))), name="weights")
            h1b = tf.Variable(tf.zeros(num_hidden1), name="biases")
            hidden1 = tf.nn.relu(tf.matmul(RX, h1w) + h1b)
            tf.summary.histogram("weights", h1w)
            tf.summary.histogram("biases", h1b)

        with tf.name_scope("hidden2"):
            h2w = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2],\
                                             stddev = 1.0 / math.sqrt(float(num_hidden1))), name="weights")
            h2b = tf.Variable(tf.zeros(num_hidden2), name = "biases")
            hidden2 = tf.nn.relu(tf.matmul(hidden1, h2w) + h2b)
            tf.summary.histogram("weights", h2w)
            tf.summary.histogram("biases", h2b)

        with tf.name_scope("dropout"):
            hidden2 = tf.nn.dropout(hidden2, 0.75)
            #print("hidden2 after dropout: ", hidden2)

        with tf.name_scope("softmax"):
            smw = tf.Variable(tf.truncated_normal([num_hidden2, 2], \
                          stddev = 1.0 / math.sqrt(float(num_hidden2))), name = "weights")
            smb = tf.Variable(tf.zeros(2), name = "biases")
            logits = tf.matmul(hidden2, smw) + smb
            tf.summary.histogram("weights", smw)
            tf.summary.histogram("biases", smb)

        with tf.name_scope("building"):
            tf.summary.scalar("InputFeatures", input_features)
            tf.summary.scalar("Hidden1",num_hidden1)
            tf.summary.scalar("Hidden2", num_hidden2)

        return logits

def build_cnn_feed(batch_size, training_data, training_labels):
    feed_dict = build_training_feed(batch_size, training_data, training_labels)
    feed_dict[y] = feed_dict[y][0:16]
    return feed_dict

def run_cnn(hlearning_rate, batch_size, std_train_data, std_validate_data, std_test_data\
    , std_train_label, std_validate_label, std_test_label, test_len):
    with tf.Graph().as_default():
        # global step
        batch_start_index = tf.placeholder(tf.int32, shape=(1), name="BatchStartIndex")
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #print("Global Step", global_step)
        learning_rate = tf.train.exponential_decay(hlearning_rate, global_step,\
                batch_size, 0.95, staircase=True)
        #print("Learning Rate", learning_rate)
        tf.summary.scalar("GlobalStep", global_step)
        tf.summary.scalar("LearningRate", learning_rate)

        input_features = std_train_data.shape[1]
        #print("input_features %d" % (input_features))
        X, y = build_training_set_placeholders(batch_size, input_features)
        #print("X", X)
        #print("y", y)

        CX = tf.reshape(X, [1, 8, 15, 15])
        #print("CX: ", CX)
        #print("validation data ", std_validate_data.shape, " validation label ", std_validate_label.shape)
        #print("test data ", std_test_data.shape, " test label ", std_test_label.shape)
        logits = build_cnn_net(CX, input_features, 10, 15)
        #print("CNN Logits: ", logits)
        loss = build_loss(logits, y)
        #print("CNN Loss: ", loss)
        trop = build_train(loss, learning_rate, global_step)
        #print("CNN Training Op: ", trop)
        accuracy = build_eval(logits, y)
        #print("CNN Accuracy: ", accuracy)
        init = tf.global_variables_initializer()
        #print("CNN Init: ", init)
        saver = tf.train.Saver()
        #print("CNN Saver: ", saver)
        sess = tf.Session()
        #print("CNN Session: ", sess)
        sw = tf.summary.FileWriter("d:\\temp\\cnn", sess.graph)
        #print("Summary Writer: ", sw)
        sess.run(init)
        #print("sesson init run complete")
        num_steps = 20000
        tf.summary.scalar("NumberOfSteps", num_steps)
        summary = tf.summary.merge_all()
        for step in np.arange(num_steps):
            feed_dict = build_cnn_feed(batch_size, std_train_data, std_train_label)
            op, step_loss = sess.run([trop, loss], feed_dict = feed_dict)
            if step % 100 == 0:
                summary_str = sess.run(summary, feed_dict = feed_dict)
                sw.add_summary(summary_str, step)
                sw.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                checkpoint_file = os.path.join("D:\\temp\\cnn", "modelcnn.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                prediction = sess.run(accuracy, feed_dict=feed_dict)
                train_accuracy = tf.cast(prediction, tf.float32) / 16
                valid_feed_dict = build_cnn_feed(batch_size, std_validate_data, std_validate_label)
                valid_prediction = accuracy.eval(session = sess, feed_dict = valid_feed_dict)
                #print("Train Prediction %0.2f Validation Prediction %0.2f" % (prediction, valid_prediction))
                valid_accuracy = tf.cast(valid_prediction, tf.float32) / 16
                #print("Train Accuracy at step %d is %0.2f and Validate accuracy is %0.2f batch start %d"  % \
                #      (step, sess.run(train_accuracy)\
                #    , sess.run(valid_accuracy), sess.run(batch_start_index, feed_dict = feed_dict)))

        total_test_accuracy = 0
        for test_steps in range(test_len):
            test_feed_dict = build_cnn_feed(batch_size, std_test_data, std_test_label)
            test_prediction = sess.run(accuracy, feed_dict = test_feed_dict)
            test_accuracy = tf.cast(test_prediction, tf.float32) / 16
            total_test_accuracy += sess.run(test_accuracy)

        print("Test Accuracy %0.2f" % (total_test_accuracy / test_len))