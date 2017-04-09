from rnn import build_rnn_feed, build_rnn_net
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
from simple import build_eval, build_loss, build_train, build_training_feed, build_training_set_placeholders
from rnn import build_rnn_feed, build_rnn_net

base_checkpoint_path = "D:\\temp\\rnnopt"
checkpoints = np.arange(100, 8100, 100)
checkpoint_files = {}
best_accuracy = 0
best_accuracy_check_point = ''

def run_rnnopt(hlearning_rate, batch_size, std_train_data, std_validate_data, std_test_data\
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
        logits = build_rnn_net(CX, input_features, 10, 15)
        #print("RNN Logits: ", logits)
        loss = build_loss(logits, y)
        #print("RNN Loss: ", loss)
        trop = build_train(loss, learning_rate, global_step)
        #print("RNN Training Op: ", trop)
        accuracy = build_eval(logits, y)
        #print("RNN Accuracy: ", accuracy)
        init = tf.global_variables_initializer()
        #print("RNN Init: ", init)
        saver = tf.train.Saver()
        #print("RNN Saver: ", saver)
        sess = tf.Session()
        #print("RNN Session: ", sess)
        global base_checkpoint_path
        sw = tf.summary.FileWriter(base_checkpoint_path, sess.graph)
        #print("Summary Writer: ", sw)
        sess.run(init)
        #print("sesson init run complete")
        num_steps = 8000
        tf.summary.scalar("NumberOfSteps", num_steps)
        summary = tf.summary.merge_all()

        global best_accuracy
        global best_accuracy_check_point
        global checkpoint_files
        for step in np.arange(num_steps):
            feed_dict = build_rnn_feed(batch_size, std_train_data, std_train_label, 128, X, y, batch_start_index)
            #print("build rnn feed in rnnopt: ", feed_dict, " , train data: ", std_train_data.shape[0])
            op, step_loss = sess.run([trop, loss], feed_dict = feed_dict)
            if step % 100 == 0:
                #print("Running Step: ", str(step))
                summary_str = sess.run(summary, feed_dict = feed_dict)
                sw.add_summary(summary_str, step)
                sw.flush()
                file_name = "modelrnnopt_" + str(step) + ".ckpt"
                checkpoint_file = os.path.join(base_checkpoint_path, file_name)
                saver.save(sess, checkpoint_file, global_step=step)
                checkpoint_files[checkpoint_file] = 0
                test_feed_dict = build_rnn_feed(batch_size, std_test_data, std_test_label,128, X, y, batch_start_index)
                ##saver.restore(sess, cp)
                test_prediction = sess.run(accuracy, feed_dict = test_feed_dict)
                test_accuracy = tf.cast(test_prediction, tf.float32) / 128
                ta = sess.run(test_accuracy, feed_dict=test_feed_dict)
                #print("Test Accuracy: ", str(ta))
                checkpoint_files[checkpoint_file] = ta
                if best_accuracy < ta and step > 0:
                    best_accuracy = ta
                    best_accuracy_check_point = checkpoint_file
                    #print("Running test on checkpoint: ", checkpoint_file, " , best_accuracy: ", best_accuracy)

        print("Best Accuracy %0.2f on step %s"% (best_accuracy, best_accuracy_check_point))
    return checkpoint_files

