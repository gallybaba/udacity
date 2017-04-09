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
import os
np.random.seed(40)

def build_training_feed(batch_size, dataset, labels):
    data_rows = dataset.shape[0]
    label_rows = labels.shape[0]
    if dataset.shape[0] < batch_size:
        raise("data ", data_rows, " needs more than batch_size ", batch_size, " cannot train")
    if data_rows != label_rows:
        raise("data and labels must be of same size: ", data_rows, label_rows)

    random_index_start = np.random.choice(np.arange(data_rows - batch_size), 1)
    tf.summary.scalar("RandomIndexStart", random_index_start)
    random_index_end = random_index_start + batch_size

    return {X: dataset[random_index_start: random_index_end], y: \
            labels[random_index_start: random_index_end], batch_start_index : random_index_start}


def build_training_set_placeholders(batch_size, input_features):
    ### input
    X = tf.placeholder(tf.float32, shape=(None, input_features), name = "X")
    y = tf.placeholder(tf.int32, shape=(None), name = "y")
    return X, y

def build_simple_net(X, input_features, num_hidden1, num_hidden2):

    ### hidden1
    with tf.name_scope("hidden1"):
        h1w = tf.Variable(tf.truncated_normal([input_features, num_hidden1],\
                          stddev = 1.0 / math.sqrt(float(input_features))), name="weights")
        h1b = tf.Variable(tf.zeros(num_hidden1), name="biases")
        hidden1 = tf.nn.relu(tf.matmul(X, h1w) + h1b)
        tf.summary.histogram("weights", h1w)
        tf.summary.histogram("biases", h1b)

    with tf.name_scope("hidden2"):
        h2w = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2],\
                                             stddev = 1.0 / math.sqrt(float(num_hidden1))), name="weights")
        h2b = tf.Variable(tf.zeros(num_hidden2), name = "biases")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, h2w) + h2b)
        tf.summary.histogram("weights", h2w)
        tf.summary.histogram("biases", h2b)

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

def build_loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits\
                , labels = labels, name = "xentropy")
    tf.summary.histogram("xentropy", cross_entropy)
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
    tf.summary.scalar("loss", loss)
    return loss

def build_train(loss, learning_rate, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tf.summary.scalar("LearningRate", learning_rate)
    train_op = optimizer.minimize(loss, global_step = global_step)
    tf.summary.scalar("GlobalStep", global_step)
    return train_op

def build_eval(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    accurate = tf.reduce_sum(tf.cast(correct, tf.int32))
    tf.summary.scalar("Accurate", accurate)
    return accurate


def run_simple(hlearning_rate, batch_size, std_train_data, std_validate_data, std_test_data\
    , std_train_label, std_validate_label, std_test_label, test_len):

    with tf.Graph().as_default():
        ## global step
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

        #print("validation data ", std_validate_data.shape, " validation label ", std_validate_label.shape)
        #print("test data ", std_test_data.shape, " test label ", std_test_label.shape)

        logits = build_simple_net(X, input_features, 10, 5)
        #print("logits", logits)
        loss = build_loss(logits, y)
        #print("loss", loss)
        trop = build_train(loss, learning_rate, global_step)
        #print("Training Op", trop)
        accurate = build_eval(logits, y)
        #print("Accuracy", accurate)
        #print("Initializing Global Variables")
        init = tf.global_variables_initializer()
        #print("Creating Saver")
        saver = tf.train.Saver()
        #print("Creating Session")
        sess = tf.Session()
        #print("Creating Summary Writer")
        summary_writer = tf.summary.FileWriter("d:\\temp\\simple", sess.graph)
        #print("Running Init")
        sess.run(init)
        num_steps = 20000
        tf.summary.scalar("NumberOfSteps", num_steps)
        summary = tf.summary.merge_all()
        for step in np.arange(num_steps):
            feed_dict = build_training_feed(batch_size, std_train_data, std_train_label)
            op, step_loss = sess.run([trop, loss], feed_dict = feed_dict)
            if step % 100 == 0:
                #print('step %d has loss %f ' % (step, step_loss) )
                summary_str = sess.run(summary, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                checkpoint_file = os.path.join("D:\\temp\\simple", 'modelsimple.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                prediction = sess.run(accurate, feed_dict = feed_dict)
                train_accuracy = tf.cast(prediction, tf.float32) / batch_size
                valid_feed_dict = build_training_feed(batch_size, std_validate_data, std_validate_label)
                valid_prediction = accurate.eval(session = sess, feed_dict = valid_feed_dict)
                #print("Train Prediction %0.2f Validation Prediction %0.2f" % (prediction, valid_prediction))
                valid_accuracy = tf.cast(valid_prediction, tf.float32) / batch_size
                #print("Train Accuracy at step %d is %0.2f and Validate accuracy is %0.2f batch start %d"  % \
                #      (step, sess.run(train_accuracy)\
                #    , sess.run(valid_accuracy), sess.run(batch_start_index, feed_dict = feed_dict)))

        total_test_accuracy = 0
        for test_steps in range(test_len):
            test_feed_dict = build_training_feed(HYPER_BATCH_SIZE, std_test_data, std_test_label)
            test_prediction = sess.run(accurate, feed_dict = test_feed_dict)
            test_accuracy = tf.cast(test_prediction, tf.float32) / batch_size
            total_test_accuracy += sess.run(test_accuracy)

        print("Test Accuracy %0.2f" % (total_test_accuracy / test_len))