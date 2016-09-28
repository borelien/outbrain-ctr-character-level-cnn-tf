# coding: utf-8
# based on ideas in https://github.com/dennybritz/cnn-text-classification-tf and ttps://github.com/scharmchi/char-level-cnn-tf/

from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from utils import ALPHABET, get_batched_one_hot, batch_iter, rank2, topk
from network import Network

timestamp = str(int(time.time()))

dirname, filename = os.path.split(os.path.abspath(__file__))
GIT_DIR = '/'.join(dirname.split('/')[:-1])

RUN_DIR = os.path.join(GIT_DIR, "runs", timestamp)
os.mkdir(RUN_DIR)
DATA_DIR = os.path.join(GIT_DIR, "data")
RESULTS_DIR = os.path.join(RUN_DIR, "resultst")
CHECKPOINTS_DIR = os.path.abspath(os.path.join(RUN_DIR, "checkpoints"))
checkpoint_prefix = os.path.join(CHECKPOINTS_DIR, "iteration")
if not os.path.exists(CHECKPOINTS_DIR):
    os.mkdir(CHECKPOINTS_DIR)


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("lambda_weight_decay", 0.000, "L2 regularizaion lambda (default: 0.0005)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 5000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on val set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow valice soft valice placement")
tf.flags.DEFINE_boolean("log_valice_placement", False, "Log placement of ops on valices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
with open(os.path.join(DATA_DIR,'text_train_purepeople.txt'), 'r') as f:
    x_train = f.read().split('\n')[:-1]

with open(os.path.join(DATA_DIR,'text_val_purepeople.txt'), 'r') as f:
    x_val = f.read().split('\n')[:-1]

with open(os.path.join(DATA_DIR,'ctr_train_purepeople.txt'), 'r') as f:
    y_train = [float(a) for a in f.read().split('\n')[:-1]]

with open(os.path.join(DATA_DIR,'ctr_val_purepeople.txt'), 'r') as f:
    y_val = [float(a) for a in f.read().split('\n')[:-1]]
x_val, y_val = get_batched_one_hot(x_val, y_val, 0, len(y_val))
print(len(x_train))
print(x_val.shape)


# Step function
# ==================================================

def train_step(x_batch, y_batch):
    feed_dict = {
      network.inputs: x_batch,
      network.labels: y_batch,
      network.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss = sess.run(
        [train_op, global_step, train_summary_op, network.loss],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}".format(time_str, step, loss))
    train_summary_writer.add_summary(summaries, step)

def val_step(x_val, y_val, step, writer=None):
    feed_dict = {
      network.inputs: x_val,
      network.labels: y_val,
      network.dropout_keep_prob: 1.0
    }
    step, summaries, loss, scores = sess.run(
        [global_step, val_summary_op, network.loss, network.scores],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    if step % FLAGS.checkpoint_every == 0:
        prec_rec = topk(sorted_like_preds=list(np.array(y_val)[np.argsort(scores)]), sorted_like_groundtrouth=list(np.sort(y_val)), step=step, res_dir=RESULTS_DIR)
    else:
        prec_rec = ''
    mAP, acc = rank2(scores, y_val, step=step, res_dir=RESULTS_DIR)
    print("{0}: step {1}, loss {2}, rank2 {3}, acc: {4}, prec_rec:{5}".format(time_str, step, loss, mAP, acc, prec_rec))
    if writer:
        writer.add_summary(summaries, step)


# Training
# ==================================================

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        network = Network(alphabet_size=len(ALPHABET))
        network.forward_all()

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(network.loss + FLAGS.lambda_weight_decay * network.weight_decay, global_step=global_step)

        loss_summary = tf.scalar_summary("loss", network.loss)
        train_summary_op = tf.merge_summary([loss_summary])
        train_summary_writer = tf.train.SummaryWriter(os.path.join(RUN_DIR, "summaries", "train"), sess.graph)
        val_summary_op = tf.merge_summary([loss_summary])
        val_summary_writer = tf.train.SummaryWriter(os.path.join(RUN_DIR, "summaries", "val"), sess.graph)
        
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        graph_path = os.path.join(CHECKPOINTS_DIR, "graph.pb")
        if not os.path.exists(graph_path):
            tf.train.write_graph(graph_def=sess.graph_def, logdir=CHECKPOINTS_DIR, name="graph.pb", as_text=False)

        batches = batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                val_step(x_val, y_val, current_step, writer=val_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))