# coding: utf-8
# based on ideas in https://github.com/dennybritz/cnn-text-classification-tf and ttps://github.com/scharmchi/char-level-cnn-tf/

from __future__ import print_function
import numpy as np
import json
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from scipy.integrate import quad

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\nêéèàôîïëçù€"
# ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

def truncated_normal(shape, stddev, mean=0.):
    rand_init = np.random.normal(loc=mean, scale=stddev, size=shape)
    inf_mask = rand_init < (mean - 2 * stddev)
    rand_init = rand_init * (1 - inf_mask) + inf_mask * (mean - 2 * stddev)
    sup_mask = rand_init > (mean + 2 * stddev)
    rand_init = rand_init * (1 - sup_mask) + sup_mask * (mean + 2 * stddev)
    return rand_init

def pad_sentence(char_seq, max_seq_length, padding_char=" "):
    len_char_seq = len(char_seq)
    if len_char_seq > max_seq_length:
        random_start = np.random.randint(len_char_seq - max_seq_length)
        char_seq = char_seq[random_start : random_start + max_seq_length]
        new_char_seq = char_seq
    else:
        num_padding = max_seq_length - len_char_seq 
        new_char_seq = list(char_seq) + [-1] * num_padding
    return new_char_seq

def string_to_int8_conversion(char_seq):
    x = np.array([ALPHABET.find(char) for char in char_seq], dtype=np.int8)
    return x

def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index, min_size=96, max_size=156):
    x_batch = char_seqs_indices[start_index:end_index]
    max_length_batch = min(max(min_size, max([len(x) for x in x_batch])), max_size)
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(ALPHABET), max_length_batch, 1])
    for example_i, seq in enumerate(x_batch):
        char_seq_indices = pad_sentence(char_seq=string_to_int8_conversion(seq), max_seq_length=max_length_batch)
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    if labels is not None:
        y_batch = labels[start_index:end_index]
        return x_batch_one_hot, y_batch
    else:
        return x_batch_one_hot

def batch_iter(x, y, batch_size, num_epochs):
    data_size = len(x)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x_shuffled = np.array(x)[shuffle_indices]
        y_shuffled = np.array(y)[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
            batch = list(zip(x_batch, y_batch))
            yield batch

def topk(sorted_like_preds, sorted_like_groundtrouth, step, res_dir):
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    length = len(sorted_like_preds)
    xs = range(length)
    ys = []
    print(sorted_like_preds)
    for x in xs:
        best_x = sorted_like_groundtrouth[-(x + 1):]
        best_x_preds = sorted_like_preds[-(x + 1):]
        dict_i = {}
        dict_j = {}
        for i, value_preds in enumerate(best_x_preds):
            for j, value_gt in enumerate(best_x):
                if value_preds == value_gt and not (i in dict_i or j in dict_j):
                    dict_i[i] = True
                    dict_j[j] = True

        nb_good = len(dict_i.keys())
        acc_x = float(nb_good) / (x + 1)
        ys.append(acc_x)
    res = quad(lambda x:ys[int(x)], 0, len(ys) - 1, limit=10)[0] / length
    plt.clf()
    plt.plot(xs, ys, label="{}".format(res))
    plt.xlabel('k')
    plt.ylabel('Top-k intersection')
    plt.ylim([0., 1.])
    plt.xlim([0., length])
    plt.title('Top-k intersection')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(res_dir, "topk_{0}.jpg".format(step)))
    return res

def rank2(y_preds, y_val, step, res_dir):
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    ratio = 2
    val_size = len(y_val)
    scores = []
    labels = []
    print(y_preds[:30])
    for i in range(val_size-1):
        for j in range(i + 1, val_size):
            if max(y_val[i], y_val[j]) / max(1e-5, min(y_val[i], y_val[j])) > ratio:
                labels.append((y_val[i] - y_val[j]) * (y_preds[i] - y_preds[j]) > 0)
                score = np.abs(y_preds[i] - y_preds[j])
                scores.append(score)
    mAP = metrics.average_precision_score(labels, scores)
    plt.clf()
    precision, recall, threshold = metrics.precision_recall_curve(labels, scores)
    plt.plot(recall, precision, label="{}".format(mAP))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0., 1.])
    plt.xlim([0., 1.])
    plt.title("Rank2 Average Precision Curves")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(res_dir, "rank2_{0}.jpg".format(step)))
    return mAP, np.mean(labels)