# coding: utf-8

from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import get_batched_one_hot
from network import Network
from frozen_graph import frozenGraph
import os
from tensorflow.python.platform import gfile
import argparse

dirname, filename = os.path.split(os.path.abspath(__file__))
GIT_DIR = '/'.join(dirname.split('/')[:-1])
DATA_DIR = os.path.join(GIT_DIR, "data")
RUNS_DIR = os.path.join(GIT_DIR, "runs")

def main(titles, run_id, iteration):

    RUN_DIR = os.path.join(RUNS_DIR, run_id, "checkpoints")
    if not os.path.exists(RUN_DIR):
        raise ValueError("Incorrect RUN_DIR: {0} doesn't exists".format(RUN_DIR))
    
    frozen_graph_path = frozenGraph(os.path.join(RUNS_DIR, run_id, "checkpoints"), iteration=iteration, bool_replace=False)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config_proto)
    sess.graph.as_default()
    
    with gfile.FastGFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    tensors = tf.import_graph_def(graph_def, return_elements=["inputs:0", "dropout_keep_prob:0", "scores:0"])

    feed_dict = {
      tensors[0]: get_batched_one_hot(titles, None, 0, len(titles)),
      tensors[1]: 1.0
    }

    res = sess.run([tensors[2]], feed_dict)
    scores = res[0]
    sortind = np.argsort([-s for s in scores])

    for i, s in enumerate(sortind):
        print(i+1, titles[s], scores[s])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_id', type=str)
    parser.add_argument('-i', '--iteration', type=int, default=None)
    args = parser.parse_args()

    run_id = args.run_id
    iteration = args.iteration

    with open(os.path.join(DATA_DIR,'text_val_purepeople.txt'), 'r') as f:
        titles = f.read().split('\n')[:-1]

    main(
            titles=titles,
            run_id=run_id,
            iteration=iteration,
        )