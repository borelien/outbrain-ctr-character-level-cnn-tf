# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

This script is designed to take a GraphDef proto, a SaverDef proto, and a set of
variable values stored in a checkpoint file, and output a GraphDef with all of
the variable ops converted into const ops containing the values of the
variables.

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded where we may not have access to the
RestoreTensor ops and file loading calls that they rely on.

An example of command-line usage is:
bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=some_graph_def.pb \
--input_checkpoint=model.ckpt-8361242 \
--output_graph=/tmp/frozen_graph.pb --output_node_names=softmax

You can also look at freeze_graph_test.py for an example of how to use it.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format

import sys
#sys.path.insert(1,'/home/matthieu/external/tensorflow/tensorflow')
from tensorflow.python.client import graph_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("input_graph", "",
                           """TensorFlow 'GraphDef' file to load.""")
tf.app.flags.DEFINE_string("input_saver", "",
                           """TensorFlow saver file to load.""")
tf.app.flags.DEFINE_string("input_checkpoint", "",
                           """TensorFlow variables file to load.""")
tf.app.flags.DEFINE_string("output_graph", "",
                           """Output 'GraphDef' file name.""")
tf.app.flags.DEFINE_boolean("input_binary", False,
                            """Whether the input files are in binary format.""")
tf.app.flags.DEFINE_string("output_node_names", "",
                           """The name of the output nodes, comma separated.""")
tf.app.flags.DEFINE_string("restore_op_name", "save/restore_all",
                           """The name of the master restore operator.""")
tf.app.flags.DEFINE_string("filename_tensor_name", "save/Const:0",
                           """The name of the tensor holding the save path.""")
tf.app.flags.DEFINE_boolean("clear_devices", True,
                            """Whether to remove device specifications.""")



def convert_variables_to_constants(sess, input_graph_def, output_node_names):
  """Replaces all the variables in a graph with constants of the same values.

  If you have a trained graph containing Variable ops, it can be convenient to
  convert them all to Const ops holding the same values. This makes it possible
  to describe the network fully with a single GraphDef file, and allows the
  removal of a lot of ops related to loading and saving the variables.

  Args:
    sess: Active TensorFlow session containing the variables.
    input_graph_def: GraphDef object holding the network.
    output_node_names: List of name strings for the result nodes of the graph.

  Returns:
    GraphDef containing a simplified version of the original.
  """
  print('call convert_variables')
  found_variables = {}
  variable_name_list = []
  found_variables_list = []
  print('search nodes...')
  for i, node in enumerate(input_graph_def.node):
    # print('node %s' % node)
    if node.op == "Assign":
      variable_name_list.append(node.input[0])
      sys.stdout.write("\r%s" % "node: {0}/{1}".format(i+1, len(input_graph_def.node)))
      sys.stdout.flush()
  print('')
  print('{0} nodes founded'.format(len(variable_name_list)))
  print('evaluate nodes..')
  found_variables_list = sess.run([v+":0" for v in variable_name_list])
  print('insert values..')
  for i, v in enumerate(variable_name_list):
    found_variables[v] = found_variables_list[i]
    sys.stdout.write("\r%s" % "node: {0}/{1}".format(i+1, len(variable_name_list)))
    sys.stdout.flush()
  print('')
     


  # This graph only includes the nodes needed to evaluate the output nodes, and
  # removes unneeded nodes like those involved in saving and assignment.
  inference_graph = graph_util.extract_sub_graph(input_graph_def, output_node_names)


  output_graph_def = graph_pb2.GraphDef()
  how_many_converted = 0
  for input_node in inference_graph.node:
    output_node = graph_pb2.NodeDef()
    if input_node.name in found_variables:
      output_node.op = "Const"
      output_node.name = input_node.name
      dtype = input_node.attr["dtype"]
      data = found_variables[input_node.name]
      output_node.attr["dtype"].CopyFrom(dtype)
      output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
          tensor=tensor_util.make_tensor_proto(data,
                                               dtype=dtype.type,
                                               shape=data.shape)))
      how_many_converted += 1
    else:
      output_node.CopyFrom(input_node)
    output_graph_def.node.extend([output_node])
  print("Converted %d variables to const ops." % how_many_converted)
  return output_graph_def

def freeze_graph(input_graph, input_saver, input_binary, input_checkpoint,
                 output_node_names, restore_op_name, filename_tensor_name,
                 output_graph, clear_devices):
  """Converts all variables in a graph and checkpoint into constants."""

  if not tf.gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1

  if input_saver and not tf.gfile.Exists(input_saver):
    print("Input saver file '" + input_saver + "' does not exist!")
    return -1

  if not tf.gfile.Exists(input_checkpoint):
    print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
    return -1

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  input_graph_def = tf.GraphDef()
  mode = "rb" if input_binary else "r"
  with open(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ""
  _ = tf.import_graph_def(input_graph_def, name="")

  with tf.Session() as sess:
    if input_saver:
      with open(input_saver, mode) as f:
        saver_def = tf.train.SaverDef()
        if input_binary:
          saver_def.ParseFromString(f.read())
        else:
          text_format.Merge(f.read(), saver_def)
        saver = tf.train.Saver(saver_def=saver_def)
        saver.restore(sess, input_checkpoint)
    else:
      sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
    output_graph_def = convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","))

  with tf.gfile.FastGFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))

import os
import re

def iter_from_checkpoint(checkpoint_path):
    checkpoint_reg = re.compile('\S+-(\d+)')
    checkpoint_match = re.match(checkpoint_reg, checkpoint_path)
    if checkpoint_match:
        return int(checkpoint_match.group(1))
    else:
        return None

def last_iter(savedir):
    latest_checkpoint = tf.train.latest_checkpoint(savedir)
    if latest_checkpoint is None:
        raise IOError('Latest checkpoint was not found')
    last_iter = iter_from_checkpoint(latest_checkpoint)
    return last_iter, latest_checkpoint

def frozenGraph(fd_model, graph_name="graph.pb", checkpoint_name="checkpoint", iteration=None, feature_layer="scores", input_binary=True, bool_replace=True):
    input_graph = os.path.join(fd_model, graph_name)
    if iteration is not None:
      checkpoint_path = os.path.join(fd_model, 'iteration-{0}'.format(iteration))
    else:
      _, checkpoint_path = last_iter(fd_model)
    splited = os.path.splitext(input_graph)
    frozen_graph_path = ''.join([splited[0],"_frozen", splited[1]])
    if not os.path.exists(frozen_graph_path) or bool_replace:
        freeze_graph(input_graph=input_graph, input_checkpoint=checkpoint_path, output_graph=frozen_graph_path,
            input_binary=input_binary, output_node_names=feature_layer, input_saver='', 
            restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', clear_devices=True)
    try:
        print("")
    except:
        raise IOError("unable to parse the graph")
    return frozen_graph_path