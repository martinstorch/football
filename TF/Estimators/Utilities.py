# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:43:50 2018

@author: 811087
"""

from tensorflow.python.training import training  
from tensorflow.python.training import saver
import tensorflow as tf
import numpy as np

def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(checkpoint_dir))
#      return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0

def optimistic_restore_vars(model_checkpoint_path):
  reader = tf.train.NewCheckpointReader(model_checkpoint_path)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
  with tf.variable_scope('', reuse=True):
      for var_name, saved_var_name in var_names:
          curr_var = name2var[saved_var_name]
          var_shape = curr_var.get_shape().as_list()
          if var_shape == saved_shapes[saved_var_name]:
              restore_vars.append(curr_var)
  return restore_vars

def upgrade_estimator_model(model_dir, model, features, labels):
  print("upgrade_estimator_model")
  global_step = _load_global_step_from_checkpoint_dir(model_dir)
  print("Existing global step = {}".format(global_step))
  
  # Create a saver for writing training checkpoints.
  #saver = tf.train.Saver(max_to_keep=50)
  checkpoint = saver.latest_checkpoint(model_dir)
  print("Checkpoint = {}".format(checkpoint))
  tf.Variable(global_step, trainable=False, name='global_step')
  model._call_model_fn(features, labels, tf.estimator.ModeKeys.TRAIN, model.config)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  #    sv = tf.train.Saver(max_to_keep=50)  
  #    sv.restore(sess, checkpoint)
    reader = tf.train.NewCheckpointReader(checkpoint)
    # Verifies that the tensors exist.
    #self.assertTrue(reader.has_tensor("v0"))
    # Verifies get_variable_to_shape_map() returns the correct information.
    var_map = reader.get_variable_to_shape_map()
    print(var_map)
    #v1_tensor = reader.get_tensor("v1")
  
  #sv = tf.train.Saver(reshape=True, max_to_keep=20, allow_empty=True, defer_build=True)
  #ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint + '/checkpoint'))
  sv = tf.train.Saver(reshape=True, max_to_keep=20, allow_empty=True, defer_build=False,
                      var_list = optimistic_restore_vars(checkpoint) if checkpoint else None)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Restoring old model ...")
    sv.restore(sess, checkpoint)
    sv= None
    #sess.run(tf.global_variables_initializer())
    print("Restored")
    sv2 = tf.train.Saver(reshape=True, max_to_keep=20)
    print("Saving new model ...")
    sv2.save(sess, model_dir+"/model.ckpt", global_step=global_step+1)
    print(model_dir)
    print(global_step)
    print(sv2.last_checkpoints)
    sv = None
    print("Saved")



# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A simple script for inspect checkpoint files."""

def print_tensors_in_checkpoint_file(model_dir, tensor_name=None, target_file_name=None, all_tensors=True, all_tensor_names=True):
    """Prints tensors in a checkpoint file.

    If no `tensor_name` is provided, prints the tensor names and shape in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.

    Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
        all_tensor_names: Boolean indicating whether to print all tensor names.
    """
    try:
        file_name = saver.latest_checkpoint(model_dir)
        print("Checkpoint = {}".format(file_name))
        reader = training.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor_name: ", key)
                if all_tensors:
                    print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            t = reader.get_tensor(tensor_name)
            print(t.shape)
            print(t)
            if (len(t.shape)<=2) and target_file_name:
              np.savetxt(model_dir+"/"+target_file_name, t)
            
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")
        if "Data loss" in str(e) and (any([e in file_name for e in [".index", ".meta", ".data"]])):
            print("It's likely that this is a V2 checkpoint and you need to provide the filename prefix*. "
                  "Try removing the '.' and extension.")



  
