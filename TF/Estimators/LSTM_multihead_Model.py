# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses import losses
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import l1_regularizer
from tensorflow.contrib import rnn

GLOBAL_REGULARIZER = l2_regularizer(scale=0.1) # 0.05 0.5
ACTIVITY_REGULARIZER = l1_regularizer(scale=0.000001)

def calc_points(pGS,pGC, gs, gc, is_home):
  with tf.variable_scope("Points_Calculation"):
    is_away = tf.logical_not(is_home)
    is_draw = tf.equal(gs,gc)
    is_win = tf.greater(gs,gc)
    is_loss = tf.less(gs,gc)
    is_full = tf.equal(gs,pGS) & tf.equal(gc,pGC)
    is_diff = tf.equal((gs-gc),(pGS-pGC))
    is_tendency = tf.equal(tf.sign(gs-gc) , tf.sign(pGS-pGC))
  
    z = tf.zeros_like(gs) 
  
    draw_points = tf.where(is_draw, tf.where(is_full, z+6, tf.where(is_diff, z+2, z)), z) 
    home_win_points  = tf.where(is_win & is_home & is_tendency, tf.where(is_full, z+4, tf.where(is_diff, z+3, z+2)), z)
    home_loss_points = tf.where(is_loss & is_home & is_tendency,  tf.where(is_full, z+7, tf.where(is_diff, z+5, z+4)), z)
    away_win_points  = tf.where(is_win & is_away & is_tendency,  tf.where(is_full, z+7, tf.where(is_diff, z+5, z+4)), z)
    away_loss_points = tf.where(is_loss & is_away & is_tendency, tf.where(is_full, z+4, tf.where(is_diff, z+3, z+2)), z)
    
    points = tf.cast(draw_points+home_win_points+home_loss_points+away_loss_points+away_win_points, tf.float32)
  return (points, is_tendency, is_diff, is_full,
          draw_points, home_win_points, home_loss_points, away_loss_points, away_win_points)

def constant_tensors():
  
  d = pd.DataFrame()
  d["index"]=range(49*49)
  d["gs1"]=d["index"]//(7*49)
  d["gc1"]=np.mod(d["index"]//49, 7)
#  d["gs2"]=np.mod(d["index"]//7, 7)
#  d["gc2"]=np.mod(d["index"], 7)
  d["gs3"]=np.mod(d["index"]//7, 7)
  d["gc3"]=np.mod(d["index"], 7)
  d["target"]=[7*(gs3-gs1)+gc3-gc1 if gs1<=gs3 and gc1<=gc3 else -1 for gs1,gs3,gc1,gc3,i in zip(d["gs1"],d["gs3"],d["gc1"],d["gc3"],d["index"])]
  d["gs2"]=np.mod(d["target"]//7, 7)
  d["gc2"]=np.mod(d["target"], 7)
#  d
#  d[40:80]
  #d[800:1600]
  
  with tf.variable_scope("Constants"):
    tc_1d_goals_i = tf.constant([0,1,2,3,4,5,6], dtype=tf.int64, shape=[7])
    tc_1d1_goals_f = tf.reshape(tf.cast(tc_1d_goals_i, tf.float32), [7,1])
    
    tc_4d_goals_i = tf.meshgrid(tc_1d_goals_i, tc_1d_goals_i, tc_1d_goals_i, tc_1d_goals_i)
    tc_4d_goals_i = tf.stack(tc_4d_goals_i)
    tc_4d_goals_i = tf.transpose(tc_4d_goals_i, [2,1,3,4,0])
    tc_4d_goals_i = tf.reshape(tc_4d_goals_i, [49*49,2*2])

    tc_home_masks = calc_points(tc_4d_goals_i[:,0],tc_4d_goals_i[:,1], tc_4d_goals_i[:,2], tc_4d_goals_i[:,3], True) [0:4]
    tc_home_points_i = tc_home_masks[0]
    tc_home_points_i = tf.reshape(tc_home_points_i, [49,49])
    
    tc_away_points_i = calc_points(tc_4d_goals_i[:,0],tc_4d_goals_i[:,1], tc_4d_goals_i[:,2], tc_4d_goals_i[:,3], False) [0]
    tc_away_points_i = tf.reshape(tc_away_points_i, [49,49])

    p_tendency_mask_f = tf.cast(tc_home_masks[1], tf.float32)
    p_tendency_mask_f  = tf.reshape(p_tendency_mask_f, [49,49])
    p_tendency_mask_f = p_tendency_mask_f / tf.reduce_sum(p_tendency_mask_f, axis=[1], keep_dims=True)
    
    p_gdiff_mask_f = tf.cast(tc_home_masks[2], tf.float32)
    p_gdiff_mask_f  = tf.reshape(p_gdiff_mask_f, [49,49])
    p_gdiff_mask_f = p_gdiff_mask_f / tf.reduce_sum(p_gdiff_mask_f, axis=[1], keep_dims=True)

    p_fulltime_index_matrix = tf.reshape(tf.constant(d["target"].as_matrix(), dtype=tf.int64), shape=[49,49])
#    with tf.Session() as sess:
#      print(sess.run([tc_1d_goals_i, tc_home_points_i,tc_away_points_i,p_home_points_i,p_away_points_i]))
#      print(sess.run([p_tendency_mask_f[0:4,:], p_gdiff_mask_f[0:4,:]]))
      
    logfactorial = [0.000000000000000,
        0.000000000000000,
        0.693147180559945,
        1.791759469228055,
        3.178053830347946,
        4.787491742782046,
        6.579251212010101,
        8.525161361065415,
        10.604602902745251,
        12.801827480081469]
  
    tc_logfactorial_f = tf.constant(logfactorial, dtype=tf.float32, shape=[10])
    
    def calc_poisson_prob(lambda0):
      poisson_prob = tf.map_fn(
        lambda x: tf.exp(-lambda0  + tf.cast(x, tf.float32)*tf.log(lambda0) - tc_logfactorial_f[x]),
        elems = tc_1d_goals_i, dtype=tf.float32)
      poisson_prob = tf.transpose(poisson_prob, [1,0])
      return poisson_prob 
  
    return (tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix)


# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, state_zoneout_prob, is_training=True, seed=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (isinstance(state_zoneout_prob, float) and
        not (state_zoneout_prob >= 0.0 and state_zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % state_zoneout_prob)
    self._cell = cell
    self._zoneout_prob = state_zoneout_prob
    self._seed = seed
    self.is_training = is_training
  
  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
      raise TypeError("Subdivided states need subdivided zoneouts.")
    if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
      raise ValueError("State and zoneout need equally many parts.")
    output, new_state = self._cell(inputs, state, scope)
    state_part_zoneout_prob = self._zoneout_prob
    if isinstance(self.state_size, tuple):
      if self.is_training:
          new_state = tuple((1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                        new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                            for new_state_part, state_part, state_part_zoneout_prob in zip(new_state, state, self._zoneout_prob))
      else:
          new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                            for new_state_part, state_part, state_part_zoneout_prob in zip(new_state, state, self._zoneout_prob))
    else:
      if self.is_training:
          new_state = (1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                        new_state - state, (1 - state_part_zoneout_prob), seed=self._seed) + state
      else:
          new_state = state_part_zoneout_prob * state + (1 - state_part_zoneout_prob) * new_state
    return output, new_state


def dense_layer(input_size, output_size, regularizer=None):  
  W = tf.get_variable(
      name="W",
      initializer=tf.random_normal_initializer(stddev=0.01),
      regularizer = regularizer,
      shape=[input_size, output_size])
  b = tf.get_variable(
      name="b",
      initializer=tf.random_normal_initializer(),
      shape=[output_size])
  def create(X):
    tf.summary.histogram("Inputs", X)
    y = tf.nn.xw_plus_b(X, W, b, name="MakeOutputs")
    tf.summary.histogram("Weights", W)
    tf.summary.histogram("Bias", b)
    tf.summary.histogram("Outputs", y )
    return y
  return create
  
def create_estimator(model_dir, num_label_columns, total_feature_columns, reset_weights):    
  
  def buildGraph(features, mode): 
      hidden_units = 10
      embedding_units = 100
      mapping_units = 80
      rnn_output_size = 40
      rnn_norm_diff_penalty_factor = 0.003
      embedding_regularizer = l2_regularizer(0.00001)
      
      with tf.variable_scope("Input_Layer"):
        features_newgame = features['newgame']
        match_history_t1 = features['match_history_t1']
        match_history_t2 = features['match_history_t2']
        match_history_t12 = features['match_history_t12']
#        goal_history_t1 = features['goal_history_t1']
#        goal_history_t2 = features['goal_history_t2']

        batch_size = tf.shape(features_newgame)[0]
        output_size = num_label_columns
#        input_size = int(match_history_t1.shape[2])
        
      with tf.variable_scope("", reuse=True):
        with tf.variable_scope("Input_Layer"):
          tf.summary.histogram("newgame", features_newgame)
          tf.summary.histogram("match_history_t1", match_history_t1)
          tf.summary.histogram("match_history_t2", match_history_t2)
#          tf.summary.histogram("goal_history_t1", goal_history_t1)
#          tf.summary.histogram("goal_history_t2", goal_history_t2)

      with tf.variable_scope("Embedding", reuse=tf.AUTO_REUSE):
#        embedding = tf.layers.dense(features_newgame, units=embedding_units, activation=tf.nn.relu, name="Match_Embedding",
#                                        #kernel_regularizer=GLOBAL_REGULARIZER,
#                                        #activity_regularizer=ACTIVITY_REGULARIZER,
#                                        )
        with tf.variable_scope("Layer1", reuse=tf.AUTO_REUSE):
          embedding_function_1 = dense_layer(total_feature_columns, embedding_units*2)
        with tf.variable_scope("Layer2", reuse=tf.AUTO_REUSE):
          embedding_function_2 = dense_layer(embedding_units*2, embedding_units)

        with tf.variable_scope("Decoder_Layer1", reuse=tf.AUTO_REUSE):
          embedding_decode_function_1 = dense_layer(embedding_units*2, total_feature_columns)
        with tf.variable_scope("Decoder_Layer2", reuse=tf.AUTO_REUSE):
          embedding_decode_function_2 = dense_layer(embedding_units, embedding_units*2)
  
      def create_embedding(match_history):
        time_steps = int(match_history.shape[1])
        input_size = int(match_history.shape[2])
        batch_mean = tf.get_variable(name="batch_mean", shape=[input_size], dtype=tf.float32)
        batch_variance = tf.get_variable(name="batch_variance", shape=[input_size], dtype=tf.float32)
        
        match_history = tf.reshape(match_history, [-1,input_size])

        if mode == tf.estimator.ModeKeys.TRAIN:
          mean, variance = tf.nn.moments(match_history, axes=[0], keep_dims=False)
          batch_mean = tf.assign(batch_mean, mean)
          batch_variance = tf.assign(batch_variance, variance)
           
        match_history = tf.nn.batch_normalization(match_history, mean=batch_mean, variance=batch_variance, offset=None, scale=None, variance_epsilon=.0001)
        
        with tf.variable_scope("Embedding_Layer1", reuse=tf.AUTO_REUSE):
          match_history = tf.nn.relu(embedding_function_1(match_history))
        with tf.variable_scope("Embedding_Layer2", reuse=tf.AUTO_REUSE):
          match_history = tf.nn.relu(embedding_function_2(match_history))
        match_history = tf.reshape(match_history, [-1,time_steps, embedding_units])
#        return tf.stop_gradient(match_history)
        return match_history
      
      def make_rnn(match_history, sequence_length):
        match_history_features = match_history[:,:,1:total_feature_columns+1]
        match_history_labels = match_history[:,:,total_feature_columns+1:]
        
        feature_embedding = create_embedding(match_history_features)
        feature_embedding = tf.stop_gradient(feature_embedding) # use pre-trained auto_encoder
        
        combined_input_data = tf.concat([feature_embedding, match_history_labels], axis=2)
        #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units )
        #rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units )
        #rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units)
        #rnn_cell = tf.contrib.rnn.UGRNNCell(num_units=hidden_units)
        #rnn_cell.activity_regularizer = GLOBAL_REGULARIZER
#        rnn_cell = tf.contrib.rnn.TimeFreqLSTMCell(num_units=hidden_units,
#                                                   feature_size=84,
#                                                   frequency_skip=1)
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, num_proj=rnn_output_size )
        
        if mode == tf.estimator.ModeKeys.TRAIN:
          # 13.12.2017: was 0.9
          rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
               input_keep_prob=0.95, 
               output_keep_prob=0.95,
               state_keep_prob=0.95)
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, combined_input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32,
                                   sequence_length = sequence_length)
        # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
        # 'state' is a tensor of shape [batch_size, num_units]
        #print(outputs)
        # Tensor("Model/RNN_1/rnn/transpose:0", shape=(1836, 10, 20), dtype=float32)
        # rnn_output_size = 20
        #print(state)

        # norm_stabilizer_loss on output activations    
        
#        frobenius_norm = tf.reduce_sum(tf.square(outputs), axis=2)
#        difference = tf.subtract(frobenius_norm[:,1:], frobenius_norm[:,:-1], name="rnn_norm_diff")
#        tf.summary.histogram("reg_rnn_norm_diff_penalty", difference)
#        l2loss = rnn_norm_diff_penalty_factor * tf.nn.l2_loss(difference)
#        ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, l2loss)
        
        #tf.summary.histogram("Intermediate_Outputs", outputs)
        #tf.summary.histogram("States", state)
#        tf.summary.histogram("gru_cell/gates/kernel", tf.get_variable("rnn/gru_cell/gates/kernel/read:0"))
#        tf.summary.histogram("gru_cell/gates/bias", tf.get_variable("rnn/gru_cell/gates/bias/read:0"))
#        tf.summary.histogram("gru_cell/candidate/kernel", tf.get_variable("rnn/gru_cell/candidate/kernel/read:0"))
#        tf.summary.histogram("gru_cell/candidate/bias", tf.get_variable("rnn/gru_cell/candidate/bias/read:0"))

#        tf.summary.histogram("ugrnn_cell/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/ugrnn_cell/kernel/read:0"))
#        tf.summary.histogram("ugrnn_cell/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/ugrnn_cell/bias/read:0"))
#        tf.summary.histogram("basic_rnn_cell/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/basic_rnn_cell/kernel/read:0"))
#        tf.summary.histogram("basic_rnn_cell/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/basic_rnn_cell/bias/read:0"))
#        tf.summary.histogram("time_freq_lstm_cell/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/time_freq_lstm_cell/W_0/read:0"))
#        tf.summary.histogram("time_freq_lstm_cell/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/time_freq_lstm_cell/B/read:0"))
        return tf.nn.tanh(state[1] )

      with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
        embedding_input = match_history_t1[10:, 0:1, 1:total_feature_columns+1]
        encoded_feature_embedding = create_embedding(embedding_input)
        encoded_feature_embedding = encoded_feature_embedding[:,0,:]
        
        loss_factor = 20.0
        l1loss = tf.multiply(loss_factor, tf.reduce_mean(tf.abs(encoded_feature_embedding)), name="reg_embedding_l1")
        ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, l1loss)

        with tf.variable_scope("Decoding_Layer2", reuse=tf.AUTO_REUSE):
          embedding_output = tf.nn.relu(embedding_decode_function_2(encoded_feature_embedding))
        with tf.variable_scope("Decoding_Layer1", reuse=tf.AUTO_REUSE):
          embedding_output = embedding_decode_function_1(embedding_output)
          embedding_output = tf.expand_dims(embedding_output, axis=1)
        tf.summary.histogram("embedding_input", embedding_input)
        tf.summary.histogram("embedding_output", embedding_output)

      with tf.variable_scope("RNN_1"):
        history_state_t1 = make_rnn(match_history_t1, sequence_length = features_newgame[:,0])  
#        tf.summary.histogram("gru_cell/gates/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/gru_cell/gates/kernel/read:0"))
#        tf.summary.histogram("gru_cell/gates/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/gru_cell/gates/bias/read:0"))
#        tf.summary.histogram("gru_cell/candidate/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/gru_cell/candidate/kernel/read:0"))
#        tf.summary.histogram("gru_cell/candidate/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/gru_cell/candidate/bias/read:0"))
        
      with tf.variable_scope("RNN_2"):
        history_state_t2 = make_rnn(match_history_t2, sequence_length = features_newgame[:,1])  

      with tf.variable_scope("RNN_12"):
        history_state_t12 = make_rnn(match_history_t12, sequence_length = features_newgame[:,3])  

#      with tf.variable_scope("RNN_Goals_1"):
#        goal_history_state_t1 = make_rnn(goal_history_t1, sequence_length = 2*features_newgame[:,0])  
#      with tf.variable_scope("RNN_Goals_2"):
#        goal_history_state_t2 = make_rnn(goal_history_t2, sequence_length = 2*features_newgame[:,0])  

      with tf.variable_scope("Combine"):
#        if mode == tf.estimator.ModeKeys.TRAIN:  # 12.12.: 0.3 -> 0.7
#          #embedding = tf.nn.dropout(embedding, keep_prob=0.01)
#          history_state_t1 = tf.nn.dropout(history_state_t1, keep_prob=0.5)
#          history_state_t2 = tf.nn.dropout(history_state_t2, keep_prob=0.5)
#          history_state_t12 = tf.nn.dropout(history_state_t12, keep_prob=0.5)
        X = tf.concat([history_state_t1, history_state_t2, history_state_t12], axis=1)
        
#      with tf.variable_scope("Mapping_Layer"):
#        X = tf.layers.dense(X, units=mapping_units, activation=tf.nn.relu, name="Mapping",
#                              #kernel_regularizer=GLOBAL_REGULARIZER,
#                              #activity_regularizer=ACTIVITY_REGULARIZER,
#                              )
        if mode == tf.estimator.ModeKeys.TRAIN:  
          X = tf.nn.dropout(X, keep_prob=0.5)

#      with tf.variable_scope("Mapping_Layer_2"):
#        X = tf.layers.dense(X, units=mapping_units//2, activation=tf.nn.relu, name="Mapping",
#                              #kernel_regularizer=GLOBAL_REGULARIZER,
#                              #activity_regularizer=ACTIVITY_REGULARIZER,
#                              )
#
#        if mode == tf.estimator.ModeKeys.TRAIN:  
#          X = tf.nn.dropout(X, keep_prob=0.7)
      X_size = int(X.shape[1])
      with tf.variable_scope("Head_Poisson"):
        outputs_poisson = dense_layer(X_size , output_size, GLOBAL_REGULARIZER)(X)

      with tf.variable_scope("Head_Softmax"):
        outputs_softmax = dense_layer(X_size , 49, GLOBAL_REGULARIZER)(X)

      with tf.variable_scope("Head_Softpoints"):
        outputs_softpoints = dense_layer(X_size , 49, GLOBAL_REGULARIZER)(tf.stop_gradient(X))

      #reset_gate = tf.get_variable(shape=[], dtype=tf.int32, name="reset_gate", initializer=tf.constant_initializer(0))
#      global_step = tf.train.get_global_step()
#      def reset_layers():
#        W = tf.get_default_graph().get_tensor_by_name("Model/Head_Poisson/W:0")
#        tf.assign(W, tf.random_normal(shape=tf.shape(W), stddev=0.01))
#        W = tf.get_default_graph().get_tensor_by_name("Model/Head_Softmax/W:0")
#        tf.assign(W, tf.random_normal(shape=tf.shape(W), stddev=0.01))
#        W = tf.get_default_graph().get_tensor_by_name("Model/Head_Softpoints/W:0")
#        tf.assign(W, tf.random_normal(shape=tf.shape(W), stddev=0.01))
##        tf.assign_add(reset_gate, 1)                
#        return outputs_poisson, outputs_softmax, outputs_softpoints
#      outputs_poisson, outputs_softmax, outputs_softpoints = tf.cond(tf.equal(global_step,30706), reset_layers, lambda: (outputs_poisson, outputs_softmax, outputs_softpoints))      
      return outputs_poisson, outputs_softmax, outputs_softpoints, embedding_input, embedding_output
        
  def create_predictions(logits, maximize_points, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        p_pred_12 = tf.nn.softmax(tf.reshape(logits, [-1, 49]))
        ev_points =  tf.where(t_is_home_bool,
                               tf.matmul(p_pred_12, tc_home_points_i),
                               tf.matmul(p_pred_12, tc_away_points_i))
        if maximize_points:
          a = tf.argmax(ev_points, axis=1)
        else:
          a = tf.argmax(p_pred_12, axis=1)
        
        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
        ev_goals_1 = tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
        ev_goals_2 = tf.matmul(p_marg_2, tc_1d1_goals_f)[:,0]

        p_poisson_1 = calc_poisson_prob(ev_goals_1)
        l_chisq_poiss_marg_1 = tf.reduce_mean(tf.square(p_marg_1 - p_poisson_1)/(p_poisson_1+0.0001), axis=1)
        p_poisson_2 = calc_poisson_prob(ev_goals_2)
        l_chisq_poiss_marg_2 = tf.reduce_mean(tf.square(p_marg_2 - p_poisson_2)/(p_poisson_2+0.0001), axis=1)

        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
        p_pred_x1 = tf.transpose(tf.nn.softmax(tf.reshape(logits, [-1,7,7]), dim=1), [0,2,1]) # tf.reshape(p_pred_12_m, [-1,5])
        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7])
        ev_goals_x1 = tf.matmul(p_pred_x1, tc_1d1_goals_f)
        p_poisson_x1 = calc_poisson_prob(ev_goals_x1[:,0])
        l_chisq_poiss_pred_x1 = tf.reduce_mean(tf.square(p_pred_x1 - p_poisson_x1)/(p_poisson_x1+0.0001), axis=1)

        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7,7])
        ev_goals_x1 = tf.reshape(ev_goals_x1, [-1,7])
        p_poisson_x1 = tf.reshape(p_poisson_x1, [-1,7,7])
        l_chisq_poiss_pred_x1 = tf.reshape(l_chisq_poiss_pred_x1, [-1,7])

        p_pred_x2 = tf.nn.softmax(tf.reshape(logits, [-1,7,7]), dim=2) # tf.reshape(tf.transpose(p_pred_12_m, [0,2,1]), [-1,5])
        p_pred_x2 = tf.reshape(p_pred_x2, [-1,7])
        ev_goals_x2 = tf.matmul(p_pred_x2, tc_1d1_goals_f)[:,0]
        p_poisson_x2 = calc_poisson_prob(ev_goals_x2)
        l_chisq_poiss_pred_x2 = tf.reduce_mean(tf.square(p_pred_x2 - p_poisson_x2)/(p_poisson_x2+0.0001), axis=1)
        
        p_pred_x2 = tf.reshape(p_pred_x2, [-1,7,7])
        ev_goals_x2 = tf.reshape(ev_goals_x2, [-1,7])
        p_poisson_x2 = tf.reshape(p_poisson_x2, [-1,7,7])
        l_chisq_poiss_pred_x2 = tf.reshape(l_chisq_poiss_pred_x2, [-1,7])

        l_chisq_poiss_marg_x1 = tf.reduce_sum(l_chisq_poiss_pred_x1, axis=1)
        l_chisq_poiss_marg_x2 = tf.reduce_sum(l_chisq_poiss_pred_x2, axis=1)

        predictions = {
          "p_marg_1":p_marg_1, 
          "p_marg_2":p_marg_2, 
          "logits":logits,
          "p_pred_x1":p_pred_x1,   
          "p_pred_x2":p_pred_x2,    
          "p_poisson_x1":p_poisson_x1, 
          "p_poisson_x2":p_poisson_x2, 
          "p_pred_12":p_pred_12, 
          "ev_points":ev_points,  
          "pred":pred,
          "ev_goals_1":ev_goals_1,
          "ev_goals_2":ev_goals_2, 
          "ev_goals_x1":ev_goals_x1, 
          "ev_goals_x2":ev_goals_x2, 
          "p_poisson_1":p_poisson_1, 
          "p_poisson_2":p_poisson_2, 
          "l_chisq_poiss_marg_1":l_chisq_poiss_marg_1, 
          "l_chisq_poiss_marg_2":l_chisq_poiss_marg_2, 
          "l_chisq_poiss_pred_x1":l_chisq_poiss_pred_x1, 
          "l_chisq_poiss_pred_x2":l_chisq_poiss_pred_x2,  
          "l_chisq_poiss_marg_x1":l_chisq_poiss_marg_x1, 
          "l_chisq_poiss_marg_x2":l_chisq_poiss_marg_x2 
        }
        return predictions

  def create_predictions_from_ev_goals(ev_goals_1, ev_goals_2, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
        
#        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        #p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        #p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
        p_poisson_1 = calc_poisson_prob(ev_goals_1)
#        l_chisq_poiss_marg_1 = tf.reduce_mean(tf.square(p_marg_1 - p_poisson_1)/(p_poisson_1+0.0001), axis=1)
        p_poisson_2 = calc_poisson_prob(ev_goals_2)
#        l_chisq_poiss_marg_2 = tf.reduce_mean(tf.square(p_marg_2 - p_poisson_2)/(p_poisson_2+0.0001), axis=1)
        px = tf.split(p_poisson_1, 7, axis=1)
        py = tf.split(p_poisson_2, 7, axis=1)
        p_xy = [x*y for x in px for y in py]
        p_pred_12 = tf.stack(p_xy, axis=1)
        p_pred_12 = tf.reshape(p_pred_12, [-1, 49])
        ev_points =  tf.where(t_is_home_bool,
                               tf.matmul(p_pred_12, tc_home_points_i),
                               tf.matmul(p_pred_12, tc_away_points_i))
        
        a = tf.argmax(ev_points, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
        p_pred_x1 = tf.transpose(tf.reshape(p_pred_12, [-1,7,7]), [0,2,1]) # tf.reshape(p_pred_12_m, [-1,5])
        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7])
        ev_goals_x1 = tf.matmul(p_pred_x1, tc_1d1_goals_f)
        p_poisson_x1 = calc_poisson_prob(ev_goals_x1[:,0])
        l_chisq_poiss_pred_x1 = tf.reduce_mean(tf.square(p_pred_x1 - p_poisson_x1)/(p_poisson_x1+0.0001), axis=1)
        
        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7,7])
        ev_goals_x1 = tf.reshape(ev_goals_x1, [-1,7])
        p_poisson_x1 = tf.reshape(p_poisson_x1, [-1,7,7])
        l_chisq_poiss_pred_x1 = tf.reshape(l_chisq_poiss_pred_x1, [-1,7])
        
        #p_pred_x2 = tf.nn.softmax(outputs, dim=2) # tf.reshape(tf.transpose(p_pred_12_m, [0,2,1]), [-1,5])
        p_pred_x2 = tf.transpose(tf.reshape(p_pred_12, [-1,7,7]), [0,1,2]) 
        p_pred_x2 = tf.reshape(p_pred_x2, [-1,7])
        ev_goals_x2 = tf.matmul(p_pred_x2, tc_1d1_goals_f)[:,0]
        p_poisson_x2 = calc_poisson_prob(ev_goals_x2)
        l_chisq_poiss_pred_x2 = tf.reduce_mean(tf.square(p_pred_x2 - p_poisson_x2)/(p_poisson_x2+0.0001), axis=1)

        p_pred_x2 = tf.reshape(p_pred_x2, [-1,7,7])
        ev_goals_x2 = tf.reshape(ev_goals_x2, [-1,7])
        p_poisson_x2 = tf.reshape(p_poisson_x2, [-1,7,7])
        l_chisq_poiss_pred_x2 = tf.reshape(l_chisq_poiss_pred_x2, [-1,7])
        
        l_chisq_poiss_marg_x1 = tf.reduce_sum(l_chisq_poiss_pred_x1, axis=1)
        l_chisq_poiss_marg_x2 = tf.reduce_sum(l_chisq_poiss_pred_x2, axis=1)

        predictions = {
          "p_marg_1":p_poisson_1, 
          "p_marg_2":p_poisson_2, 
          "p_pred_x1":p_pred_x1,   
          "p_pred_x2":p_pred_x2,    
          "p_poisson_x1":p_poisson_x1, 
          "p_poisson_x2":p_poisson_x2, 
          "p_pred_12":p_pred_12, 
          "ev_points":ev_points,  
          "pred":pred,
          "ev_goals_1":ev_goals_1,
          "ev_goals_2":ev_goals_2, 
          "ev_goals_x1":ev_goals_x1, 
          "ev_goals_x2":ev_goals_x2, 
          "p_poisson_1":p_poisson_1, 
          "p_poisson_2":p_poisson_2, 
#          "l_chisq_poiss_marg_1":l_chisq_poiss_marg_1, 
#          "l_chisq_poiss_marg_2":l_chisq_poiss_marg_2, 
          "l_chisq_poiss_pred_x1":l_chisq_poiss_pred_x1, 
          "l_chisq_poiss_pred_x2":l_chisq_poiss_pred_x2,  
          "l_chisq_poiss_marg_x1":l_chisq_poiss_marg_x1, 
          "l_chisq_poiss_marg_x2":l_chisq_poiss_marg_x2 
        }
        return predictions


  def add_noise(labels, labels2, stddev):  
    with tf.variable_scope("Label_Noise"):
      noise = tf.random_normal(shape=tf.shape(labels), mean=0.0, stddev=stddev, dtype=tf.float32) 
      noise2 = tf.random_normal(shape=tf.shape(labels), mean=0.0, stddev=stddev, dtype=tf.float32) 
      noise = tf.cast(tf.round(noise), tf.int32)
      noise2 = tf.cast(tf.round(noise2), tf.int32)
      labels=tf.nn.relu(labels+noise)
      labels2=tf.nn.relu(labels2+noise2)
    return labels, labels2 

  def create_model_regularization_metrics():
    with tf.variable_scope("Regularization"):
  #     Weight penalty for Layer 1
  #    W = tf.get_default_graph().get_tensor_by_name("Layer1/dense/kernel/read:0")
  #    reg_layer1_L2 = tf.reduce_mean(tf.square(W), axis=1)
  #    reg_layer1_L1 = tf.reduce_mean(tf.abs(W), axis=1)
  #    # Weight penalty for Layer 2
  #    W = tf.get_default_graph().get_tensor_by_name("Layer2/dense/kernel/read:0")
  #    reg_layer2_L2 = tf.reduce_mean(tf.square(W), axis=1)
  #    reg_layer2_L1 = tf.reduce_mean(tf.abs(W), axis=1)
  #    # Weight penalty for Output Layer 
      W = tf.get_default_graph().get_tensor_by_name("Model/Output_Layer/W:0")
      reg_layerOut_L2 = tf.reduce_mean(tf.square(W), axis=1)
      reg_layerOut_L1 = tf.reduce_mean(tf.abs(W), axis=1)
      
      #l_regularization = tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)
  #    l_regularization = tf.reduce_mean(0.001*reg_layer1_L1+1000*reg_layer1_L2)
  #    l_regularization += tf.reduce_mean(0.001*reg_layer2_L1+1000*reg_layer2_L2)
      #l_regularization += tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)

      
      K1 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/gru_cell/gates/kernel/read:0")
      K2 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/gru_cell/candidate/kernel/read:0")
      reg_Gates_L2 = tf.reduce_mean(tf.square(K1), axis=1)
      reg_Candidates_L2 = tf.reduce_mean(tf.square(K2), axis=1)      
      l_regularization = tf.reduce_mean(reg_Gates_L2+reg_Candidates_L2)

      K12 = tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/gates/kernel/read:0")
      K22 = tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/candidate/kernel/read:0")
      reg_Gates_L22 = tf.reduce_mean(tf.square(K12), axis=1)
      reg_Candidates_L22 = tf.reduce_mean(tf.square(K22), axis=1)      
      l_regularization += tf.reduce_mean(reg_Gates_L22+reg_Candidates_L22)

#      UGRNN_Cell_kernel = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/ugrnn_cell/kernel/read:0")
#      reg_Kernel_L2 = tf.reduce_mean(tf.square(UGRNN_Cell_kernel), axis=1)
#      RNN_Cell_kernel = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/basic_rnn_cell/kernel/read:0")
#      RNN_Cell_kernel = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/time_freq_lstm_cell/W_0/read:0")
#      reg_Kernel_L2 = tf.reduce_mean(tf.square(RNN_Cell_kernel), axis=1)
#      l_regularization += tf.reduce_mean(reg_Kernel_L2)
      
      reg_eval_metric_ops = {
            "l_regularization": tf.metrics.mean(l_regularization)
  #        , "reg_layer1_L1": tf.metrics.mean(reg_layer1_L1)
  #        , "reg_layer1_L2": tf.metrics.mean(reg_layer1_L2)
  #        , "reg_layer2_L1": tf.metrics.mean(reg_layer2_L1)
  #        , "reg_layer2_L2": tf.metrics.mean(reg_layer2_L2)
          , "reg_layerOut_L1": tf.metrics.mean(reg_layerOut_L1)
          , "reg_layerOut_L2": tf.metrics.mean(reg_layerOut_L2)
#          , "reg_Kernel_L2": tf.metrics.mean(reg_Kernel_L2)
          , "reg_Gates_L2_1": tf.metrics.mean(reg_Gates_L2)
          , "reg_Candidates_L2_1": tf.metrics.mean(reg_Candidates_L2)
          , "reg_Gates_L2_2": tf.metrics.mean(reg_Gates_L22)
          , "reg_Candidates_L2_2": tf.metrics.mean(reg_Candidates_L22)
      }
    return l_regularization, reg_eval_metric_ops

  def create_rnn_regularization_metrics():
    
    def rnn_summary(scope, node_name, reg_eval_metric_ops, loss_factor=0.0):
      node = tf.get_default_graph().get_tensor_by_name("Model/"+scope+"/rnn/lstm_cell/"+node_name+"/read:0")
      node_l2 = tf.reduce_mean(tf.square(node))
      reg_eval_metric_ops.update({"regularization/"+scope+"_"+node_name+"_L2": tf.metrics.mean(node_l2)})
      tf.summary.histogram(scope+"/rnn/lstm_cell/"+node_name, node)
      if (loss_factor>0.0):
        l2loss = tf.multiply(loss_factor, tf.nn.l2_loss(node), name=scope+"/"+node_name)
        ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, l2loss)
      return ops 
      
    reg_eval_metric_ops = {}
    rnn_summary("RNN_1", "kernel", reg_eval_metric_ops, 0.0001)
    rnn_summary("RNN_1", "bias", reg_eval_metric_ops)
    rnn_summary("RNN_2", "kernel", reg_eval_metric_ops, 0.0001)
    rnn_summary("RNN_2", "bias", reg_eval_metric_ops)
    rnn_summary("RNN_12", "kernel", reg_eval_metric_ops, 0.0001)
    rnn_summary("RNN_12", "bias", reg_eval_metric_ops)
    

    return reg_eval_metric_ops

  def corrcoef(x_t,y_t):
    def t(x): return tf.transpose(x)
    x_t = tf.expand_dims(x_t, axis=0)
    y_t = tf.expand_dims(y_t, axis=0)
    xy_t = tf.concat([x_t, y_t], axis=0)
    mean_t = tf.reduce_mean(xy_t, axis=1, keep_dims=True)
    cov_t = ((xy_t-mean_t) @ t(xy_t-mean_t))/tf.cast(tf.shape(x_t)[1]-1, tf.float32)
    cov2_t = tf.diag(1/tf.sqrt(tf.diag_part(cov_t)))
    cor = cov2_t @ cov_t @ cov2_t
    #print(cor)
    return cor[0,1]
    
  def create_result_metrics(pGS,pGC, labels, labels2, t_is_home_bool, probs, ev_goals_1, ev_goals_2, tc):      
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    
    with tf.variable_scope("Metrics"):
      point_summary = calc_points(pGS,pGC, labels, labels2, t_is_home_bool)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary

      pred_draw = tf.cast(tf.equal(pGS,pGC), tf.float32)
      pred_home_win = tf.cast((tf.greater(pGS, pGC) & t_is_home_bool) | (tf.less(pGS, pGC) & tf.logical_not(t_is_home_bool)), tf.float32)
      pred_away_win = tf.cast((tf.less(pGS, pGC) & t_is_home_bool) | (tf.greater(pGS, pGC) & tf.logical_not(t_is_home_bool)), tf.float32)

      labels_float  = tf.cast(labels, tf.float32)
      labels_float2 = tf.cast(labels2, tf.float32)
      # reduce to 6 goals max. for evaluation
      gs = tf.minimum(labels,6)
      gc = tf.minimum(labels2,6)

      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    
      
      pt_log_softpoints = tf.reduce_sum(tf.log(probs+0.02) * achievable_points_mask, axis=1)
      pt_softpoints = tf.reduce_sum(probs * achievable_points_mask, axis=1)

      capped_probs = tf.minimum(probs, 0.10)                    
      pt_softpoints_capped = tf.reduce_sum(capped_probs * achievable_points_mask, axis=1)
      
      l_diff_ev_goals_L1 = tf.abs(labels_float-labels_float2-( ev_goals_1-ev_goals_2))

      eval_metric_ops = {
           "z_points": tf.metrics.mean(pt_actual_points)
          , "metric_is_tendency": tf.metrics.mean(is_tendency)
          , "metric_is_diff": tf.metrics.mean(is_diff)
          , "metric_is_full": tf.metrics.mean(is_full)
          , "metric_pred_home_win": tf.metrics.mean(pred_home_win)
          , "metric_pred_away_win": tf.metrics.mean(pred_away_win)
          , "metric_pred_draw": tf.metrics.mean(pred_draw)
#          , "points/pt_draw_points": tf.metrics.mean(pt_draw_points)
#          , "points/pt_home_win_points": tf.metrics.mean(pt_home_win_points)
#          , "points/pt_away_loss_points": tf.metrics.mean(pt_away_loss_points)
#          , "points/pt_away_win_points": tf.metrics.mean(pt_away_win_points)
#          , "points/pt_home_loss_points": tf.metrics.mean(pt_home_loss_points)
          , "pt_softpoints" : tf.metrics.mean(pt_softpoints)
          , "pt_softpoints_capped" : tf.metrics.mean(pt_softpoints_capped)
          , "pt_log_softpoints" : tf.metrics.mean(pt_log_softpoints)
          , "metric_ev_goals1_L1": tf.metrics.mean_absolute_error(labels=labels_float, predictions=ev_goals_1 )
          , "metric_ev_goals2_L1": tf.metrics.mean_absolute_error(labels=labels_float2, predictions=ev_goals_2 )
          , "metric_ev_goals1_L2": tf.metrics.mean_squared_error(labels=labels_float, predictions=ev_goals_1 )
          , "metric_ev_goals2_L2": tf.metrics.mean_squared_error(labels=labels_float2, predictions=ev_goals_2 )
          , "metric_ev_goals_diff_L1": tf.metrics.mean(l_diff_ev_goals_L1)
          , "metric_cor_1":tf.metrics.mean(corrcoef(ev_goals_1, labels_float))
          , "metric_cor_2":tf.metrics.mean(corrcoef(ev_goals_2, labels_float2))
          , "metric_cor_diff":tf.metrics.mean(corrcoef(ev_goals_1-ev_goals_2, labels_float-labels_float2))
      }
    return eval_metric_ops

  def create_losses_and_metrics_autoencoder(embedding_input, embedding_output):
    with tf.variable_scope("Autoencoder"):
      l_autoenc = tf.losses.mean_squared_error(labels=embedding_input, predictions=embedding_output, reduction=tf.losses.Reduction.MEAN)
      #l_autoenc = tf.nn.log_poisson_loss(targets=embedding_input, log_input=embedding_output)
      
      eval_metric_ops = {
           "loss/l_autoenc": tf.metrics.mean(l_autoenc)
      }
    return eval_metric_ops, tf.reduce_mean(l_autoenc)
    
  def create_losses_RNN(outputs_poisson, outputs_softmax, outputs_softpoints, t_labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

    with tf.variable_scope("Prediction"):
      labels_float  = t_labels[:,0]
      labels_float2 = t_labels[:,1]
      labels = tf.cast(labels_float, tf.int32)
      labels2 = tf.cast(labels_float2, tf.int32)

      if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
        labels, labels2 = add_noise(labels, labels2, 0.01)      

      # reduce to 6 goals max. for training
      gs = tf.minimum(labels,6)
      gc = tf.minimum(labels2,6)
  
    with tf.variable_scope("Losses"):
      l_loglike_poisson = tf.nn.log_poisson_loss(targets=t_labels, log_input=outputs_poisson)
      
#      l_loglike_ev_goals1 = ev_goals_1 - labels_float*tf.log(ev_goals_1)
#      l_loglike_ev_goals2 = ev_goals_2 - labels_float2*tf.log(ev_goals_2)
  
#      z = tf.cast(tf.zeros_like(gs), tf.float32)
      
  #    p_tendency = tf.gather(p_tendency_mask_f, gs*7+gc)
  #    p_gdiff = tf.gather(p_gdiff_mask_f, gs*7+gc)
      p_full = tf.one_hot(gs*7+gc, 49)
  #    
      l_full       = tf.nn.softmax_cross_entropy_with_logits(labels=p_full, logits=tf.reshape(outputs_softmax, [-1,49]))
  #    l_gdiff       = tf.nn.softmax_cross_entropy_with_logits(labels=p_gdiff, logits=tf.reshape(logits, [-1,49]))
  #    l_tendency       = tf.nn.softmax_cross_entropy_with_logits(labels=p_tendency, logits=tf.reshape(logits, [-1,49]))
  #
  #        l_loglike_ev_goals1 = tf.nn.log_poisson_loss(labels_float, log_lambda[:,0])
  #        l_loglike_ev_goals2 = tf.nn.log_poisson_loss(labels_float2, log_lambda[:,1])
  #        loss_l2 = tf.reduce_mean(tf.square(logits*0.2))

      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    
      
      pt_log_softpoints = tf.reduce_sum(tf.log(predictions["sp/p_pred_12"]+0.02) * achievable_points_mask, axis=1)
      print(pt_log_softpoints)

#       loss = l_full # + l_loglike_ev_goals1 + l_loglike_ev_goals2 
#      loss *= tf.where( t_is_home_loss_bool, z+2, z+1) # away win = 4 x
#      loss *= tf.where( t_is_home_win_bool, z+2, z+1)  # home win = 2 x, draw = 1 x
      loss = 10*tf.reduce_mean(l_loglike_poisson)
      loss += 1*tf.reduce_mean(l_full)  
      # loss += tf.reduce_mean(l_softpoints) 
      loss -= 0.01*tf.reduce_mean(pt_log_softpoints)
      #loss += 10*l_regularization
      
      regularization_metric_ops = create_rnn_regularization_metrics()
      reg_term = tf.losses.get_regularization_loss()
      reg_terms = tf.losses.get_regularization_losses()  
      print(reg_terms)
      #['Model/Embedding/Layer1/W/Regularizer/l2_regularizer:0', 'Model/Embedding/Layer2/W/Regularizer/l2_regularizer:0', 'Model/RNN_1/mul:0', 'Model/RNN_2/mul:0', 'Model/RNN_12/mul:0', 'Model/Head_Poisson/W/Regularizer/l2_regularizer:0', 'Model/Head_Softmax/W/Regularizer/l2_regularizer:0', 'Model/Head_Softpoints/W/Regularizer/l2_regularizer:0']
      #print([r.name for r in reg_terms])
      loss += reg_term
      
      tf.summary.scalar("loss", loss)
      
      eval_metric_ops = {
          "loss/l_loglike_poisson": tf.metrics.mean(l_loglike_poisson)
          , "loss/l_full": tf.metrics.mean(l_full)
          , "regularization/total": tf.metrics.mean(reg_term)
      }
      eval_metric_ops.update(regularization_metric_ops)
      eval_metric_ops.update({"regularization/"+k.name[:-2] : tf.metrics.mean(k) for k in reg_terms})
      
    return eval_metric_ops, loss

  def reset_layers():
    W = tf.get_default_graph().get_tensor_by_name("Model/Head_Poisson/W:0")
    op1 = tf.assign(W, tf.random_normal(shape=tf.shape(W), stddev=0.01))
    W = tf.get_default_graph().get_tensor_by_name("Model/Head_Softmax/W:0")
    op2 = tf.assign(W, tf.random_normal(shape=tf.shape(W), stddev=0.01))
    W = tf.get_default_graph().get_tensor_by_name("Model/Head_Softpoints/W:0")
    op3 = tf.assign(W, tf.random_normal(shape=tf.shape(W), stddev=0.01))
#        tf.assign_add(reset_gate, 1)                
    return tf.group(op1, op2, op3)

  def reset_rnns(): 
    node1 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/lstm_cell/kernel/Assign:0")
    node2 = tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/lstm_cell/kernel/Assign:0")
    node12 = tf.get_default_graph().get_tensor_by_name("Model/RNN_12/rnn/lstm_cell/kernel/Assign:0")
    return tf.group(node1, node2, node12)

  def model(features, labels, mode):
    tc = constant_tensors()
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    
    with tf.variable_scope("Model"):
      outputs_poisson, outputs_softmax, outputs_softpoints, embedding_input, embedding_output = buildGraph(features, mode)
      t_is_home_bool = tf.equal(features["newgame"][:,2] , 1)
#      t_is_train_bool = tf.equal(features["Train"] , True)
      
      #predictions = create_predictions(outputs_poisson, outputs_softmax, outputs_softpoints, t_is_home_bool, tc)
      predictions_softmax = create_predictions(outputs_softmax, True, t_is_home_bool, tc)
      predictions_softpoints = create_predictions(outputs_softpoints, False, t_is_home_bool, tc)
      T1_GFT = tf.exp(outputs_poisson[:,0])
      T2_GFT = tf.exp(outputs_poisson[:,1])
      T1_GHT = tf.exp(outputs_poisson[:,2])
      T2_GHT = tf.exp(outputs_poisson[:,3])
      T1_GH2 = tf.exp(outputs_poisson[:,16])
      T2_GH2 = tf.exp(outputs_poisson[:,17])
      predictions_poisson_FT = create_predictions_from_ev_goals(T1_GFT, T2_GFT, t_is_home_bool, tc)
      predictions_poisson_HT = create_predictions_from_ev_goals(T1_GHT+T1_GH2, T2_GHT+T2_GH2, t_is_home_bool, tc)

      predictions_softmax = {"sm/"+k:v for k,v in predictions_softmax.items() }
      predictions_softpoints = {"sp/"+k:v for k,v in predictions_softpoints.items() }
      predictions_poisson_FT = {"p1/"+k:v for k,v in predictions_poisson_FT.items() }
      predictions_poisson_HT = {"p2/"+k:v for k,v in predictions_poisson_HT.items() }
      
      predictions = predictions_softmax 
      predictions.update(predictions_softpoints)
      predictions.update(predictions_poisson_FT)
      predictions.update(predictions_poisson_HT)
      predictions["outputs_poisson"] = outputs_poisson

#      DEBUG_LABEL = True      
#      if DEBUG_LABEL:
#        t_goals_1  = tf.cast(labels[:, 0], dtype=tf.int32)
#        t_goals_2  = tf.cast(labels[:, 1], dtype=tf.int32)
#        t_goals = tf.stack([t_goals_1,t_goals_2], axis=1)
#        predictions.update({"labels":labels, "goals":t_goals})
      
      export_outputs = {
          "predictions": tf.estimator.export.ClassificationOutput(predictions["sm/p_pred_12"])
      }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    with tf.variable_scope("Evaluation"):

      t_goals_1  = tf.cast(labels[:, 0], dtype=tf.int32)
      t_goals_2  = tf.cast(labels[:, 1], dtype=tf.int32)
      t_goals = tf.stack([t_goals_1,t_goals_2], axis=1)
      
      t_is_home_loss_bool = (t_is_home_bool & tf.less(t_goals_1, t_goals_2)) | (tf.logical_not(t_is_home_bool) & tf.greater(t_goals_1, t_goals_2))
      t_is_home_win_bool = (t_is_home_bool & tf.greater(t_goals_1, t_goals_2)) | (tf.logical_not(t_is_home_bool) & tf.less(t_goals_1, t_goals_2))
#      t_is_draw_bool = tf.equal(t_goals_1, t_goals_2)
      
      def append_result_metrics(result_metrics, prefix):
        result_metrics_new = create_result_metrics(predictions[prefix+"/pred"][:,0], predictions[prefix+"/pred"][:,1], t_goals[:,0], t_goals[:,1], t_is_home_bool, predictions["sp/p_pred_12"], predictions[prefix+"/ev_goals_1"], predictions[prefix+"/ev_goals_2"], tc)
        result_metrics_new = {prefix+"/"+k:v for k,v in result_metrics_new.items() }
        result_metrics.update(result_metrics_new)

      result_metrics = {}
      append_result_metrics(result_metrics, "sm")
      append_result_metrics(result_metrics, "sp")
      append_result_metrics(result_metrics, "p1")
      append_result_metrics(result_metrics, "p2")

      eval_metric_ops, auto_loss = create_losses_and_metrics_autoencoder(embedding_input, embedding_output)
      eval_metric_ops.update(result_metrics)

      eval_rnn_metric_ops, loss = create_losses_RNN(outputs_poisson, outputs_softmax, outputs_softpoints, labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool)
      eval_metric_ops.update(eval_rnn_metric_ops)
      
      global_step = tf.train.get_global_step()
      loss = tf.cond(tf.less(global_step, 1000), lambda: auto_loss, lambda: loss+auto_loss) # first steps: only pretraining of auto_encoder

    eval_metric_ops.update({"summary/"+k:v for k,v in eval_metric_ops.items() if "z_points" in k })
    for key, value in eval_metric_ops.items():
      tf.summary.scalar(key, value[1])

#    for key in ["z_points", "pt_softpoints", "metric_is_tendency", "metric_is_diff", "metric_is_full", "metric_pred_home_win", "metric_pred_away_win", "metric_pred_draw"]:
#      tf.summary.scalar(key, eval_metric_ops[key][1], family="Summary") 
    tf.summary.scalar("summary/loss", loss) 
    
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, loss= loss, eval_metric_ops=eval_metric_ops)
  
  
    #optimizer = tf.train.GradientDescentOptimizer(1e-4)
    learning_rate = 1e-3
    print("Learning rate = {}".format(learning_rate))
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    if reset_weights:
      train_op = tf.group(reset_layers(), train_op)
#      train_op = tf.group(reset_rnns(), train_op)
      
    summary_op=tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                       output_dir=model_dir+"/train",
                                       scaffold=None,
                                       summary_op=summary_op)
  
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions 
                                      , loss= loss, train_op=train_op
                                      , eval_metric_ops=eval_metric_ops
                                      , training_hooks = [summary_hook]  )

  return tf.estimator.Estimator(model_fn=model, model_dir=model_dir,
                                config = tf.estimator.RunConfig(save_checkpoints_steps=100,save_summary_steps=100))


def makeStaticPrediction(features, labels):
  with tf.variable_scope("Static_Prediction"):
    tc_1d1_goals_f, tc_home_points_i, _ , calc_poisson_prob, _ , _, _ = constant_tensors()
    
    features = features["newgame"]
    labels = labels[features[:,2]==1] # Home only
    features = features[features[:,2]==1] # Home only
    labels = labels.round(0).astype('int32') # integer goal values only
    
    def count_probs(t1goals, t2goals):
      results = pd.Series(["{}:{}".format(gs, gc) for gs,gc in zip(t1goals, t2goals)])
      counts = results.value_counts()
      print(counts)
      counts2 = ["{}:{}".format(x, y) in counts 
              and (counts["{}:{}".format(x, y)] / np.sum(counts))
              or 0.000001 for x in range(7) for y in range(7)]
      return tf.constant(counts2, shape=[1, 49], dtype=tf.float32)
   
    H1_p_pred_12 = count_probs(labels[:,2], labels[:,3]) # GHT
    H2_p_pred_12 = count_probs(labels[:,0] - labels[:,2], labels[:,1] - labels[:,3]) # GFT - GHT
    p_pred_12 = count_probs(labels[:,0], labels[:,1]) # GFT
    ev_points =  tf.matmul(p_pred_12, tc_home_points_i)
  
    a = tf.argmax(ev_points, axis=1)
    pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
  
    def build_predictions(p_pred_12, prefix):
      p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
      p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
      p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
      ev_goals_1 = tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
      ev_goals_2 = tf.matmul(p_marg_2, tc_1d1_goals_f)[:,0]
      
      p_poisson_1 = calc_poisson_prob(ev_goals_1)
      p_poisson_2 = calc_poisson_prob(ev_goals_2)
      
      predictions = {
        prefix+"p_marg_1":p_marg_1.eval()[0],
        prefix+"p_marg_2":p_marg_2.eval()[0],
        prefix+"p_pred_12":p_pred_12.eval()[0], 
        prefix+"ev_goals_1":ev_goals_1.eval()[0],
        prefix+"ev_goals_2":ev_goals_2.eval()[0],
        prefix+"p_poisson_1":p_poisson_1.eval()[0],
        prefix+"p_poisson_2":p_poisson_2.eval()[0],
      }
      return predictions
    
    predictions = {
      "ev_points":ev_points.eval()[0],
      "pred":pred.eval()[0],
    }
  
    predictions.update(build_predictions(p_pred_12, ""))
    predictions.update(build_predictions(H1_p_pred_12, "H1_"))
    predictions.update(build_predictions(H2_p_pred_12, "H2_"))
  return predictions

      
