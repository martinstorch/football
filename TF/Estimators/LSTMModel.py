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
from tensorflow.contrib import rnn

GLOBAL_REGULARIZER = l2_regularizer(scale=0.0001)

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

  
def create_estimator(model_dir, feature_columns, label_columns, teamnames):    
  
  def buildGraph(features, mode): 
      max_time = 34
      batch_size = (tf.shape(features["sequence_length"])[0])//max_time
      with tf.variable_scope("Input_Layer"):
        X = tf.feature_column.input_layer(features, feature_columns)        
        #X_partitions = tf.split(X, batch_size)
        #X_partitions = tf.split(X, [max_time]*batch_size)
        #X = tf.stack(X_partitions)
        X = tf.reshape(X, [tf.shape(X)[0] // max_time, max_time, int(X.shape[1])])
        tf.summary.histogram("Inputs", X)
      
      with tf.variable_scope("RNN"):
        #print(features["sequence_length"][::max_time])
        
        hidden_units = 10
        output_size = 33
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units )
        # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units )
        #rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, num_proj=output_size )
        #rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units)
        #rnn_cell = tf.contrib.rnn.UGRNNCell(num_units=hidden_units)
        #rnn_cell.activity_regularizer = GLOBAL_REGULARIZER
        rnn_cell = tf.contrib.rnn.TimeFreqLSTMCell(num_units=hidden_units,
                                                   feature_size=84,
                                                   frequency_skip=1)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
          rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
               input_keep_prob=1.0, 
               output_keep_prob=0.7,
               state_keep_prob=0.7)
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, X,
                                   initial_state=initial_state,
                                   dtype=tf.float32,
                                   sequence_length = features["sequence_length"][::max_time])
        # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
        # 'state' is a tensor of shape [batch_size, num_units]
        outputs = tf.reshape(outputs, [-1, hidden_units])
        tf.summary.histogram("Outputs", outputs)
        tf.summary.histogram("States", state[0])
#        tf.summary.histogram("gru_cell/gates/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/gru_cell/gates/kernel/read:0"))
#        tf.summary.histogram("gru_cell/gates/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/gru_cell/gates/bias/read:0"))
#        tf.summary.histogram("gru_cell/candidate/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/gru_cell/candidate/kernel/read:0"))
#        tf.summary.histogram("gru_cell/candidate/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/gru_cell/candidate/bias/read:0"))
#        tf.summary.histogram("ugrnn_cell/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/ugrnn_cell/kernel/read:0"))
#        tf.summary.histogram("ugrnn_cell/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/ugrnn_cell/bias/read:0"))
#        tf.summary.histogram("basic_rnn_cell/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/basic_rnn_cell/kernel/read:0"))
#        tf.summary.histogram("basic_rnn_cell/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/basic_rnn_cell/bias/read:0"))
        tf.summary.histogram("time_freq_lstm_cell/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/time_freq_lstm_cell/W_0/read:0"))
        tf.summary.histogram("time_freq_lstm_cell/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/time_freq_lstm_cell/B/read:0"))
        
      with tf.variable_scope("Mapping_Layer"):
        #X = tf.nn.batch_normalization(X, mean=0, variance=1, offset=None, scale=None, variance_epsilon=.000000001)
        tf.summary.histogram("Inputs", outputs)
        W = tf.get_variable(
            name="WM",
            initializer=tf.random_normal_initializer(stddev=0.01),
            regularizer = GLOBAL_REGULARIZER,
            shape=[hidden_units, output_size])
        b = tf.get_variable(
            name="bM",
            initializer=tf.random_normal_initializer(),
            shape=[output_size])
        outputs = tf.nn.xw_plus_b(outputs, W, b, name="MakeOutputs")
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Bias", b)
        tf.summary.histogram("Outputs", outputs )

      with tf.variable_scope("Output_Layer"):
        #X = tf.nn.batch_normalization(X, mean=0, variance=1, offset=None, scale=None, variance_epsilon=.000000001)
        tf.summary.histogram("Inputs", outputs)
        W = tf.get_variable(
            name="W",
            initializer=tf.random_normal_initializer(stddev=0.01),
            regularizer = GLOBAL_REGULARIZER,
            shape=[output_size , 49])
        b = tf.get_variable(
            name="b",
            initializer=tf.random_normal_initializer(),
            shape=[49])
        logits = tf.nn.xw_plus_b(tf.reshape((outputs), [-1, output_size]), W, b, name="MakeOutputs")
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Bias", b)
        tf.summary.histogram("Outputs", logits )

      return outputs, logits
        
  def create_predictions(outputs, logits, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        p_pred_12 = tf.nn.softmax(tf.reshape(logits, [-1, 49]))
        
        
        ev_points =  tf.where(t_is_home_bool,
                               tf.matmul(p_pred_12, tc_home_points_i),
                               tf.matmul(p_pred_12, tc_away_points_i))
        
        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
        ev_goals_1 = tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
        ev_goals_2 = tf.matmul(p_marg_2, tc_1d1_goals_f)[:,0]

        p_poisson_1 = calc_poisson_prob(ev_goals_1)
        l_chisq_poiss_marg_1 = tf.reduce_mean(tf.square(p_marg_1 - p_poisson_1)/(p_poisson_1+0.0001), axis=1)
        p_poisson_2 = calc_poisson_prob(ev_goals_2)
        l_chisq_poiss_marg_2 = tf.reduce_mean(tf.square(p_marg_2 - p_poisson_2)/(p_poisson_2+0.0001), axis=1)

        #a = tf.argmax(ev_points, axis=1)
        a = tf.argmax(p_pred_12, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
#        p_pred_x1 = tf.transpose(tf.reshape(p_pred_12, [-1,7,7]), [0,2,1]) # tf.reshape(p_pred_12_m, [-1,5])
#        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7])
#        ev_goals_x1 = tf.matmul(p_pred_x1, tc_1d1_goals_f)
#        p_poisson_x1 = calc_poisson_prob(ev_goals_x1[:,0])
#        l_chisq_poiss_pred_x1 = tf.reduce_mean(tf.square(p_pred_x1 - p_poisson_x1)/(p_poisson_x1+0.0001), axis=1)
#        
        p_pred_x1 = tf.transpose(tf.nn.softmax(tf.reshape(logits, [-1,7,7]), dim=1), [0,2,1]) # tf.reshape(p_pred_12_m, [-1,5])
        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7])
        ev_goals_x1 = tf.matmul(p_pred_x1, tc_1d1_goals_f)
        p_poisson_x1 = calc_poisson_prob(ev_goals_x1[:,0])
        l_chisq_poiss_pred_x1 = tf.reduce_mean(tf.square(p_pred_x1 - p_poisson_x1)/(p_poisson_x1+0.0001), axis=1)

        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7,7])
        ev_goals_x1 = tf.reshape(ev_goals_x1, [-1,7])
        p_poisson_x1 = tf.reshape(p_poisson_x1, [-1,7,7])
        l_chisq_poiss_pred_x1 = tf.reshape(l_chisq_poiss_pred_x1, [-1,7])

#        #p_pred_x2 = tf.nn.softmax(outputs, dim=2) # tf.reshape(tf.transpose(p_pred_12_m, [0,2,1]), [-1,5])
#        p_pred_x2 = tf.transpose(tf.reshape(p_pred_12, [-1,7,7]), [0,1,2]) 
#        p_pred_x2 = tf.reshape(p_pred_x2, [-1,7])
#        ev_goals_x2 = tf.matmul(p_pred_x2, tc_1d1_goals_f)[:,0]
#        p_poisson_x2 = calc_poisson_prob(ev_goals_x2)
#        l_chisq_poiss_pred_x2 = tf.reduce_mean(tf.square(p_pred_x2 - p_poisson_x2)/(p_poisson_x2+0.0001), axis=1)
#
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
          "outputs":outputs,
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
  
  def create_predictions_from_outputs(outputs, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
        
#        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        #p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        #p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
        ev_goals_1 = tf.exp(outputs[:,label_idx("T1_GFT")]) # tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
        ev_goals_1 = tf.exp(outputs[:,label_idx("T1_GHT")])+tf.exp(outputs[:,label_idx("T1_GH2")]) # tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
        ev_goals_2 = tf.exp(outputs[:,label_idx("T2_GFT")]) # tf.matmul(p_marg_2, tc_1d1_goals_f)[:,0]
        ev_goals_2 = tf.exp(outputs[:,label_idx("T2_GHT")])+tf.exp(outputs[:,label_idx("T2_GH2")]) # tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
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
          "outputs":outputs,
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
      noise = tf.cast(tf.round(noise), tf.int64)
      noise2 = tf.cast(tf.round(noise2), tf.int64)
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
      
      l_regularization = tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)
  #    l_regularization = tf.reduce_mean(0.001*reg_layer1_L1+1000*reg_layer1_L2)
  #    l_regularization += tf.reduce_mean(0.001*reg_layer2_L1+1000*reg_layer2_L2)
      l_regularization += tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)

#      K1 = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/gru_cell/gates/kernel/read:0")
#      K2 = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/gru_cell/candidate/kernel/read:0")
#      reg_Gates_L2 = tf.reduce_mean(tf.square(K1), axis=1)
#      reg_Candidates_L2 = tf.reduce_mean(tf.square(K2), axis=1)      
#      UGRNN_Cell_kernel = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/ugrnn_cell/kernel/read:0")
#      reg_Kernel_L2 = tf.reduce_mean(tf.square(UGRNN_Cell_kernel), axis=1)
#      l_regularization += tf.reduce_mean(reg_Gates_L2+reg_Candidates_L2)
#      RNN_Cell_kernel = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/basic_rnn_cell/kernel/read:0")
      RNN_Cell_kernel = tf.get_default_graph().get_tensor_by_name("Model/RNN/rnn/time_freq_lstm_cell/W_0/read:0")
      reg_Kernel_L2 = tf.reduce_mean(tf.square(RNN_Cell_kernel), axis=1)
      l_regularization += tf.reduce_mean(reg_Kernel_L2)
      
      reg_eval_metric_ops = {
            "l_regularization": tf.metrics.mean(l_regularization)
  #        , "reg_layer1_L1": tf.metrics.mean(reg_layer1_L1)
  #        , "reg_layer1_L2": tf.metrics.mean(reg_layer1_L2)
  #        , "reg_layer2_L1": tf.metrics.mean(reg_layer2_L1)
  #        , "reg_layer2_L2": tf.metrics.mean(reg_layer2_L2)
          , "reg_layerOut_L1": tf.metrics.mean(reg_layerOut_L1)
          , "reg_layerOut_L2": tf.metrics.mean(reg_layerOut_L2)
          , "reg_Kernel_L2": tf.metrics.mean(reg_Kernel_L2)
#          , "reg_Gates_L2": tf.metrics.mean(reg_Gates_L2)
#          , "reg_Candidates_L2": tf.metrics.mean(reg_Candidates_L2)
      }
    return l_regularization, reg_eval_metric_ops
    
  def create_losses_and_metrics(predictions, labels, labels2, t_is_home_bool, mode, tc):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f = tc

    if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
      labels, labels2 = add_noise(labels, labels2, 0.03)      

    with tf.variable_scope("Prediction"):
      labels_float = tf.cast(labels, tf.float32)
      labels_float2 = tf.cast(labels2, tf.float32)
      # reduce to 6 goals max. for training
      gs = tf.minimum(labels,6)
      gc = tf.minimum(labels2,6)
  
      ev_goals_1 = predictions["ev_goals_1"]
      ev_goals_2 = predictions["ev_goals_2"]
      pGS = predictions["pred"][:,0] 
      pGC = predictions["pred"][:,1] 
      p_pred_12 = predictions["p_pred_12"]
      outputs = predictions["outputs"]
      outputs = tf.reshape(outputs, [-1, 49])

    with tf.variable_scope("Losses"):
      l_loglike_ev_goals1 = ev_goals_1 - labels_float*tf.log(ev_goals_1)
      l_loglike_ev_goals2 = ev_goals_2 - labels_float2*tf.log(ev_goals_2)
  
      l_diff_ev_goals_L1 = tf.sqrt(tf.abs(labels_float-labels_float2-( ev_goals_1-ev_goals_2)))
  
      z = tf.cast(tf.zeros_like(gs), tf.float32)
      
      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))                        
      pt_softpoints = tf.reduce_sum(p_pred_12 * achievable_points_mask, axis=1)
      
  #    p_tendency = tf.gather(p_tendency_mask_f, gs*7+gc)
  #    p_gdiff = tf.gather(p_gdiff_mask_f, gs*7+gc)
      p_full = tf.one_hot(gs*7+gc, 49)
  #    
      l_full       = tf.nn.softmax_cross_entropy_with_logits(labels=p_full, logits=tf.reshape(logits, [-1,49]))
  #    l_gdiff       = tf.nn.softmax_cross_entropy_with_logits(labels=p_gdiff, logits=tf.reshape(logits, [-1,49]))
  #    l_tendency       = tf.nn.softmax_cross_entropy_with_logits(labels=p_tendency, logits=tf.reshape(logits, [-1,49]))
  #
      t_is_home_win_bool = (t_is_home_bool & tf.greater(gs,gc)) | (tf.logical_not(t_is_home_bool) & tf.less(gs,gc))
      t_is_home_loss_bool = (t_is_home_bool & tf.less(gs,gc)) | (tf.logical_not(t_is_home_bool) & tf.greater(gs,gc))
      t_is_draw_bool = tf.equal(gs,gc)
  
  #        l_expected_points = tf.where( t_is_draw_bool, z+4, tf.where( t_is_home_win_bool, z+1, z+2)) * l_full
  #        l_expected_points += tf.where( t_is_home_loss_bool, z+4, z+2) * l_tendency
  #        l_expected_points += tf.where( tf.logical_not(t_is_draw_bool), l_gdiff, z)
  
  #    l_expected_points = 2 * tf.where( t_is_draw_bool, z+0.6, tf.where( t_is_home_win_bool, z+4, z+7)) * l_full
  #    l_expected_points += 3 * tf.where( t_is_draw_bool, z+2, tf.where( t_is_home_win_bool, z+2, z+4)) * l_tendency
  #    l_expected_points += 1 * tf.where( t_is_draw_bool, z+2, tf.where( t_is_home_win_bool, z+3, z+5)) * l_gdiff
      
  #        l_loglike_ev_goals1 = tf.nn.log_poisson_loss(labels_float, log_lambda[:,0])
  #        l_loglike_ev_goals2 = tf.nn.log_poisson_loss(labels_float2, log_lambda[:,1])
  #        loss_l2 = tf.reduce_mean(tf.square(logits*0.2))
      point_summary = calc_points(pGS,pGC, labels, labels2, t_is_home_bool)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary
      
  #    pt_expected_points = tf.where( t_is_draw_bool, z+4, tf.where( t_is_home_win_bool, z+1, z+2)) * tf.cast(is_full, tf.float32)
  #    pt_expected_points += tf.where( t_is_home_loss_bool, z+4, z+2) * tf.cast(is_tendency, tf.float32)
  #    pt_expected_points += tf.where( tf.logical_not(t_is_draw_bool), tf.cast(is_diff, tf.float32), z)
  
      l_regularization, reg_eval_metric_ops = create_model_regularization_metrics()
      
      pt_softpoints_capped = tf.reduce_sum(tf.minimum(p_pred_12, 0.15) * achievable_points_mask, axis=1)
      l_softpoints = -tf.log(5+pt_softpoints)*1
      l_softpoints *= tf.where( t_is_home_loss_bool, z+2, z+1)
      l_softpoints -= pt_softpoints_capped
  ##    l_hinge_full       = tf.losses.hinge_loss(labels=p_full, logits=tf.reshape(logits, [-1,49]))
  ##    l_hinge_full   *= tf.where( t_is_home_loss_bool, z+2, z+1)
  ##    l_hinge_gdiff      = tf.losses.hinge_loss(labels=tf.sign(p_gdiff), logits=tf.reshape(logits, [-1,49]))
  ##    l_hinge_tendency   = tf.losses.hinge_loss(labels=tf.sign(p_tendency), logits=tf.reshape(logits, [-1,49]))
  ##    l_hinge_tendency   *= tf.where( t_is_home_loss_bool, z+4, z+2)
  #    l_hinge_full = multiclass_hinge_loss(p_full, p_full, logits)
  #    tf.summary.scalar("l_hinge_full_max", tf.reduce_max(l_hinge_full))
  #    tf.summary.scalar("l_hinge_full_min", tf.reduce_min(l_hinge_full))
  #    l_hinge_full *= tf.where( t_is_home_loss_bool, z+2, z+1)
  #    l_hinge_gdiff = multiclass_hinge_loss(p_full, tf.sign(p_gdiff), logits)
  #    l_hinge_tendency = multiclass_hinge_loss(p_full, tf.sign(p_tendency), logits)
  #    l_hinge_tendency *= tf.where( t_is_home_loss_bool, z+4, z+2)
      
      #l_gdiff       = tf.nn.softmax_cross_entropy_with_logits(labels=p_gdiff, logits=tf.reshape(logits, [-1,49]))
      #l_tendency       = tf.nn.softmax_cross_entropy_with_logits(labels=p_tendency, logits=tf.reshape(logits, [-1,49]))
  
  #    pred_idx = tf.argmax(tf.reshape(logits, [-1, 49]), axis=1)
  #    pt_expected_points = tf.gather(achievable_points_mask, pred_idx, axis=1)
      
      loss = l_softpoints + 0.3*l_full
      #loss = l_hinge_full #+ l_hinge_gdiff + l_hinge_tendency #- pt_softpoints
      #loss = -pt_expected_points 
  #    loss = l_expected_points - pt_softpoints 
      #loss = 1*l_full
      
      #loss +=  0.01*(l_loglike_ev_goals1+l_loglike_ev_goals2)
  #        loss +=  0.10*(l_chisq_poiss_marg_1+l_chisq_poiss_marg_2)
      #loss += 0.01*(predictions["l_chisq_poiss_marg_x1"] + predictions["l_chisq_poiss_marg_x2"])
  #    loss += 0.5*l_diff_ev_goals_L1
  #    loss *= tf.where( t_is_home_loss_bool, z+2, z+1)
  #   
      loss = tf.reduce_mean(loss) + 1000*l_regularization
      tf.summary.scalar("loss", loss)
      
      eval_metric_ops = {
  #          "l_full": tf.metrics.mean(l_full)
  #        , "l_gdiff": tf.metrics.mean(l_gdiff)
  #        , "l_tendency": tf.metrics.mean(l_tendency)
  #        , "l_hinge_full": tf.metrics.mean(l_hinge_full)
  #        , "l_hinge_gdiff": tf.metrics.mean(l_hinge_gdiff)
  #        , "l_hinge_tendency": tf.metrics.mean(l_hinge_tendency)
          "l_softpoints": tf.metrics.mean(l_softpoints) 
  #        , "l_expected_points": tf.metrics.mean(l_expected_points) 
          , "l_loglike_ev_goals1": tf.metrics.mean(l_loglike_ev_goals1)
          , "l_loglike_ev_goals2": tf.metrics.mean(l_loglike_ev_goals2)
          , "l_chisq_poiss_marg_1": tf.metrics.mean(predictions["l_chisq_poiss_marg_1"])
          , "l_chisq_poiss_marg_2": tf.metrics.mean(predictions["l_chisq_poiss_marg_2"])
          , "l_chisq_poiss_marg_x1":tf.metrics.mean(predictions["l_chisq_poiss_marg_x1"])
          , "l_chisq_poiss_marg_x2":tf.metrics.mean(predictions["l_chisq_poiss_marg_x2"])
          , "l_diff_ev_goals_L1": tf.metrics.mean(l_diff_ev_goals_L1)
          , "pt_softpoints": tf.metrics.mean(pt_softpoints)
          , "pt_softpoints_capped": tf.metrics.mean(pt_softpoints_capped)
          , "metric_ev_goals1_L1": tf.metrics.mean_absolute_error(labels=labels_float, predictions=ev_goals_1 )
          , "metric_ev_goals2_L1": tf.metrics.mean_absolute_error(labels=labels_float2, predictions=ev_goals_2 )
          , "metric_ev_goals1_L2": tf.metrics.mean_squared_error(labels=labels_float, predictions=ev_goals_1 )
          , "metric_ev_goals2_L2": tf.metrics.mean_squared_error(labels=labels_float2, predictions=ev_goals_2 )
          , "z_points": tf.metrics.mean(pt_actual_points)
  #        , "pt_expected_points": tf.metrics.mean(pt_expected_points)
          , "metric_is_tendency": tf.metrics.mean(is_tendency)
          , "metric_is_diff": tf.metrics.mean(is_diff)
          , "metric_is_full": tf.metrics.mean(is_full)
          , "pt_draw_points": tf.metrics.mean(pt_draw_points)
          , "pt_home_win_points": tf.metrics.mean(pt_home_win_points)
          , "pt_away_loss_points": tf.metrics.mean(pt_away_loss_points)
          , "pt_away_win_points": tf.metrics.mean(pt_away_win_points)
          , "pt_home_loss_points": tf.metrics.mean(pt_home_loss_points)
      }
      eval_metric_ops.update(reg_eval_metric_ops)
      
    for key, value in eval_metric_ops.items():
      tf.summary.scalar(key, value[1]) 
    return eval_metric_ops, loss
  
  def create_losses_and_metrics_RNN(outputs, logits, t_labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

#    if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
#      labels, labels2 = add_noise(labels, labels2, 0.01)      

    with tf.variable_scope("Prediction"):
      labels_float  = t_labels[:,label_idx("T1_GFT")]
      labels_float2 = t_labels[:,label_idx("T2_GFT")]
      labels = tf.cast(labels_float, tf.int32)
      labels2 = tf.cast(labels_float2, tf.int32)
      # reduce to 6 goals max. for training
      gs = tf.minimum(labels,6)
      gc = tf.minimum(labels2,6)
  
      ev_goals_1 = predictions["ev_goals_1"]
      ev_goals_2 = predictions["ev_goals_2"]
      pGS = predictions["pred"][:,0] 
      pGC = predictions["pred"][:,1] 
#      p_pred_12 = predictions["p_pred_12"]
#      logits = predictions["logits"]
#      logits = tf.reshape(logits, [-1, 49])

    with tf.variable_scope("Losses"):
      l_loglike_poisson = tf.nn.log_poisson_loss(targets=t_labels, log_input=outputs)
      
#      l_loglike_ev_goals1 = ev_goals_1 - labels_float*tf.log(ev_goals_1)
#      l_loglike_ev_goals2 = ev_goals_2 - labels_float2*tf.log(ev_goals_2)
      l_diff_ev_goals_L1 = tf.sqrt(tf.abs(labels_float-labels_float2-( ev_goals_1-ev_goals_2)))
  
#      z = tf.cast(tf.zeros_like(gs), tf.float32)
      
  #    p_tendency = tf.gather(p_tendency_mask_f, gs*7+gc)
  #    p_gdiff = tf.gather(p_gdiff_mask_f, gs*7+gc)
      p_full = tf.one_hot(gs*7+gc, 49)
  #    
      l_full       = tf.nn.softmax_cross_entropy_with_logits(labels=p_full, logits=tf.reshape(logits, [-1,49]))
  #    l_gdiff       = tf.nn.softmax_cross_entropy_with_logits(labels=p_gdiff, logits=tf.reshape(logits, [-1,49]))
  #    l_tendency       = tf.nn.softmax_cross_entropy_with_logits(labels=p_tendency, logits=tf.reshape(logits, [-1,49]))
  #
  #        l_loglike_ev_goals1 = tf.nn.log_poisson_loss(labels_float, log_lambda[:,0])
  #        l_loglike_ev_goals2 = tf.nn.log_poisson_loss(labels_float2, log_lambda[:,1])
  #        loss_l2 = tf.reduce_mean(tf.square(logits*0.2))
      point_summary = calc_points(pGS,pGC, labels, labels2, t_is_home_bool)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary
      
            # softpoints
      gs = tf.minimum(features["T1_GFT"],6)
      gc = tf.minimum(features["T2_GFT"],6)
      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    
      
      z = tf.cast(tf.zeros_like(gs), tf.float32)
      pt_softpoints = tf.reduce_sum(predictions["p_pred_12"] * achievable_points_mask, axis=1)
      pt_log_softpoints = tf.reduce_sum(tf.log(predictions["p_pred_12"]+0.3) * achievable_points_mask, axis=1)

#      pt_softpoints_capped = tf.reduce_sum(tf.minimum(predictions["p_pred_12"], 0.15) * achievable_points_mask, axis=1)
#      l_softpoints = -tf.log(5+pt_softpoints)*1
      capped_probs = tf.minimum(predictions["p_pred_12"], 0.10)                    
      pt_softpoints_capped = tf.reduce_sum(capped_probs * achievable_points_mask, axis=1)
      l_softpoints = -pt_softpoints_capped
      l_softpoints *= tf.where( t_is_home_loss_bool, z+2, z+1)
#      loss -= tf.reduce_mean(tf.where(t_is_draw_bool, 
#                                      0.0*tf.zeros_like(pt_softpoints), 
#                                      0.1*tf.ones_like(pt_softpoints)) *
#                   (0*tf.log(pt_softpoints)+(pt_softpoints)))

      
      l_regularization, reg_eval_metric_ops = create_model_regularization_metrics()
      
      #l_gdiff       = tf.nn.softmax_cross_entropy_with_logits(labels=p_gdiff, logits=tf.reshape(logits, [-1,49]))
      #l_tendency       = tf.nn.softmax_cross_entropy_with_logits(labels=p_tendency, logits=tf.reshape(logits, [-1,49]))
  
#       loss = l_full # + l_loglike_ev_goals1 + l_loglike_ev_goals2 
#      loss *= tf.where( t_is_home_loss_bool, z+2, z+1) # away win = 4 x
#      loss *= tf.where( t_is_home_win_bool, z+2, z+1)  # home win = 2 x, draw = 1 x
      loss = tf.reduce_mean(l_loglike_poisson)
      loss += 0.0001*tf.reduce_mean(l_full)  
      # loss += tf.reduce_mean(l_softpoints) 
      loss -= tf.reduce_mean(pt_log_softpoints)
      loss += 1*l_regularization
      
      #print(tf.get_collection(tf.GraphKeys.WEIGHTS))
      reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      #print(reg_variables)
      reg_term = tf.contrib.layers.apply_regularization(GLOBAL_REGULARIZER, reg_variables)
      #print(reg_term)
      loss += reg_term
      
      #loss += tf.reduce_mean(tf.square(labels_float-ev_goals_1))
      #loss += tf.reduce_mean(tf.square(labels_float2-ev_goals_2))
      l_chisq_poiss_pred_x1 = predictions["l_chisq_poiss_pred_x1"]
      l_chisq_poiss_pred_x2 = predictions["l_chisq_poiss_pred_x2"]
      loss += 0.0001*tf.reduce_mean(l_chisq_poiss_pred_x1)
      loss += 0.0001*tf.reduce_mean(l_chisq_poiss_pred_x2)
      
      tf.summary.scalar("loss", loss)
      
      eval_metric_ops = {
#            "l_loglike_ev_goals1": tf.metrics.mean(l_loglike_ev_goals1)
#          , "l_loglike_ev_goals2": tf.metrics.mean(l_loglike_ev_goals2)
           "l_diff_ev_goals_L1": tf.metrics.mean(l_diff_ev_goals_L1)
          , "l_loglike_poisson": tf.metrics.mean(l_loglike_poisson)
          , "l_chisq_poiss_pred_x1": tf.metrics.mean(l_chisq_poiss_pred_x1)
          , "l_chisq_poiss_pred_x2": tf.metrics.mean(l_chisq_poiss_pred_x2)
          , "l_full": tf.metrics.mean(l_full)
          , "metric_ev_goals1_L1": tf.metrics.mean_absolute_error(labels=labels_float, predictions=ev_goals_1 )
          , "metric_ev_goals2_L1": tf.metrics.mean_absolute_error(labels=labels_float2, predictions=ev_goals_2 )
          , "metric_ev_goals1_L2": tf.metrics.mean_squared_error(labels=labels_float, predictions=ev_goals_1 )
          , "metric_ev_goals2_L2": tf.metrics.mean_squared_error(labels=labels_float2, predictions=ev_goals_2 )
          , "metric_is_tendency": tf.metrics.mean(is_tendency)
          , "metric_is_diff": tf.metrics.mean(is_diff)
          , "metric_is_full": tf.metrics.mean(is_full)
          , "pt_softpoints" : tf.metrics.mean(pt_softpoints)
          , "pt_log_softpoints" : tf.metrics.mean(pt_log_softpoints)
          , "pt_softpoints_capped" : tf.metrics.mean(pt_softpoints_capped)
      }
      eval_metric_ops.update(reg_eval_metric_ops)
      
    return eval_metric_ops, loss

  def create_result_metrics(pGS,pGC, labels, labels2, t_is_home_bool):      
    with tf.variable_scope("Metrics"):
      point_summary = calc_points(pGS,pGC, labels, labels2, t_is_home_bool)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary
      eval_metric_ops = {
  #        "l_softpoints": tf.metrics.mean(l_softpoints) 
  #        , "pt_softpoints": tf.metrics.mean(pt_softpoints)
  #        , "pt_softpoints_capped": tf.metrics.mean(pt_softpoints_capped)
           "z_points": tf.metrics.mean(pt_actual_points)
          , "metric_is_tendency": tf.metrics.mean(is_tendency)
          , "metric_is_diff": tf.metrics.mean(is_diff)
          , "metric_is_full": tf.metrics.mean(is_full)
          , "pt_draw_points": tf.metrics.mean(pt_draw_points)
          , "pt_home_win_points": tf.metrics.mean(pt_home_win_points)
          , "pt_away_loss_points": tf.metrics.mean(pt_away_loss_points)
          , "pt_away_win_points": tf.metrics.mean(pt_away_win_points)
          , "pt_home_loss_points": tf.metrics.mean(pt_home_loss_points)
      }
    return eval_metric_ops

  def label_idx(label_name):
    return sorted([c.name for c in label_columns]).index(label_name)
  
  def model(features, labels, mode):
    print(mode)
    tc = constant_tensors()
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    
    with tf.variable_scope("Model"):
      outputs, logits = buildGraph(features, mode)
      t_is_home_bool = tf.equal(features["Where"] , "Home")
#      t_is_train_bool = tf.equal(features["Train"] , True)
      
      predictions = create_predictions(outputs, logits, t_is_home_bool, tc)
      
      DEBUG_LABEL = True      
      if DEBUG_LABEL:
        t_labels = tf.feature_column.input_layer(features, label_columns)
        t_goals_1  = tf.cast(t_labels[:, label_idx("T1_GFT")], dtype=tf.int32)
        t_goals_2  = tf.cast(t_labels[:, label_idx("T2_GFT")], dtype=tf.int32)
        t_goals = tf.stack([t_goals_1,t_goals_2], axis=1)
        predictions.update({"labels":t_labels, "goals":t_goals})
      
      export_outputs = {
          "predictions": tf.estimator.export.ClassificationOutput(predictions["p_pred_12"])
      }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    with tf.variable_scope("Evaluation"):

      def drop_all_but_last(x):
        return {key:value[33::34] for key,value in x.items()}
      
      def gather_from_dict(x, index):
        return {key:tf.gather(value, index) for key,value in x.items()}
      
      if mode == tf.estimator.ModeKeys.EVAL:
        # use only the last game in each series for evaluation
        seq_len = tf.cast(features["sequence_length"][::34], tf.int32)
        index = tf.range(start=0, limit=tf.shape(features["sequence_length"])[0], delta=34, dtype=tf.int32)
        index = index + seq_len - 1
        outputs = tf.gather(outputs, index)
        logits = tf.gather(logits, index)
        predictions = gather_from_dict(predictions, index)
        features = drop_all_but_last(features)
        #predictions = drop_all_but_last(predictions)
        #outputs = outputs[33::34]
        t_is_home_bool = tf.equal(features["Where"] , "Home")
        
      t_labels = tf.feature_column.input_layer(features, label_columns)
      t_goals_1  = tf.cast(t_labels[:, label_idx("T1_GFT")], dtype=tf.int32)
      t_goals_2  = tf.cast(t_labels[:, label_idx("T2_GFT")], dtype=tf.int32)
      t_goals = tf.stack([t_goals_1,t_goals_2], axis=1)
      
      t_is_home_loss_bool = (t_is_home_bool & tf.less(features["T1_GFT"],features["T2_GFT"])) | (tf.logical_not(t_is_home_bool) & tf.greater(features["T1_GFT"],features["T2_GFT"]))
      t_is_home_win_bool = (t_is_home_bool & tf.greater(features["T1_GFT"],features["T2_GFT"])) | (tf.logical_not(t_is_home_bool) & tf.less(features["T1_GFT"],features["T2_GFT"]))
      t_is_draw_bool = tf.equal(features["T1_GFT"] , features["T2_GFT"])
      
      eval_metric_ops, loss = create_losses_and_metrics_RNN(outputs, logits, t_labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool)


      result_metrics = create_result_metrics(predictions["pred"][:,0], predictions["pred"][:,1], t_goals[:,0], t_goals[:,1], t_is_home_bool)
      eval_metric_ops.update(result_metrics)
      # loss -= eval_metric_ops["z_points"][1]

    for key, value in eval_metric_ops.items():
      tf.summary.scalar(key, value[1]) 
    
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, loss= loss, eval_metric_ops=eval_metric_ops)
  
  
    global_step = tf.train.get_global_step()
    #optimizer = tf.train.GradientDescentOptimizer(1e-4)
    learning_rate = 1e-2
    print("Learning rate = {}".format(learning_rate))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    
    summary_op=tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                       output_dir=model_dir+"/train",
                                       scaffold=None,
                                       summary_op=summary_op)
  
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions 
                                      , loss= loss, train_op=train
                                      , eval_metric_ops=eval_metric_ops
                                      , training_hooks = [summary_hook]  )

  return tf.estimator.Estimator(model_fn=model, model_dir=model_dir,
                                config = tf.estimator.RunConfig(save_checkpoints_steps=100,save_summary_steps=100))


def makeStaticPrediction(features):
  with tf.variable_scope("Static_Prediction"):
    tc_1d1_goals_f, tc_home_points_i, _ , calc_poisson_prob, _ , _, _ = constant_tensors()
    features = features[features["Where"]=="Home"]
  
    def count_probs(t1goals, t2goals):
      results = pd.Series(["{}:{}".format(gs, gc) for gs,gc in zip(t1goals, t2goals)])
      counts = results.value_counts()
      counts2 = ["{}:{}".format(x, y) in counts 
              and (counts["{}:{}".format(x, y)] / np.sum(counts))
              or 0.000001 for x in range(7) for y in range(7)]
      return tf.constant(counts2, shape=[1, 49], dtype=tf.float32)
   
    H1_p_pred_12 = count_probs(features["T1_GHT"], features["T2_GHT"])
    H2_p_pred_12 = count_probs(features["T1_GFT"]-features["T1_GHT"], features["T2_GFT"]-features["T2_GHT"])
    p_pred_12 = count_probs(features["T1_GFT"], features["T2_GFT"])
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

      
