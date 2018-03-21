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

GLOBAL_REGULARIZER = l2_regularizer(scale=0.1)
MINOR_REGULARIZER = l2_regularizer(scale=0.005)
#MINOR_REGULARIZER = l2_regularizer(scale=0.0)
#GLOBAL_REGULARIZER = l2_regularizer(scale=1.0)
# 16.1.2018
#GLOBAL_REGULARIZER = l2_regularizer(scale=3.0)
# 8.1.2018
#GLOBAL_REGULARIZER = l2_regularizer(scale=10.0)

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
    tc_1d_goals_f = tf.constant([0,1,2,3,4,5,6], dtype=tf.float32, shape=[7])
    tc_1d1_goals_f = tf.reshape(tc_1d_goals_f, [7,1])
    tc_1d7_goals_f = tf.reshape(tc_1d_goals_f, [1,7])
    
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
        6.579251212010101]
#    ,
#        8.525161361065415,
#        10.604602902745251,
#        12.801827480081469]
    tc_1d7_logfactorial_f = tf.constant(logfactorial, dtype=tf.float32, shape=[1,7])
    
    def calc_poisson_prob(lambda0):
      lambda0 = tf.reshape(lambda0, [-1,1]) # make sure that rank=2
      loglambda = tf.log(lambda0)
      x = tf.matmul(loglambda, tc_1d7_goals_f) - tc_1d7_logfactorial_f
      x = tf.exp(x - lambda0)
      return x

    return (tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix)

def add_weight_noise(X, stddev_factor):  
  with tf.variable_scope("Weight_Noise"):
    mean, variance = tf.nn.moments(X, axes=[0], keep_dims=True)
    noise = tf.random_normal(shape=tf.shape(X), mean=0.0, stddev=variance * stddev_factor, dtype=tf.float32) 
  return X+noise
  

def harmonize_outputs(outputs, label_column_names):
  # label_column_names=['T1_GFT', 'T2_GFT', 'T1_GHT', 'T2_GHT', 
  #                     'T1_S', 'T2_S', 'T1_ST', 'T2_ST', 'T1_F', 'T2_F', 'T1_C', 'T2_C', 'T1_Y', 'T2_Y', 'T1_R', 'T2_R', 
  #                     'T1_GH2', 'T2_GH2', 'Loss', 'Draw', 'Win', 'HTLoss', 'HTDraw', 'HTWin', 
  #                     'zScore0:3', 'zScore1:3', 'zScore0:2', 'zScore1:2', 'zScore0:1', 'zScore0:0', 'zScore1:1', 
  #                     'zScore1:0', 'zScore2:1', 'zScore2:0', 'zScore3:1', 'zScore3:0', 'zScore2:2', 'zScore3:2', 'zScore2:3']

  inverse_label_column_names=['T2_GFT', 'T1_GFT', 'T2_GHT', 'T1_GHT', 
                       'T2_S', 'T1_S', 'T2_ST', 'T1_ST', 'T2_F', 'T1_F', 'T2_C', 'T1_C', 'T2_Y', 'T1_Y', 'T2_R', 'T1_R', 
                       'T2_GH2', 'T1_GH2', 'Win', 'Draw', 'Loss', 'HTWin', 'HTDraw', 'HTLoss', 'HT2Win', 'HT2Draw', 'HT2Loss', 
                       'zScore3:0', 'zScore3:1', 'zScore2:0', 'zScore2:1', 'zScore1:0', 'zScore0:0', 'zScore1:1', 
                       'zScore0:1', 'zScore1:2', 'zScore0:2', 'zScore1:3', 'zScore0:3', 'zScore2:2', 'zScore2:3', 'zScore3:2']

  outputs = tf.exp(outputs)  
  
  dict_outputs = {c:outputs[:,i]  for i,c in enumerate(label_column_names)}
  
  batch_length = tf.shape(outputs)[0]
  index = tf.range(0, batch_length)
  dict_outputs["index"]=index
  
  dict_avg = {}
  dict_avg['T1_GFT'] = (dict_outputs['T1_GFT'][0::2] + dict_outputs['T2_GFT'][1::2])/2
  dict_avg['T2_GFT'] = (dict_outputs['T2_GFT'][0::2] + dict_outputs['T1_GFT'][1::2])/2
  dict_avg['T1_GHT'] = (dict_outputs['T1_GHT'][0::2] + dict_outputs['T2_GHT'][1::2])/2
  dict_avg['T2_GHT'] = (dict_outputs['T2_GHT'][0::2] + dict_outputs['T1_GHT'][1::2])/2
  dict_avg['T1_GH2'] = (dict_outputs['T1_GH2'][0::2] + dict_outputs['T2_GH2'][1::2])/2
  dict_avg['T2_GH2'] = (dict_outputs['T2_GH2'][0::2] + dict_outputs['T1_GH2'][1::2])/2
  dict_avg['T1_S'] = (dict_outputs['T1_S'][0::2] + dict_outputs['T2_S'][1::2])/2
  dict_avg['T2_S'] = (dict_outputs['T2_S'][0::2] + dict_outputs['T1_S'][1::2])/2
  dict_avg['T1_ST'] = (dict_outputs['T1_ST'][0::2] + dict_outputs['T2_ST'][1::2])/2
  dict_avg['T2_ST'] = (dict_outputs['T2_ST'][0::2] + dict_outputs['T1_ST'][1::2])/2
  dict_avg['T1_F'] = (dict_outputs['T1_F'][0::2] + dict_outputs['T2_F'][1::2])/2
  dict_avg['T2_F'] = (dict_outputs['T2_F'][0::2] + dict_outputs['T1_F'][1::2])/2
  dict_avg['T1_C'] = (dict_outputs['T1_C'][0::2] + dict_outputs['T2_C'][1::2])/2
  dict_avg['T2_C'] = (dict_outputs['T2_C'][0::2] + dict_outputs['T1_C'][1::2])/2
  dict_avg['T1_Y'] = (dict_outputs['T1_Y'][0::2] + dict_outputs['T2_Y'][1::2])/2
  dict_avg['T2_Y'] = (dict_outputs['T2_Y'][0::2] + dict_outputs['T1_Y'][1::2])/2
  dict_avg['T1_R'] = (dict_outputs['T1_R'][0::2] + dict_outputs['T2_R'][1::2])/2
  dict_avg['T2_R'] = (dict_outputs['T2_R'][0::2] + dict_outputs['T1_R'][1::2])/2
  dict_avg['Loss'] = (dict_outputs['Loss'][0::2] + dict_outputs['Win'][1::2])/2
  dict_avg['Win'] = (dict_outputs['Win'][0::2] + dict_outputs['Loss'][1::2])/2
  dict_avg['Draw'] = (dict_outputs['Draw'][0::2] + dict_outputs['Draw'][1::2])/2
  dict_avg['HTLoss'] = (dict_outputs['HTLoss'][0::2] + dict_outputs['HTWin'][1::2])/2
  dict_avg['HTWin'] = (dict_outputs['HTWin'][0::2] + dict_outputs['HTLoss'][1::2])/2
  dict_avg['HTDraw'] = (dict_outputs['HTDraw'][0::2] + dict_outputs['HTDraw'][1::2])/2
  dict_avg['HT2Loss'] = (dict_outputs['HT2Loss'][0::2] + dict_outputs['HT2Win'][1::2])/2
  dict_avg['HT2Win'] = (dict_outputs['HT2Win'][0::2] + dict_outputs['HT2Loss'][1::2])/2
  dict_avg['HT2Draw'] = (dict_outputs['HT2Draw'][0::2] + dict_outputs['HT2Draw'][1::2])/2
  dict_avg['zScore0:3'] = (dict_outputs['zScore0:3'][0::2] + dict_outputs['zScore3:0'][1::2])/2
  dict_avg['zScore1:3'] = (dict_outputs['zScore1:3'][0::2] + dict_outputs['zScore3:1'][1::2])/2
  dict_avg['zScore2:3'] = (dict_outputs['zScore2:3'][0::2] + dict_outputs['zScore3:2'][1::2])/2
  dict_avg['zScore0:2'] = (dict_outputs['zScore0:2'][0::2] + dict_outputs['zScore2:0'][1::2])/2
  dict_avg['zScore1:2'] = (dict_outputs['zScore1:2'][0::2] + dict_outputs['zScore2:1'][1::2])/2
  dict_avg['zScore2:2'] = (dict_outputs['zScore2:2'][0::2] + dict_outputs['zScore2:2'][1::2])/2
  dict_avg['zScore3:2'] = (dict_outputs['zScore3:2'][0::2] + dict_outputs['zScore2:3'][1::2])/2
  dict_avg['zScore0:1'] = (dict_outputs['zScore0:1'][0::2] + dict_outputs['zScore1:0'][1::2])/2
  dict_avg['zScore1:1'] = (dict_outputs['zScore1:1'][0::2] + dict_outputs['zScore1:1'][1::2])/2
  dict_avg['zScore2:1'] = (dict_outputs['zScore2:1'][0::2] + dict_outputs['zScore1:2'][1::2])/2
  dict_avg['zScore3:1'] = (dict_outputs['zScore3:1'][0::2] + dict_outputs['zScore1:3'][1::2])/2
  dict_avg['zScore0:0'] = (dict_outputs['zScore0:0'][0::2] + dict_outputs['zScore0:0'][1::2])/2
  dict_avg['zScore1:0'] = (dict_outputs['zScore1:0'][0::2] + dict_outputs['zScore0:1'][1::2])/2
  dict_avg['zScore2:0'] = (dict_outputs['zScore2:0'][0::2] + dict_outputs['zScore0:2'][1::2])/2
  dict_avg['zScore3:0'] = (dict_outputs['zScore3:0'][0::2] + dict_outputs['zScore0:3'][1::2])/2
  
  #print(dict_avg)
  home_list = [dict_avg[c]  for c in label_column_names]
  #print(home_list)
  away_list = [dict_avg[c]  for c in inverse_label_column_names]
  #print(away_list)
  t_home = tf.stack(home_list, axis=1)
  t_away = tf.stack(away_list, axis=1)
  t_outputs = tf.concat([t_home, t_away], axis=0)

  # reorder list such that home+away matches are interleaved again
  batch_length = tf.shape(t_outputs)[0]
  index = tf.range(0, batch_length)
  index = tf.where( tf.equal(tf.mod(index,2), 0), index // 2, (batch_length // 2) + (index // 2))  
  t_outputs = tf.gather(t_outputs, index)
              
  t_outputs = tf.log(t_outputs)  
  return t_outputs, index

def create_estimator(model_dir, label_column_names, save_steps, max_to_keep, teams_count):    
  
  
  def buildGraph(features, mode): 
      hidden_units = 20
      
      with tf.variable_scope("Input_Layer"):
        features_newgame = features['newgame']
        match_history_t1 = features['match_history_t1']
        match_history_t2 = features['match_history_t2'] 
        match_history_t12 = features['match_history_t12']
        
        f_date_round = features_newgame[:,4+2*teams_count : 4+2*teams_count+2]
        suppress_team_names = True
        if suppress_team_names:
          features_newgame = tf.concat([features_newgame[:,0:4], features_newgame[:,4+2*teams_count:]], axis=1)
          match_history_t1 = tf.concat([match_history_t1[:,:,0:2], match_history_t1[:,:,2+2*teams_count:]], axis=2)
          match_history_t2 = tf.concat([match_history_t2[:,:,0:2], match_history_t2[:,:,2+2*teams_count:]], axis=2)
          match_history_t12 = tf.concat([match_history_t12[:,:,0:2], match_history_t12[:,:,2+2*teams_count:]], axis=2)
          
        batch_size = tf.shape(features_newgame)[0]
        num_label_columns = len(label_column_names)
        output_size = num_label_columns

        match_history_t1_seqlen = 10*features_newgame[:,0]
        match_history_t2_seqlen = 10*features_newgame[:,1]
        match_history_t12_seqlen = 10*features_newgame[:,3]
        # batch normalization
        features_newgame = tf.layers.batch_normalization(features_newgame, axis=1, momentum=0.8, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        match_history_t1 = tf.layers.batch_normalization(match_history_t1, axis=2, momentum=0.8, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        match_history_t2 = tf.layers.batch_normalization(match_history_t2, axis=2, momentum=0.8, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        match_history_t12 = tf.layers.batch_normalization(match_history_t12, axis=2, momentum=0.8, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        
      with tf.variable_scope("", reuse=True):
        with tf.variable_scope("Input_Layer"):
          tf.summary.histogram("newgame", features_newgame)
          tf.summary.histogram("match_history_t1", match_history_t1)
          tf.summary.histogram("match_history_t2", match_history_t2)
          tf.summary.histogram("match_history_t12", match_history_t12)
          
      def make_rnn_cell():
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units, 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        if mode == tf.estimator.ModeKeys.TRAIN:
          # 13.12.2017: was 0.9
          rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
               input_keep_prob=0.98, 
               output_keep_prob=0.98,
               state_keep_prob=0.98)
        return rnn_cell
      
      number_of_layers = 2
      rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
          [make_rnn_cell() for _ in range(number_of_layers)])


      def make_rnn(match_history, sequence_length):
        #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units )
        # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units )
        #rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, num_proj=output_size )
          # rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units)
        #rnn_cell = tf.contrib.rnn.UGRNNCell(num_units=hidden_units)
        #rnn_cell.activity_regularizer = GLOBAL_REGULARIZER
#        rnn_cell = tf.contrib.rnn.TimeFreqLSTMCell(num_units=hidden_units,
#                                                   feature_size=84,
#                                                   frequency_skip=1)
        
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, match_history,
                                   initial_state=initial_state,
                                   dtype=tf.float32,
                                   sequence_length = sequence_length)
        # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
        # 'state' is a tensor of shape [batch_size, num_units]
        tf.summary.histogram("Intermediate_Outputs", outputs)
        tf.summary.histogram("States", state)
        tf.summary.histogram("Sequence_Length", sequence_length)
        return state[1] # use upper layer state

      with tf.variable_scope("RNN_1"):
        history_state_t1 = make_rnn(match_history_t1, sequence_length = match_history_t1_seqlen)  
        if True:
          tf.summary.histogram("gru_cell/gates0/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/read:0"))
          tf.summary.histogram("gru_cell/gates0/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/read:0"))
          tf.summary.histogram("gru_cell/candidate0/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/read:0"))
          tf.summary.histogram("gru_cell/candidate0/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/read:0"))
          tf.summary.histogram("gru_cell/gates1/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel/read:0"))
          tf.summary.histogram("gru_cell/gates1/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias/read:0"))
          tf.summary.histogram("gru_cell/candidate1/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel/read:0"))
          tf.summary.histogram("gru_cell/candidate1/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias/read:0"))
        
      with tf.variable_scope("RNN_2"):
        history_state_t2 = make_rnn(match_history_t2, sequence_length = match_history_t2_seqlen)  
#        tf.summary.histogram("gru_cell/gates/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/gates/kernel/read:0"))
#        tf.summary.histogram("gru_cell/gates/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/gates/bias/read:0"))
#        tf.summary.histogram("gru_cell/candidate/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/candidate/kernel/read:0"))
#        tf.summary.histogram("gru_cell/candidate/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/candidate/bias/read:0"))
#
      with tf.variable_scope("RNN_12"):
        history_state_t12 = make_rnn(match_history_t12, sequence_length = match_history_t12_seqlen)  
#        tf.summary.histogram("gru_cell/gates/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_12/rnn/gru_cell/gates/kernel/read:0"))
#        tf.summary.histogram("gru_cell/gates/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_12/rnn/gru_cell/gates/bias/read:0"))
#        tf.summary.histogram("gru_cell/candidate/kernel", tf.get_default_graph().get_tensor_by_name("Model/RNN_12/rnn/gru_cell/candidate/kernel/read:0"))
#        tf.summary.histogram("gru_cell/candidate/bias", tf.get_default_graph().get_tensor_by_name("Model/RNN_12/rnn/gru_cell/candidate/bias/read:0"))

      with tf.variable_scope("Combine"):
        embedding = features_newgame
#        if mode == tf.estimator.ModeKeys.TRAIN:  # 12.12.: 0.3 -> 0.7
#          embedding = tf.nn.dropout(features_newgame, keep_prob=0.5)
#          history_state_t1 = tf.nn.dropout(history_state_t1, keep_prob=0.5)
#          history_state_t2 = tf.nn.dropout(history_state_t2, keep_prob=0.5)
#          history_state_t12 = tf.nn.dropout(history_state_t12, keep_prob=0.5)
        X = tf.concat([embedding, history_state_t1, history_state_t2, history_state_t12], axis=1)
        input_tensor = X
        
      with tf.variable_scope("Layer1"):
        X = tf.layers.dense(X, 128, name="Dense", activation=tf.nn.relu, 
                            #kernel_regularizer = MINOR_REGULARIZER,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        tf.summary.histogram("Outputs", X)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.get_default_graph().get_tensor_by_name("Model/Layer1/Dense/kernel/read:0"))
        X = tf.layers.batch_normalization(X, axis=1, momentum=0.9, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
      
      if mode == tf.estimator.ModeKeys.TRAIN:  # 12.12.: 0.3 -> 0.7
        X = tf.nn.dropout(X, keep_prob=0.7)

      with tf.variable_scope("Layer2"):
        X = tf.layers.dense(X, 64, name="Dense", activation=tf.nn.relu, 
                            #kernel_regularizer = MINOR_REGULARIZER,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        tf.summary.histogram("Outputs", X)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.get_default_graph().get_tensor_by_name("Model/Layer2/Dense/kernel/read:0"))
        #X0=X
        X = tf.layers.batch_normalization(X, axis=1, momentum=0.9, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        #X=X0
      if mode == tf.estimator.ModeKeys.TRAIN:  # 12.12.: 0.3 -> 0.7
        X = tf.nn.dropout(X, keep_prob=0.9)

      with tf.variable_scope("Mapping_Layer"):
        tf.summary.histogram("Inputs", X)
        W = tf.get_variable(
            name="WM",
            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            #regularizer = l2_regularizer(scale=0.1),#1.5),
            shape=[int(X.shape[1]), output_size])
        b = tf.get_variable(
            name="bM",
            initializer=tf.random_normal_initializer(),
            shape=[output_size])

        #tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)

        if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
          #W = add_weight_noise(W, 0.001)      
          #W = add_weight_noise(W, 0.01)      
          #W = add_weight_noise(W, 0.1)      
#          W = add_weight_noise(W, 0.3)      
          #W = add_weight_noise(W, 1.0)      
          pass

        outputs = tf.nn.xw_plus_b(X, W, b, name="MakeOutputs")
        
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Bias", b)
        tf.summary.histogram("Outputs", outputs )
        
        outputs, index = harmonize_outputs(outputs, label_column_names)

        tf.summary.histogram("Outputs_harmonized", outputs )
        #outputs = tf.layers.batch_normalization(outputs_raw, axis=1, momentum=0.9, training=(mode == tf.estimator.ModeKeys.TRAIN))
        
      with tf.variable_scope("Output_Layer"):
        #X = tf.nn.batch_normalization(X, mean=0, variance=1, offset=None, scale=None, variance_epsilon=.000000001)
        tf.summary.histogram("Inputs", outputs)
        W = tf.get_variable(
            name="W",
            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            #24.1.2018 
            regularizer = GLOBAL_REGULARIZER ,
            shape=[output_size , 49])
        b = tf.get_variable(
            name="b",
            initializer=tf.random_normal_initializer(),
            shape=[49])
        #logits = tf.nn.xw_plus_b(tf.reshape(tf.stop_gradient(outputs), [-1, output_size]), W, b, name="MakeOutputs") # tf.stop_gradient 13.12.2017: no tf.stop_gradient
        # change: 5.1.2018
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)

        # 17.1.2018        
#        if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
#          W = add_weight_noise(W, 0.05)
          # 16.1.2018
          #W = add_weight_noise(W, 0.01)      
          #
          #W = add_weight_noise(W, 0.1)      

        logits = tf.nn.xw_plus_b(tf.reshape((outputs), [-1, output_size]), W, b, name="MakeOutputs") # tf.stop_gradient 13.12.2017: no tf.stop_gradient
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Bias", b)
        tf.summary.histogram("Outputs", logits )

      with tf.variable_scope("Softmax"):
        X = tf.reshape((outputs), [-1, output_size])
#        X = tf.layers.dense(X, 15, activation=tf.nn.relu, name="Layer1")
#        tf.summary.histogram("Layer1", X)
#        X = tf.nn.dropout(X, keep_prob=0.7)
        X = tf.layers.dense(X, 49, name="Logits", kernel_regularizer = l2_regularizer(scale=0.01),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        tf.summary.histogram("Outputs", X)
        sm_logits = X
        #tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.get_default_graph().get_tensor_by_name("Model/Softmax/Layer1/kernel/read:0"))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.get_default_graph().get_tensor_by_name("Model/Softmax/Logits/kernel/read:0"))
        
      # home_bias = tf.get_variable("home_bias", shape=[], dtype=tf.float32)
      home_bias = tf.constant(-1.0, name = "home_bias", shape=[], dtype=tf.float32)
      home_bias = tf.nn.sigmoid(home_bias)

      with tf.variable_scope("Ensemble"):
        X = tf.concat([outputs[0::2], outputs[1::2], logits[0::2], logits[1::2], sm_logits[0::2], sm_logits[1::2], 
                       f_date_round[0::2], 0*input_tensor[0::2, 2:] ], axis=1)
        X = tf.stop_gradient(X)
#        X = tf.layers.dense(X, 36, activation=tf.nn.relu, name="Layer1")
#        tf.summary.histogram("Layer1", X)
#        X = tf.nn.dropout(X, keep_prob=0.7)
        X = tf.layers.dense(X, 18, name="Logits", kernel_regularizer = l2_regularizer(scale=0.1),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        tf.summary.histogram("Outputs", X)
        ensemble_logits = X
#        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.get_default_graph().get_tensor_by_name("Model/Ensemble/Layer1/kernel/read:0"))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.get_default_graph().get_tensor_by_name("Model/Ensemble/Logits/kernel/read:0"))

      return outputs, logits, home_bias, sm_logits, ensemble_logits, index 
        
  def create_predictions(outputs, logits, t_is_home_bool, tc, use_max_points=False):
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
        p_poisson_2 = calc_poisson_prob(ev_goals_2)

        if use_max_points:
          a = tf.argmax(ev_points, axis=1)
        else:
          a = tf.argmax(p_pred_12, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
        predictions = {
          "p_marg_1":p_marg_1, 
          "p_marg_2":p_marg_2, 
          "outputs":outputs,
          "logits":logits,
          "p_pred_12":p_pred_12, 
          "ev_points":ev_points,  
          "pred":pred,
          "ev_goals_1":ev_goals_1,
          "ev_goals_2":ev_goals_2, 
          "p_poisson_1":p_poisson_1, 
          "p_poisson_2":p_poisson_2, 
        }
        return predictions
  
  def calc_joint_poisson_prob(lambda1, lambda2):
    logfactorial = [0.000000000000000, 0.000000000000000, 0.693147180559945, 1.791759469228055, 3.178053830347946, 4.787491742782046, 6.579251212010101]
    tc_1d7_goals_f = tf.constant([0,1,2,3,4,5,6], dtype=tf.float32, shape=[1,7])
    tc_1d7_logfactorial_f = tf.constant(logfactorial, dtype=tf.float32, shape=[1,7])

    lambda1 = tf.reshape(lambda1, [-1,1]) # make sure that rank=2
    lambda2 = tf.reshape(lambda2, [-1,1]) # make sure that rank=2
    loglambda1 = tf.log(lambda1)
    loglambda2 = tf.log(lambda2)
    x1 = tf.matmul(loglambda1, tc_1d7_goals_f) - tc_1d7_logfactorial_f # shape=(-1, 7)
    x2 = tf.matmul(loglambda2, tc_1d7_goals_f) - tc_1d7_logfactorial_f # shape=(-1, 7)
    x = tf.concat([x1,x2], axis=1) # shape=(-1, 14)
    matrix = [[1 if j==i1 or j==7+i2 else 0 for j in range(14)] for i1 in range(7) for i2 in range(7) ] # shape=(14, 49)
    t_matrix = tf.constant(matrix, dtype=tf.float32, shape=[49,14])
    t_matrix = tf.transpose(t_matrix)
    x = tf.matmul(x, t_matrix)
    x = tf.exp(x)
    normalization_factor = tf.exp(-lambda1-lambda2) 
    return x, normalization_factor 


  def apply_home_bias(prediction, home_bias):

    prediction = {k:tf.stop_gradient(v) for k,v in prediction.items()}
    
    ev_points = prediction["ev_points"]
    #p_pred_softmax = tf.nn.softmax(ev_points, dim=1)
    p_pred_softmax = ev_points / tf.reduce_sum(ev_points, axis=1, keep_dims=True)
    p_pred_blended = (1-home_bias) * p_pred_softmax + home_bias * prediction["p_pred_12"]
    a = tf.argmax(p_pred_blended, axis=1)
    pred_blended = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
    pred_blended= tf.cast(pred_blended, tf.int32)
    
    prediction["pred"] = pred_blended
    prediction["p_pred_12"] = p_pred_blended
    prediction["ev_points"] = p_pred_softmax*100
    #prediction["home_bias"] = home_bias
    
    return prediction
  
  def create_predictions_from_ev_goals(ev_goals_1, ev_goals_2, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
        
        ev_goals_1 = tf.stop_gradient(ev_goals_1)
        ev_goals_2 = tf.stop_gradient(ev_goals_2)
        
        p_pred_12, normalization_factor = calc_joint_poisson_prob(ev_goals_1, ev_goals_2)
        p_pred_12 *= normalization_factor # normalize to 1
        ev_points =  tf.where(t_is_home_bool,
                               tf.matmul(p_pred_12, tc_home_points_i),
                               tf.matmul(p_pred_12, tc_away_points_i))
        a = tf.argmax(ev_points, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        p_poisson_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        p_poisson_2  = tf.reduce_sum(p_pred_12_m, axis=1)

        predictions = {
          "p_marg_1":p_poisson_1, 
          "p_marg_2":p_poisson_2, 
          "p_pred_12":p_pred_12, 
          "ev_points":ev_points, 
          "pred":pred,
          "ev_goals_1":ev_goals_1,
          "ev_goals_2":ev_goals_2, 
          "p_poisson_1":p_poisson_1, 
          "p_poisson_2":p_poisson_2, 
        }
        return predictions
  
  def create_predictions_from_poisson(outputs, t_is_home_bool, tc):
    with tf.variable_scope("Prediction_Poisson"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
  
        outputs = tf.stop_gradient(outputs)  
        outputs = tf.exp(outputs)  
        p_loss = outputs[:,18]
        p_draw = outputs[:,19]
        p_win = outputs[:,20]
  
        z_score0_3 = outputs[:,24]
        z_score1_3 = outputs[:,25]
        z_score0_2 = outputs[:,26]
        z_score1_2 = outputs[:,27]
        z_score0_1 = outputs[:,28]
        z_score0_0 = outputs[:,29]
        z_score1_1 = outputs[:,30]
        z_score1_0 = outputs[:,31]
        z_score2_1 = outputs[:,32]
        z_score2_0 = outputs[:,33]
        z_score3_1 = outputs[:,34]
        z_score3_0 = outputs[:,35]
        z_score2_2 = outputs[:,36]
        z_score3_2 = outputs[:,37]
        z_score2_3 = outputs[:,38]

        scores_list = []
        for i in range(7):
          for j in range(7):
            v = tf.zeros_like(z_score0_0)
            if i==0 and j==0:
              v = z_score0_0 * p_draw 
            if i==1 and j==1:
              v = z_score1_1 * p_draw 
            if i==2 and j==2:
              v = z_score2_2 * p_draw 
            if i==0 and j==1:
              v = z_score0_1 * p_loss
            if i==0 and j==2:
              v = z_score0_2 * p_loss
            if i==0 and j==3:
              v = z_score0_3 * p_loss
            if i==1 and j==2:
              v = z_score1_2 * p_loss
            if i==1 and j==3:
              v = z_score1_3 * p_loss
            if i==2 and j==3:
              v = z_score2_3 * p_loss
            if i==1 and j==0:
              v = z_score1_0 * p_win
            if i==2 and j==0:
              v = z_score2_0 * p_win
            if i==3 and j==0:
              v = z_score3_0 * p_win
            if i==2 and j==1:
              v = z_score2_1 * p_win
            if i==3 and j==1:
              v = z_score3_1 * p_win
            if i==3 and j==2:
              v = z_score3_2 * p_win
            scores_list.extend([v])
        
        p_pred_12 = tf.stack(scores_list, axis=1)
        
        p_pred_12 = p_pred_12 / tf.reduce_sum(p_pred_12, axis=1, keep_dims=True)
      
        ev_points =  tf.where(t_is_home_bool,
                               tf.matmul(p_pred_12, tc_home_points_i),
                               tf.matmul(p_pred_12, tc_away_points_i))
        
        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
        ev_goals_1 = tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
        ev_goals_2 = tf.matmul(p_marg_2, tc_1d1_goals_f)[:,0]

        p_poisson_1 = calc_poisson_prob(ev_goals_1)
        p_poisson_2 = calc_poisson_prob(ev_goals_2)

        a = tf.argmax(ev_points, axis=1)
        pred_ev = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred_ev = tf.cast(pred_ev, tf.int32)

        a = tf.argmax(p_pred_12, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)

        a = tf.where(tf.equal(a // 7, tf.mod(a, 7)), tf.where(t_is_home_bool, a+1, a+7), a) # if draw: away team wins with an extra goal

        predictions = {
          "p_marg_1":p_marg_1, 
          "p_marg_2":p_marg_2, 
          "outputs":outputs,
          "p_pred_12":p_pred_12, 
          "ev_points":ev_points,  
          "pred":pred,
          "pred_ev":pred_ev,
          "ev_goals_1":ev_goals_1,
          "ev_goals_2":ev_goals_2, 
          "p_poisson_1":p_poisson_1, 
          "p_poisson_2":p_poisson_2, 
        }
        return predictions

  def create_ensemble_predictions(ensemble_logits, predictions, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
#        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        p_ensemble = tf.nn.softmax(tf.reshape(ensemble_logits, [-1, 18]))
        
        prefix_list = ["p1/", "p2/", "p3/", "p4/", "p5/", "p6/", "sp/", "sm/", "smhb/"]
        t_home_preds = tf.stack([predictions[p+"pred"][0::2] for p in prefix_list], axis=1)
        t_away_preds = tf.stack([predictions[p+"pred"][1::2] for p in prefix_list], axis=1)
        t_away_preds = tf.stack([t_away_preds[:,:,1], t_away_preds[:,:,0]], axis=2) # swap home and away goals
        t_preds = tf.concat([t_home_preds, t_away_preds], axis=1)
        
        # expand p_ensemble and t_preds from half lenght to full length - else TF will complain in PREDICT mode
        index = tf.range(0, 2 * tf.shape(p_ensemble)[0])
        index = index // 2 # every index element repeats twice
        p_ensemble = tf.gather(p_ensemble, index)
        t_preds = tf.gather(t_preds, index)
        
        selected_strategy = tf.argmax(p_ensemble, axis=1, output_type=tf.int32)
        index2 = tf.stack([tf.range(0, tf.shape(selected_strategy)[0]), selected_strategy], axis=1)
        pred = tf.gather_nd(t_preds, index2)
        
        predictions = {
          "ensemble_logits":tf.gather(ensemble_logits, index),
          "selected_strategy":selected_strategy, 
          "pred":pred,
          "p_ensemble":p_ensemble, 
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

  def create_model_regularization_metrics(eval_metric_ops, predictions, labels):
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
      W = tf.get_default_graph().get_tensor_by_name("Model/Mapping_Layer/WM:0")
      reg_layerMapping_L2 = tf.reduce_mean(tf.square(W), axis=1)
      reg_layerMapping_L1 = tf.reduce_mean(tf.abs(W), axis=1)

  #    # Weight penalty for Output Layer 
      W = tf.get_default_graph().get_tensor_by_name("Model/Output_Layer/W:0")
      reg_layerOut_L2 = tf.reduce_mean(tf.square(W), axis=1)
      reg_layerOut_L1 = tf.reduce_mean(tf.abs(W), axis=1)
      
      print(tf.get_collection(tf.GraphKeys.WEIGHTS))
      metrics = {}
      for w in tf.get_collection(tf.GraphKeys.WEIGHTS):
        wname = w.name[6:-2]
        tf.summary.histogram(wname, w)
        metrics["regularization/"+wname+"_L1"] = tf.metrics.mean(tf.abs(w))
        metrics["regularization/"+wname+"_L2"] = tf.metrics.mean(tf.square(w))

#      with tf.variable_scope("Softmax"):
#        X = outputs
#        X = tf.layers.dense(X, 15, activation=tf.nn.relu, name="Layer1")
#        tf.summary.histogram("Layer1", X)
#        X = tf.nn.dropout(X, keep_prob=0.7)
#        X = tf.layers.dense(X, 49, name="Logits")
#        tf.summary.histogram("Outputs", X)
      #l_regularization = tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)
  #    l_regularization = tf.reduce_mean(0.001*reg_layer1_L1+1000*reg_layer1_L2)
  #    l_regularization += tf.reduce_mean(0.001*reg_layer2_L1+1000*reg_layer2_L2)
      #l_regularization += tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)

      
      if True:
        K1 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/read:0")
        K2 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/read:0")
        K3 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel/read:0")
        K4 = tf.get_default_graph().get_tensor_by_name("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel/read:0")
        reg_Gates0_L2 = tf.reduce_mean(tf.square(K1), axis=1)
        reg_Candidates0_L2 = tf.reduce_mean(tf.square(K2), axis=1)      
        reg_Gates1_L2 = tf.reduce_mean(tf.square(K3), axis=1)
        reg_Candidates1_L2 = tf.reduce_mean(tf.square(K4), axis=1)      
        l_regularization = tf.reduce_mean(reg_Gates0_L2)
        l_regularization += tf.reduce_mean(reg_Candidates0_L2)
        l_regularization += tf.reduce_mean(reg_Gates1_L2)
        l_regularization += tf.reduce_mean(reg_Candidates1_L2)

#      K12 = tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/gates/kernel/read:0")
#      K22 = tf.get_default_graph().get_tensor_by_name("Model/RNN_2/rnn/gru_cell/candidate/kernel/read:0")
#      reg_Gates_L22 = tf.reduce_mean(tf.square(K12), axis=1)
#      reg_Candidates_L22 = tf.reduce_mean(tf.square(K22), axis=1)      
#      l_regularization += tf.reduce_mean(reg_Gates_L22+reg_Candidates_L22)

#      #"summary/p3/metric_is_tendency"
#      #"Evaluation/p3/Metrics/metric_is_tendency"
#      def add_mapping_weight_noise(name, stddev_factor, contraction_factor):  
#        X = tf.get_default_graph().get_tensor_by_name(name)
#        with tf.variable_scope("Mapping_Weight_Noise", reuse=tf.AUTO_REUSE):
#          mean, variance = tf.nn.moments(X, axes=[1], keep_dims=True)
#          stddev = tf.sqrt(variance)
#          noise = tf.random_normal(shape=tf.shape(X), mean=-contraction_factor*X, stddev=stddev * stddev_factor, dtype=tf.float32) 
#          tf.summary.scalar(name[6:-2]+"_noise", stddev_factor)
#          tf.summary.histogram(name[6:-2]+"_stddev", stddev)
#        return X+noise
#
#      def add_weight_noise(name, noise_factor, contraction_factor):
#        W = tf.get_default_graph().get_tensor_by_name(name)
#        tf.summary.histogram(name[6:-2], W)
#        W_stddev = tf.sqrt(tf.reduce_mean(tf.square(W)))
#        tf.summary.scalar(name[6:-2]+"_stddev", W_stddev)
#        tf.summary.scalar(name[6:-2]+"_noise", noise_factor)
#        #return tf.assign(W, W+tf.random_normal(shape=tf.shape(W), mean=-0.01*noise_factor*W, stddev=W*noise_factor, dtype=tf.float32)) 
#        return tf.assign(W, W+tf.random_normal(shape=tf.shape(W), mean=-contraction_factor*W, stddev=W_stddev*noise_factor, dtype=tf.float32)) 
#        
      prefix_list = ["p1/", "p2/", "p3/", "p4/", "p5/", "p6/"] #, "sp/", "sm/", "smhb/"]
      l_tendencies = [tf.reduce_mean(predictions[p + "is_tendency"]) for p in prefix_list]
      t_tendencies = tf.stack(l_tendencies)
      current_tendency = tf.reduce_max(t_tendencies)
      # current_tendency = tf.maximum(tf.reduce_mean(predictions["p6/is_tendency"]), tf.reduce_mean(predictions["p3/is_tendency"]))
      tf.summary.scalar("current_tendency", current_tendency)
      #base_noise_factor = 0.01 * tf.exp((current_tendency - 0.45)/0.05)
      # 30.1.2018
      base_noise_factor = 0.002 * tf.exp((current_tendency - 0.40)/0.06)

      is_win  = labels[:,18]
      is_loss = labels[:,20]
      pred_win  = tf.exp(predictions["outputs_poisson"][:,18])
      pred_loss = tf.exp(predictions["outputs_poisson"][:,20])
      
      current_winloss_corr = corrcoef(is_win, pred_win) + corrcoef(is_loss, pred_loss)
      #eval_metric_ops["poisson/Win"][1]+eval_metric_ops["poisson/Loss"][1]
      tf.summary.scalar("current_winloss_corr", current_winloss_corr)
      
      #noise_factor = 0.015
      #noise_factor = 0.01
      #noise_factor = 0.003 + 0.4 * tf.nn.relu(current_tendency - 0.50)
      rnn_noise_factor = base_noise_factor + 0.01 * 0.3*tf.exp((current_winloss_corr - 0.75)/0.05)
      #tf.summary.scalar("rnn_noise_factor", rnn_noise_factor)
      contraction_factor = 0.2 * rnn_noise_factor
      #contraction_factor = 0.5 * noise_factor

      current_diff_corr = eval_metric_ops["p1/metric_cor_diff"][1]
      tf.summary.scalar("current_diff_corr", current_diff_corr)
      #noise_factor_mapping = base_noise_factor + 0.001 + 0.01 * tf.exp((current_diff_corr - 0.45)/0.05)
      # 30.1.2018
      noise_factor_mapping = 0.03 * tf.exp((current_diff_corr - 0.45)/0.05)

      contraction_factor_mapping = 0.01 * noise_factor_mapping 
      #tf.summary.scalar("mapping_noise_factor", noise_factor_mapping)

      ens_tendency = tf.reduce_mean(predictions["ens/is_tendency"])
      tf.summary.scalar("ens_tendency", ens_tendency)
      ens_noise = 0.01 * tf.exp((ens_tendency - 0.45)/0.05)
      #tf.summary.scalar("ens_noise", ens_noise)
      ens_contraction_factor = 0.9 * ens_noise
      
      sm_tendency = tf.maximum(tf.reduce_mean(predictions["sm/is_tendency"]), tf.reduce_mean(predictions["smhb/is_tendency"]))
      tf.summary.scalar("sm_tendency", sm_tendency)
      #sm_noise = 0.01 * tf.exp((sm_tendency - 0.45)/0.05)
      # 31.01.2018
      sm_noise = 0.01 * tf.exp((sm_tendency - 0.45)/0.05)

      sm_full = tf.reduce_mean(predictions["sm/is_full"])
      tf.summary.scalar("sm_full", sm_full)
      sm_noise += 0.01 * tf.exp((sm_full - 0.12)/0.01)

      sm_diff = tf.reduce_mean(predictions["sm/is_diff"])
      tf.summary.scalar("sm_diff", sm_diff)
      sm_noise += 0.01 * tf.exp((sm_diff - 0.20)/0.02)

      sp_tendency = tf.reduce_mean(predictions["sp/is_tendency"])
      tf.summary.scalar("sp_tendency", sm_tendency)
      #sp_noise = 0.01 * tf.exp((sp_tendency - 0.45)/0.05)
      # 31.01.2018
      sp_noise = 0.01 * tf.exp((sp_tendency - 0.45)/0.05)
      
      pred_gs = predictions["outputs_poisson"][::2, 0]
      gs_mean, gs_variance = tf.nn.moments(pred_gs, axes=[0])
      # variance should be in the order of 1.0
      #base_noise_factor /= tf.sqrt(tf.minimum(gs_variance, 0.1))
      #base_noise_factor *= 0.01
      #base_noise_factor *= 0.1
      tf.summary.scalar("gs_variance", gs_variance)
      tf.summary.scalar("base_noise_factor", base_noise_factor)
      
      def apply_noise_old(axes, name, stddev_factor, contraction_factor):  
        X = tf.get_default_graph().get_tensor_by_name(name)
        with tf.variable_scope("noise", reuse=tf.AUTO_REUSE):
          tf.summary.scalar(name[6:-2]+"_noise", stddev_factor)
          mean, variance = tf.nn.moments(X, axes=axes, keep_dims=True)
          stddev = tf.sqrt(variance)
          noise = tf.random_normal(shape=tf.shape(X), mean=-contraction_factor*X, stddev=stddev * stddev_factor, dtype=tf.float32) 
          if len(X.shape)==len(axes):
            tf.summary.scalar(name[6:-2]+"_stddev", tf.squeeze(stddev))
          else:
            tf.summary.histogram(name[6:-2]+"_stddev", stddev)
        return tf.assign(X, X+noise)

      # This function neither changes means nor variance for a given SNR
      def apply_noise(axes, name, noise_factor, _ignored=None ):
        SNR = 1.0 / noise_factor # = signal_to_noise_ratio
        X = tf.get_default_graph().get_tensor_by_name(name)
        a1 = tf.sqrt(1-1/(2+tf.pow(SNR,2)))
        #a2 = 1-tf.sqrt((tf.square(SNR)+1) / (tf.square(SNR)+2))
        a2 = 1.0 - a1
        #b2 = tf.sqrt((2-a2)/a2)
        b2 = tf.sqrt((1+a1)/(a2+1e-10))
        
        with tf.variable_scope("noise", reuse=tf.AUTO_REUSE):
#          tf.summary.scalar(name[6:-2]+"_a1", tf.squeeze(a1))
#          tf.summary.scalar(name[6:-2]+"_a2", tf.squeeze(a2))
#          tf.summary.scalar(name[6:-2]+"_b2", tf.squeeze(b2))
          tf.summary.scalar(name[6:-2]+"_SNR", SNR)
          mean, variance = tf.nn.moments(X, axes=axes, keep_dims=True)
          a1 *= tf.cast(tf.less(variance, 1.0), tf.float32) # hard reset to noise only if variance too large
          a2 += tf.cast(tf.less(variance, 1.0e-6), tf.float32) # add noise if variance too small
          stddev = tf.sqrt(variance)
          shrink_factor = tf.exp(tf.minimum(10.0-variance, 0.0))
          noise = tf.random_normal(shape=tf.shape(X), mean=mean*shrink_factor, stddev=stddev * b2, dtype=tf.float32) 
          if len(X.shape)==len(axes):
            tf.summary.scalar(name[6:-2]+"_stddev", tf.squeeze(stddev))
          else:
            tf.summary.histogram(name[6:-2]+"_stddev", stddev)
        return tf.assign(X, a1*X+a2*noise)

      # This function neither changes means nor variance for a given SNR
      def apply_multiplicative_noise(name, noise_factor):
        X = tf.get_default_graph().get_tensor_by_name(name)
        X_reinitialize = tf.get_default_graph().get_tensor_by_name(name[:-2]+"/Assign:0")
        with tf.variable_scope("mult_noise", reuse=tf.AUTO_REUSE):
#          tf.summary.scalar(name[6:-2]+"_a1", tf.squeeze(a1))
#          tf.summary.scalar(name[6:-2]+"_a2", tf.squeeze(a2))
#          tf.summary.scalar(name[6:-2]+"_b2", tf.squeeze(b2))
          tf.summary.scalar(name[6:-2]+"_noise", noise_factor)
          noise = tf.random_normal(shape=tf.shape(X), dtype=tf.float32) 
          # weight reset if needed
          mean = tf.reduce_mean(X)
          variance = tf.reduce_mean(tf.square(X-mean))
          stddev = tf.sqrt(variance)
          initial_variance = tf.sqrt(2. / tf.cast(tf.shape(X)[0] + tf.shape(X)[1], tf.float32))
          too_high = tf.greater(variance, 1.0) # hard reset to noise only if variance too large
          too_low  = tf.less(variance, 0.0001*initial_variance) # add noise if variance too small
          shrink_factor = tf.exp(tf.minimum(10.0-variance, 0.0))
          tf.summary.scalar(name[6:-2]+"_stddev", tf.squeeze(stddev))
        return tf.assign(X, X*(shrink_factor+noise_factor*noise))
        #return tf.cond(tf.logical_or(too_high, too_low), lambda: X_reinitialize, lambda: tf.assign(X, X*(1.0+noise_factor*noise)))    

      def apply_gradient_noise(name, noise_factor):
        X = tf.get_default_graph().get_tensor_by_name(name)
        with tf.variable_scope("grad_noise", reuse=tf.AUTO_REUSE):
          tf.summary.scalar(name[6:-2]+"_noise", noise_factor)
          noise = tf.random_normal(shape=tf.shape(X), dtype=tf.float32) 
          # weight reset if needed
          mean = tf.reduce_mean(X)
          variance = tf.reduce_mean(tf.square(X-mean))
          stddev = tf.sqrt(variance)
          tf.summary.scalar(name[6:-2]+"_stddev", tf.squeeze(stddev))
          shrink_factor = tf.exp(tf.minimum(10.0-variance, 0.0))
          #reg_loss = tf.losses.mean_squared_error(X, X*tf.stop_gradient(shrink_factor+100*noise_factor*noise)) / tf.stop_gradient(variance)
          reg_loss = tf.losses.mean_squared_error(X, X*tf.stop_gradient(shrink_factor+1000*noise_factor*noise)) 
          reg_loss = tf.identity(reg_loss, name=name[6:-2]+"_loss")
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg_loss)
          
#          return tf.losses.mean_squared_error(X, X*(shrink_factor+100*noise_factor*noise), 
#                                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) # this node will be evaluated in main loss function

      if False: #"Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates"
        make_noise_op = tf.group(
           apply_noise([0,1], "Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel:0", rnn_noise_factor, contraction_factor), 
           apply_noise([0,1], "Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel:0", rnn_noise_factor, contraction_factor), 
           apply_noise([0,1], "Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel:0", rnn_noise_factor*0.01, contraction_factor*0.1), 
           apply_noise([0,1], "Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel:0", rnn_noise_factor*0.01, contraction_factor*0.1), 
#           apply_noise([0], "Model/Mapping_Layer/WM:0", 0.5*noise_factor_mapping, contraction_factor_mapping), 
#           apply_noise([1], "Model/Ensemble/Logits/kernel:0", ens_noise, ens_contraction_factor), 
#           apply_noise([0,1], "Model/Mapping_Layer/WM:0", base_noise_factor*0.00005, base_noise_factor*0.005), 
#           apply_noise([1], "Model/Softmax/Logits/kernel:0", sm_noise, sm_noise*1.001), 
#           apply_noise([1], "Model/Output_Layer/W:0", sp_noise, sp_noise*1.001), 
#           apply_noise([0], "Model/Layer2/Dense/kernel:0", base_noise_factor*0.1, base_noise_factor*0.0005), 
#           apply_noise([0,1], "Model/Layer1/Dense/kernel:0", base_noise_factor*0.01, base_noise_factor*0.005), 

           apply_noise([0,1], "Model/Layer1/Dense/kernel:0", base_noise_factor*1.0, base_noise_factor*0.005), 
           apply_noise([0,1], "Model/Layer2/Dense/kernel:0", base_noise_factor*0.7, base_noise_factor*0.0005), 
#           apply_noise([0,1], "Model/Mapping_Layer/WM:0", base_noise_factor*0.3, contraction_factor_mapping), 
#           apply_noise([0,1], "Model/Softmax/Logits/kernel:0", sm_noise, sm_noise*0.101), 
#           apply_noise([0,1], "Model/Output_Layer/W:0", sp_noise, sp_noise*0.101), 
#           apply_noise([0,1], "Model/Ensemble/Logits/kernel:0", ens_noise, ens_contraction_factor), 
           apply_multiplicative_noise("Model/Mapping_Layer/WM:0", base_noise_factor*0.3), 
           apply_multiplicative_noise("Model/Softmax/Logits/kernel:0", sm_noise), 
           apply_multiplicative_noise("Model/Output_Layer/W:0", sp_noise), 
           apply_multiplicative_noise("Model/Ensemble/Logits/kernel:0", ens_noise), 

           #add_weight_noise("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel:0"), 
           #add_weight_noise("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel:0"), 
#                               add_weight_noise("Model/RNN_2/rnn/gru_cell/gates/kernel:0"), 
#                               add_weight_noise("Model/RNN_2/rnn/gru_cell/candidate/kernel:0"), 
#                               add_weight_noise("Model/RNN_12/rnn/gru_cell/gates/kernel:0"), 
#                               add_weight_noise("Model/RNN_12/rnn/gru_cell/candidate/kernel:0")
           ) 
      else:
        make_noise_op = tf.no_op()
        l_regularization = tf.reduce_mean(reg_layerOut_L1)
        apply_gradient_noise("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel:0", rnn_noise_factor*0.1), 
        apply_gradient_noise("Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel:0", rnn_noise_factor*0.1), 
        apply_gradient_noise("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel:0", rnn_noise_factor*0.1), 
        apply_gradient_noise("Model/RNN_1/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel:0", rnn_noise_factor*0.1), 
        apply_gradient_noise("Model/Layer1/Dense/kernel:0", base_noise_factor*10.0), 
        apply_gradient_noise("Model/Layer2/Dense/kernel:0", base_noise_factor*7.7), 
        apply_gradient_noise("Model/Mapping_Layer/WM:0", base_noise_factor*30.3), 
        apply_gradient_noise("Model/Softmax/Logits/kernel:0", sm_noise*0.3), 
        apply_gradient_noise("Model/Output_Layer/W:0", sp_noise), 
        apply_gradient_noise("Model/Ensemble/Logits/kernel:0", ens_noise), 
 
                         
      reg_eval_metric_ops = {
            "regularization/l_regularization": tf.metrics.mean(l_regularization),
            "summary/l_regularization": tf.metrics.mean(l_regularization),
  #        , "reg_layer1_L1": tf.metrics.mean(reg_layer1_L1)
  #        , "reg_layer1_L2": tf.metrics.mean(reg_layer1_L2)
  #        , "reg_layer2_L1": tf.metrics.mean(reg_layer2_L1)
  #        , "reg_layer2_L2": tf.metrics.mean(reg_layer2_L2)
          "regularization/reg_layerMapping_L1": tf.metrics.mean(reg_layerMapping_L1)
          , "regularization/reg_layerMapping_L2": tf.metrics.mean(reg_layerMapping_L2)
          , "regularization/reg_layerOut_L1": tf.metrics.mean(reg_layerOut_L1)
          , "regularization/reg_layerOut_L2": tf.metrics.mean(reg_layerOut_L2)
#          , "reg_Kernel_L2": tf.metrics.mean(reg_Kernel_L2)
          , "regularization/reg_Gates_L2_0": tf.metrics.mean(reg_Gates0_L2)
          , "regularization/reg_Candidates_L2_0": tf.metrics.mean(reg_Candidates0_L2)
          , "regularization/reg_Gates_L2_1": tf.metrics.mean(reg_Gates1_L2)
          , "regularization/reg_Candidates_L2_1": tf.metrics.mean(reg_Candidates1_L2)
      }
      reg_eval_metric_ops .update(metrics)
      
    return l_regularization, reg_eval_metric_ops, make_noise_op

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
    
  def create_poisson_correlation_metrics(outputs, t_labels):
    metrics={}
    for i,col in enumerate(label_column_names):
      metrics["poisson/"+col.replace(":", "_")]=tf.metrics.mean(corrcoef(outputs[:,i], t_labels[:,i]))
    return metrics
  
  def create_losses_RNN(outputs, logits, sm_logits, t_labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool, eval_metric_ops):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

#    if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
#      labels, labels2 = add_noise(labels, labels2, 0.01)      

    with tf.variable_scope("Prediction"):
      labels_float  = t_labels[:,0]
      labels_float2 = t_labels[:,1]
      labels = tf.cast(labels_float, tf.int32)
      labels2 = tf.cast(labels_float2, tf.int32)
      # reduce to 6 goals max. for training
      gs = tf.minimum(labels,6)
      gc = tf.minimum(labels2,6)
  
    with tf.variable_scope("Losses"):
      match_date = features["newgame"][:,4]
      sequence_length = features['newgame'][:,0]
      t_weight = tf.exp(0.5*match_date) * (sequence_length + 0.05) # sequence_length ranges from 0 to 1 - depending on the number of prior matches of team 1
      l_loglike_poisson = tf.expand_dims(t_weight, axis=1) * tf.nn.log_poisson_loss(targets=t_labels, log_input=outputs)
      poisson_column_weights = tf.ones(shape=[1, t_labels.shape[1]])
      poisson_column_weights = tf.concat([
          poisson_column_weights[:,0:4] *3,
          poisson_column_weights[:,4:16],
          poisson_column_weights[:,16:20] *3,
          poisson_column_weights[:,20:],
          ], axis=1)
      l_loglike_poisson *= poisson_column_weights
    
      p_full    = tf.one_hot(gs*7+gc, 49)
      l_full    = t_weight * tf.nn.softmax_cross_entropy_with_logits(labels=p_full, logits=tf.reshape(logits, [-1,49]))
      l_softmax = t_weight * tf.nn.softmax_cross_entropy_with_logits(labels=p_full, logits=tf.reshape(sm_logits, [-1,49]))

      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    
      
#      z = tf.cast(tf.zeros_like(gs), tf.float32)
#      pt_softpoints = tf.reduce_sum(predictions["sp/p_pred_12"] * achievable_points_mask, axis=1)
      pt_log_softpoints = t_weight * tf.reduce_sum(tf.log(predictions["sp/p_pred_12"]+0.0002) * achievable_points_mask, axis=1)
      
#      p3_softpoints = t_weight * tf.reduce_sum(tf.pow(predictions["p3/p_pred_12"],3) * achievable_points_mask, axis=1)
#      p5_softpoints = t_weight * tf.reduce_sum(tf.pow(predictions["p5/p_pred_12"],3) * achievable_points_mask, axis=1)
#      p6_softpoints = t_weight * tf.reduce_sum(tf.pow(predictions["p6/p_pred_12"],3) * achievable_points_mask, axis=1)
#      smhb_softpoints = t_weight * tf.reduce_sum(tf.pow(predictions["smhb/p_pred_12"],3) * achievable_points_mask, axis=1)


#      pt_softpoints_capped = tf.reduce_sum(tf.minimum(predictions["p_pred_12"], 0.15) * achievable_points_mask, axis=1)
#      l_softpoints = -tf.log(5+pt_softpoints)*1
#      capped_probs = tf.minimum(predictions["sp/p_pred_12"], 0.10)                    
#      pt_softpoints_capped = tf.reduce_sum(capped_probs * achievable_points_mask, axis=1)
#      l_softpoints = -pt_softpoints_capped
#      l_softpoints *= tf.where( t_is_home_loss_bool, z+2, z+1)
#      loss -= tf.reduce_mean(tf.where(t_is_draw_bool, 
#                                      0.0*tf.zeros_like(pt_softpoints), 
#                                      0.1*tf.ones_like(pt_softpoints)) *
#                   (0*tf.log(pt_softpoints)+(pt_softpoints)))

      
      l_regularization, reg_eval_metric_ops, make_noise_op = create_model_regularization_metrics(eval_metric_ops, predictions, t_labels)
      
#       loss = l_full # + l_loglike_ev_goals1 + l_loglike_ev_goals2 
#      loss *= tf.where( t_is_home_loss_bool, z+2, z+1) # away win = 4 x
#      loss *= tf.where( t_is_home_win_bool, z+2, z+1)  # home win = 2 x, draw = 1 x
      l_loglike_poisson = tf.reduce_sum(l_loglike_poisson, axis=1)
      loss = tf.reduce_mean(l_loglike_poisson)

      loss += tf.reduce_mean(l_softmax[0::2])

      #loss += 1*tf.reduce_mean(l_full)  
      # 5.1.2018
      #loss += 0.00*tf.reduce_mean(l_full)  

      # loss += tf.reduce_mean(l_softpoints) 
      
      #loss -= 0.5*tf.reduce_mean(pt_log_softpoints)
      #5.1.2018
      #loss -= 0.05*tf.reduce_mean(pt_log_softpoints)
      #7.1.2018
      #loss -= 0.5*tf.reduce_mean(pt_log_softpoints)
      #22.1.2018
      loss -= 0.1*tf.reduce_mean(pt_log_softpoints)
      #24.01.2018 
      #loss -= 0.5*tf.reduce_mean(pt_log_softpoints[0::2])


      #6.1.2018
      #loss += 10*l_regularization

#      loss -= tf.reduce_mean(p3_softpoints)
#      loss -= tf.reduce_mean(p5_softpoints)
#      loss -= tf.reduce_mean(p6_softpoints)
#      loss -= tf.reduce_mean(smhb_softpoints)
      
      #print(tf.get_collection(tf.GraphKeys.WEIGHTS))
      reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      #reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
      print(reg_variables)
      if (reg_variables):
        #reg_term = tf.contrib.layers.apply_regularization(GLOBAL_REGULARIZER, reg_variables)
        for r in reg_variables:
            tf.summary.scalar("Regularization/"+r.name[6:-2], r)
            reg_eval_metric_ops["Regularization/"+r.name[6:-2]] = tf.metrics.mean(r)
            loss += r
#        #print(reg_term)
#        tf.summary.scalar("Regularization/reg_term", reg_term)
#        loss += reg_term
#      for r in reg_variables:
#        tf.summary.scalar("regularization/"+r.name[6:-2], tf.contrib.layers.apply_regularization(GLOBAL_REGULARIZER, r))
#        
      
      ### ensemble
      prefix_list = ["p1/", "p2/", "p3/", "p4/", "p5/", "p6/", "sp/", "sm/", "smhb/"]
      t_home_points = tf.stack([predictions[p+"z_points"][0::2] for p in prefix_list], axis=1)
      t_away_points = tf.stack([predictions[p+"z_points"][1::2] for p in prefix_list], axis=1)
      t_points = tf.concat([t_home_points, t_away_points], axis=1)
      t_ensemble_softpoints = t_weight[::2] * tf.reduce_sum(predictions["ens/p_ensemble"][0::2]*t_points, axis=1)

      #loss -= tf.reduce_mean(30*t_ensemble_softpoints)
      # 30.1.2018
      loss -= tf.reduce_mean(10*t_ensemble_softpoints)
      
      
      tf.summary.scalar("loss", loss)
      
      eval_metric_ops = {
          "losses/l_loglike_poisson": tf.metrics.mean(l_loglike_poisson)
          , "losses/l_full": tf.metrics.mean(l_full)
          , "losses/l_softmax": tf.metrics.mean(l_softmax)
          , "losses/l_loss": tf.metrics.mean(loss)
          , "summary/l_loss": tf.metrics.mean(loss)
#          , "losses/l_p3_softpoints": tf.metrics.mean(p3_softpoints)
#          , "losses/l_p5_softpoints": tf.metrics.mean(p5_softpoints)
#          , "losses/l_p6_softpoints": tf.metrics.mean(p6_softpoints)
#          , "losses/l_smhb_softpoints": tf.metrics.mean(smhb_softpoints)
          , "losses/l_ens_softpoints": tf.metrics.mean(t_ensemble_softpoints)
#          , "losses/l_regterm": tf.metrics.mean(reg_term)
      }
      eval_metric_ops.update(reg_eval_metric_ops)
      
    return eval_metric_ops, loss, make_noise_op

  def create_ensemble_result_metrics(predictions, labels, labels2, achievable_points_mask, tc):      
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    prefix="ens"
    pGS = predictions[prefix+"/pred"][0::2, 0]
    pGC = predictions[prefix+"/pred"][0::2, 1]
    labels = labels[0::2]
    labels2 = labels2[0::2]
    achievable_points_mask = achievable_points_mask[0::2]
    
    with tf.variable_scope(prefix+"Metrics"):
      point_summary = calc_points(pGS,pGC, labels, labels2, True)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary

      predictions["ens/z_points"] = pt_actual_points
      predictions["ens/is_tendency"] = tf.cast(is_tendency, tf.float32)

      pred_draw = tf.cast(tf.equal(pGS,pGC), tf.float32)
      pred_home_win = tf.cast(tf.greater(pGS, pGC)  , tf.float32)
      pred_away_win = tf.cast(tf.less(pGS, pGC) , tf.float32)
      
      
      eval_metric_ops = {
           "z_points": tf.metrics.mean(pt_actual_points, name="z_points")
          , "metric_is_tendency": tf.metrics.mean(is_tendency, name="metric_is_tendency")
          , "metric_is_diff": tf.metrics.mean(is_diff, name="metric_is_diff")
          , "metric_is_full": tf.metrics.mean(is_full, name="metric_is_full")
          , "metric_pred_home_win": tf.metrics.mean(pred_home_win, name="metric_pred_home_win")
          , "metric_pred_away_win": tf.metrics.mean(pred_away_win, name="metric_pred_away_win")
          , "metric_pred_draw": tf.metrics.mean(pred_draw, name="metric_pred_draw")
#          , "points/pt_draw_points": tf.metrics.mean(pt_draw_points)
#          , "points/pt_home_win_points": tf.metrics.mean(pt_home_win_points)
#          , "points/pt_away_loss_points": tf.metrics.mean(pt_away_loss_points)
#          , "points/pt_away_win_points": tf.metrics.mean(pt_away_win_points)
#          , "points/pt_home_loss_points": tf.metrics.mean(pt_home_loss_points)
      }

      prefix_list = ["p1/", "p2/", "p3/", "p4/", "p5/", "p6/", "sp/", "sm/", "smhb/"]
      prefix_list = [p[:-1] for p in prefix_list]
      prefix_list = [p+"_home" for p in prefix_list]+[p+"_away" for p in prefix_list]
      
#      prefix_list = ["p1_home", "p1_away", "p2_home", "p2_away", 
#                     "p3_home", "p3_away", "p4_home", "p4_away", 
#                     "p5_home", "p5_away", "p6_home", "p6_away", 
#                     "sp_home", "sp_away", 
#                     "sm_home", "sm_away", "smhb_home", "smhb_away"] 

      for i,p in enumerate(prefix_list):
        eval_metric_ops[p] = tf.metrics.mean(
            tf.cast(tf.equal(predictions["ens/selected_strategy"],i), tf.float32))

    return eval_metric_ops

  def create_result_metrics(prefix, predictions, labels, labels2, t_is_home_bool, achievable_points_mask, tc):      
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    pGS = predictions[prefix+"/pred"][:,0]
    pGC = predictions[prefix+"/pred"][:,1]
    probs = predictions[prefix+"/p_pred_12"]
    ev_goals_1 = predictions[prefix+"/ev_goals_1"]
    ev_goals_2 = predictions[prefix+"/ev_goals_2"]
    
    with tf.variable_scope(prefix+"Metrics"):
      point_summary = calc_points(pGS,pGC, labels, labels2, t_is_home_bool)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary

      predictions[prefix+"/z_points"] = pt_actual_points
      predictions[prefix+"/is_tendency"] = tf.cast(is_tendency, tf.float32)
      predictions[prefix+"/is_full"] = tf.cast(is_full, tf.float32)
      predictions[prefix+"/is_diff"] = tf.cast(is_diff, tf.float32)
      
      pred_draw = tf.cast(tf.equal(pGS,pGC), tf.float32)
      pred_home_win = tf.cast((tf.greater(pGS, pGC) & t_is_home_bool) | (tf.less(pGS, pGC) & tf.logical_not(t_is_home_bool)), tf.float32)
      pred_away_win = tf.cast((tf.less(pGS, pGC) & t_is_home_bool) | (tf.greater(pGS, pGC) & tf.logical_not(t_is_home_bool)), tf.float32)

      labels_float  = tf.cast(labels, tf.float32)
      labels_float2 = tf.cast(labels2, tf.float32)
      
      l_diff_ev_goals_L1 = tf.abs(labels_float-labels_float2-( ev_goals_1-ev_goals_2))

      pt_log_softpoints = tf.reduce_sum(tf.log(probs+0.2) * achievable_points_mask, axis=1)
      pt_softpoints = tf.reduce_sum(probs * achievable_points_mask, axis=1)

      capped_probs = tf.minimum(probs, 0.10)                    
      pt_softpoints_capped = tf.reduce_sum(capped_probs * achievable_points_mask, axis=1)

      eval_metric_ops = {
           "z_points": tf.metrics.mean(pt_actual_points, name="z_points")
          , "metric_is_tendency": tf.metrics.mean(is_tendency, name="metric_is_tendency")
          , "metric_is_diff": tf.metrics.mean(is_diff, name="metric_is_diff")
          , "metric_is_full": tf.metrics.mean(is_full, name="metric_is_full")
          , "metric_pred_home_win": tf.metrics.mean(pred_home_win, name="metric_pred_home_win")
          , "metric_pred_away_win": tf.metrics.mean(pred_away_win, name="metric_pred_away_win")
          , "metric_pred_draw": tf.metrics.mean(pred_draw, name="metric_pred_draw")
#          , "points/pt_draw_points": tf.metrics.mean(pt_draw_points)
#          , "points/pt_home_win_points": tf.metrics.mean(pt_home_win_points)
#          , "points/pt_away_loss_points": tf.metrics.mean(pt_away_loss_points)
#          , "points/pt_away_win_points": tf.metrics.mean(pt_away_win_points)
#          , "points/pt_home_loss_points": tf.metrics.mean(pt_home_loss_points)
          , "pt_softpoints" : tf.metrics.mean(pt_softpoints, name="pt_softpoints")
          , "pt_softpoints_capped" : tf.metrics.mean(pt_softpoints_capped, name="pt_softpoints_capped")
          , "pt_log_softpoints" : tf.metrics.mean(pt_log_softpoints, name="pt_log_softpoints")
          , "metric_ev_goals1_L1": tf.metrics.mean_absolute_error(labels=labels_float, predictions=ev_goals_1 , name="metric_ev_goals1_L1")
          , "metric_ev_goals2_L1": tf.metrics.mean_absolute_error(labels=labels_float2, predictions=ev_goals_2 , name="metric_ev_goals2_L1")
          , "metric_ev_goals1_L2": tf.metrics.mean_squared_error(labels=labels_float, predictions=ev_goals_1 , name="metric_ev_goals1_L2")
          , "metric_ev_goals2_L2": tf.metrics.mean_squared_error(labels=labels_float2, predictions=ev_goals_2 , name="metric_ev_goals2_L2")
          , "metric_ev_goals_diff_L1": tf.metrics.mean(l_diff_ev_goals_L1, name="metric_ev_goals_diff_L1")
          , "metric_cor_1":tf.metrics.mean(corrcoef(ev_goals_1, labels_float), name="metric_cor_1")
          , "metric_cor_2":tf.metrics.mean(corrcoef(ev_goals_2, labels_float2), name="metric_cor_2")
          , "metric_cor_diff":tf.metrics.mean(corrcoef(ev_goals_1-ev_goals_2, labels_float-labels_float2), name="metric_cor_diff")
      }
    return eval_metric_ops, pt_log_softpoints

  def model(features, labels, mode):
    tc = constant_tensors()
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    
    with tf.variable_scope("Model"):
      outputs, logits, home_bias, sm_logits, ensemble_logits, index = buildGraph(features, mode)
      t_is_home_bool = tf.equal(features["newgame"][:,2] , 1)
#      t_is_train_bool = tf.equal(features["Train"] , True)

      def apply_prefix(predictions, prefix):
        return {prefix+k:v for k,v in predictions.items() }

      predictions = create_predictions(outputs, logits, t_is_home_bool, tc)
      predictions = apply_prefix(predictions, "sp/")

      sm_predictions = create_predictions(outputs, sm_logits, t_is_home_bool, tc, True)
      predictions.update(apply_prefix(sm_predictions, "sm/"))
      sm_predictions_hbias = apply_home_bias(sm_predictions, home_bias*0.5)
      predictions.update(apply_prefix(sm_predictions_hbias, "smhb/"))

      T1_GFT = tf.exp(outputs[:,0])
      T2_GFT = tf.exp(outputs[:,1])
      T1_GHT = tf.exp(outputs[:,2])
      T2_GHT = tf.exp(outputs[:,3])
      T1_GH2 = tf.exp(outputs[:,16])
      T2_GH2 = tf.exp(outputs[:,17])

      predictions_poisson_FT = create_predictions_from_ev_goals(T1_GFT, T2_GFT, t_is_home_bool, tc)
      predictions_poisson_rule_based_p3 = apply_home_bias(predictions_poisson_FT, home_bias)
      predictions.update(apply_prefix(predictions_poisson_rule_based_p3, "p3/"))
#      if mode==tf.estimator.ModeKeys.TRAIN:
#        predictions_poisson_FT["p_pred_12"]=tf.zeros_like(predictions_poisson_FT["p_pred_12"]) 
      predictions.update(apply_prefix(predictions_poisson_FT, "p1/"))

      predictions_poisson_HT = create_predictions_from_ev_goals(T1_GHT+T1_GH2, T2_GHT+T2_GH2, t_is_home_bool, tc)
      predictions_poisson_rule_based_p5 = apply_home_bias(predictions_poisson_HT, home_bias)
      predictions.update(apply_prefix(predictions_poisson_rule_based_p5, "p5/"))
#      if mode==tf.estimator.ModeKeys.TRAIN:
#        predictions_poisson_HT["p_pred_12"]=tf.zeros_like(predictions_poisson_HT["p_pred_12"]) 
      predictions.update(apply_prefix(predictions_poisson_HT, "p2/"))
      
      predictions_poisson_rule_based = create_predictions_from_poisson(outputs, t_is_home_bool, tc)
#      if mode==tf.estimator.ModeKeys.TRAIN:
#        predictions_poisson_rule_based ["p_pred_12"]=tf.zeros_like(predictions_poisson_rule_based ["p_pred_12"]) 
      
      predictions_poisson_rule_based_p4 = apply_prefix(predictions_poisson_rule_based, "p4/")
      predictions_poisson_rule_based_p4["p4/pred"] = predictions_poisson_rule_based_p4["p4/pred_ev"] # use max expected value instead of max likelyhood
      predictions.update(predictions_poisson_rule_based_p4)

      predictions_poisson_rule_based_p6 = apply_home_bias(predictions_poisson_rule_based, home_bias*2)
      predictions.update(apply_prefix(predictions_poisson_rule_based_p6, "p6/"))

      predictions["outputs_poisson"] = outputs
      predictions["index"] = index
      
      predictions_ensemble = create_ensemble_predictions(ensemble_logits, predictions, t_is_home_bool, tc)
      predictions_ensemble = apply_prefix(predictions_ensemble, "ens/")
      predictions.update(predictions_ensemble)
      
      export_outputs = {
          "predictions": tf.estimator.export.ClassificationOutput(predictions["sp/p_pred_12"])
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
      
      eval_metric_ops = create_poisson_correlation_metrics(outputs, labels)
      
      # prepare derived data from labels
      gs = tf.minimum(t_goals_1,6)
      gc = tf.minimum(t_goals_2,6)
      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    

      def append_result_metrics(result_metrics, prefix):
        result_metrics_new, pt_log_softpoints = create_result_metrics(prefix, predictions, t_goals[:,0], t_goals[:,1], t_is_home_bool, achievable_points_mask, tc)
        predictions["pt_log_softpoints"]=pt_log_softpoints
        result_metrics_new = {prefix+"/"+k:v for k,v in result_metrics_new.items() }
        result_metrics.update(result_metrics_new)

      result_metrics = {}
      append_result_metrics(result_metrics, "sp")
      append_result_metrics(result_metrics, "sm")
      append_result_metrics(result_metrics, "smhb")
      append_result_metrics(result_metrics, "p1")
      append_result_metrics(result_metrics, "p2")
      append_result_metrics(result_metrics, "p3")
      append_result_metrics(result_metrics, "p4")
      append_result_metrics(result_metrics, "p5")
      append_result_metrics(result_metrics, "p6")
      ensemble_metrics = create_ensemble_result_metrics(predictions, t_goals[:,0], t_goals[:,1], achievable_points_mask, tc)
      result_metrics.update({"ens/"+k:v for k,v in ensemble_metrics.items() })
      
    eval_metric_ops.update(result_metrics)
    eval_metric_ops.update({"summary/metric_home_bias": tf.metrics.mean(home_bias)})

    eval_loss_ops, loss, make_noise_op = create_losses_RNN(outputs, logits, sm_logits, labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool, eval_metric_ops)
      
    eval_metric_ops.update(eval_loss_ops)
    eval_metric_ops.update({"summary/"+k:v for k,v in eval_metric_ops.items() if "z_points" in k })
    eval_metric_ops.update({"summary/"+k:v for k,v in eval_metric_ops.items() if "is_tendency" in k })

    if mode == tf.estimator.ModeKeys.EVAL:
      for key, value in eval_metric_ops.items():
        tf.summary.scalar(key, value[1])
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, loss= loss, eval_metric_ops=eval_metric_ops)
  
  
    global_step = tf.train.get_global_step()
    #optimizer = tf.train.GradientDescentOptimizer(1e-4)
    learning_rate = 1e-4 # 1e-3 -> 1e-2 on 4.1.2018 and back 1e-4, 3e-4
    print("Learning rate = {}".format(learning_rate))
    optimizer = tf.train.AdamOptimizer(learning_rate)


#    reset_node = tf.get_default_graph().get_tensor_by_name("Model/HomeBias/HomeBias/bias:0")
#    reset_op = tf.assign(reset_node, reset_node - 0.5)
    #reset_node = tf.get_default_graph().get_tensor_by_name("Model/home_bias:0")
    #reset_op = tf.assign(reset_node, tf.minimum(reset_node, -1.0))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # Ensures that we execute the update_ops before performing the train_step
      train = tf.group( make_noise_op, optimizer.minimize(loss), tf.assign_add(global_step, 1))
    
    # keep only summary-level metrics for training
    eval_metric_ops = {k:v for k,v in eval_metric_ops.items() if "summary/" in k }
    for key, value in eval_metric_ops.items():
      tf.summary.scalar(key, value[1])

    summary_op=tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                       output_dir=model_dir+"/train",
                                       scaffold=None,
                                       summary_op=summary_op)
    
    checkpoint_hook = tf.train.CheckpointSaverHook(model_dir, 
                                            save_steps=save_steps, 
                                            saver = tf.train.Saver(max_to_keep=max_to_keep))
 
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions 
                                      , loss= loss, train_op=train
                                      , eval_metric_ops=eval_metric_ops
                                      , training_hooks = [summary_hook, checkpoint_hook]  )

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
      print_counts = {k:v for k,v in zip(counts.keys().tolist(), counts.tolist())}
      print(sorted( ((v,k) for k,v in print_counts.items()), reverse=True))
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

      
