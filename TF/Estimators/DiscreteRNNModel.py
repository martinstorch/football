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

def multiclass_hinge_loss(labels, multi_labels_mask, logits):
  """Adds Ops for computing the multiclass hinge loss.
  The implementation is based on the following paper:
  On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
  by Crammer and Singer.
  link: http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
  This is a generalization of standard (binary) hinge loss. For a given instance
  with correct label c*, the loss is given by:
    loss = max_{c != c*} logits_c - logits_{c*} + 1.
  or equivalently
    loss = max_c { logits_c - logits_{c*} + I_{c != c*} }
  where I_{c != c*} = 1 if c != c* and 0 otherwise.
  """  
  with tf.variable_scope("Hinge_Loss"):  
    #label_logits = tf.reduce_sum(labels * logits, 1, keep_dims=True)
    label_logits = tf.reduce_max(multi_labels_mask * logits, axis=1, keep_dims=True)
    margin = (logits - label_logits + 1) * (1-multi_labels_mask)
    margin = nn_ops.relu(margin)
    loss = math_ops.reduce_max(margin, axis=1)
  return loss  



  
def create_estimator(model_dir, columns):    
  
  def buildGraph(prefix, features, columns, mode, previous_state=None): 
      with tf.variable_scope(prefix+"_Input_Layer"):
        X = tf.feature_column.input_layer(features, columns)
        if previous_state is not None:
          previous_state = tf.cast(previous_state, tf.float32)
          X = tf.concat([X, previous_state], axis=1)
        tf.summary.histogram("Inputs", X)
      
      with tf.variable_scope(prefix+"_Layer1"):
#        mean, variance = tf.nn.moments(X, axes=[0], keep_dims=True)
#        X = tf.nn.batch_normalization(X, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=.000000001)
        X_inputs = X
        tf.summary.histogram("Inputs", X)
        X = tf.layers.dense(inputs=X, units=10, reuse=tf.AUTO_REUSE, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(stddev=0.2))
        #X = tf.layers.dense(inputs=X, units=30, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(0), kernel_initializer=tf.random_uniform_initializer(-0.2, 0.2))
        X = tf.layers.dropout(X,rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
        tf.summary.histogram("Outputs", X)
        W = tf.get_default_graph().get_tensor_by_name("Model/"+prefix+"_Layer1/dense/kernel/read:0")
        tf.summary.histogram("Weights", W)
        b = tf.get_default_graph().get_tensor_by_name("Model/"+prefix+"_Layer1/dense/bias/read:0")
        tf.summary.histogram("Bias", b)
        X = tf.concat([X_inputs, X], axis=1)
  #      
  #    with tf.variable_scope("Layer2"):
  #      #X = tf.nn.batch_normalization(X, mean=0, variance=1, offset=None, scale=None, variance_epsilon=.000000001)
  #      tf.summary.histogram("Inputs", X)
  #      X = tf.layers.dense(inputs=X, units=10, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(stddev=0.2))
  #      X = tf.layers.dropout(X,rate=0.3, training=(mode == tf.estimator.ModeKeys.TRAIN))
  #      tf.summary.histogram("Outputs", X)
  #      W = tf.get_default_graph().get_tensor_by_name("Layer2/dense/kernel/read:0")
  #      tf.summary.histogram("Weights", W)
  #      b = tf.get_default_graph().get_tensor_by_name("Layer2/dense/bias/read:0")
  #      tf.summary.histogram("Bias", b)
  
      with tf.variable_scope(prefix+"_Output_Layer"):
        #X = tf.nn.batch_normalization(X, mean=0, variance=1, offset=None, scale=None, variance_epsilon=.000000001)
        tf.summary.histogram("Inputs", X)
        log_lambda = tf.layers.dense(inputs=X, units=49, reuse=tf.AUTO_REUSE, activation=None, bias_initializer=tf.constant_initializer(0), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        W = tf.get_default_graph().get_tensor_by_name("Model/"+prefix+"_Output_Layer/dense/kernel/read:0")
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Outputs", log_lambda)
        b = tf.get_default_graph().get_tensor_by_name("Model/"+prefix+"_Output_Layer/dense/bias/read:0")
        tf.summary.histogram("Bias", b)
          
      logits = tf.reshape(log_lambda, (-1,7,7))
      return logits
        
  def create_predictions(logits, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        p_pred_12 = tf.nn.softmax(tf.reshape(logits, [-1, 49]))
        
        
        ev_points =  tf.where(t_is_home_bool,
                               tf.matmul(p_pred_12, tc_home_points_i),
                               tf.matmul(p_pred_12, tc_away_points_i))
        
        #a = tf.argmax(tf.reshape(logits, [-1, 49]), axis=1)
        _ , best_preds = tf.nn.top_k(tf.reshape(logits, [-1, 49]), k=2)
        best_preds = tf.cast(best_preds, tf.int64)
        a = best_preds[:,0]
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        a2 = best_preds[:,1]
        alt_pred = tf.reshape(tf.stack([a2 // 7, tf.mod(a2, 7)], axis=1), [-1,2])
#        a = tf.argmax(ev_points, axis=1)
#        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])

        p_pred_12_m = tf.reshape(p_pred_12, [-1,7,7])
        p_marg_1  = tf.reduce_sum(p_pred_12_m, axis=2)
        p_marg_2  = tf.reduce_sum(p_pred_12_m, axis=1)
        ev_goals_1 = tf.matmul(p_marg_1, tc_1d1_goals_f)[:,0]
        ev_goals_2 = tf.matmul(p_marg_2, tc_1d1_goals_f)[:,0]
        
        p_poisson_1 = calc_poisson_prob(ev_goals_1)
        l_chisq_poiss_marg_1 = tf.reduce_mean(tf.square(p_marg_1 - p_poisson_1)/(p_poisson_1+0.0001), axis=1)
        p_poisson_2 = calc_poisson_prob(ev_goals_2)
        l_chisq_poiss_marg_2 = tf.reduce_mean(tf.square(p_marg_2 - p_poisson_2)/(p_poisson_2+0.0001), axis=1)
        
        p_pred_x1 = tf.transpose(tf.nn.softmax(logits, dim=1), [0,2,1]) # tf.reshape(p_pred_12_m, [-1,5])
        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7])
        ev_goals_x1 = tf.matmul(p_pred_x1, tc_1d1_goals_f)
        p_poisson_x1 = calc_poisson_prob(ev_goals_x1[:,0])
        l_chisq_poiss_pred_x1 = tf.reduce_mean(tf.square(p_pred_x1 - p_poisson_x1)/(p_poisson_x1+0.0001), axis=1)
        
        p_pred_x1 = tf.reshape(p_pred_x1, [-1,7,7])
        ev_goals_x1 = tf.reshape(ev_goals_x1, [-1,7])
        p_poisson_x1 = tf.reshape(p_poisson_x1, [-1,7,7])
        l_chisq_poiss_pred_x1 = tf.reshape(l_chisq_poiss_pred_x1, [-1,7])
        
        p_pred_x2 = tf.nn.softmax(logits, dim=2) # tf.reshape(tf.transpose(p_pred_12_m, [0,2,1]), [-1,5])
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
          "alt_pred":alt_pred,
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
  
  def add_noise(labels, labels2, stddev):  
    with tf.variable_scope("Label_Noise"):
      noise = tf.random_normal(shape=tf.shape(labels), mean=0.0, stddev=stddev, dtype=tf.float32) 
      noise2 = tf.random_normal(shape=tf.shape(labels), mean=0.0, stddev=stddev, dtype=tf.float32) 
      noise = tf.cast(tf.round(noise), tf.int64)
      noise2 = tf.cast(tf.round(noise2), tf.int64)
      labels=tf.nn.relu(labels+noise)
      labels2=tf.nn.relu(labels2+noise2)
    return labels, labels2 

  def create_model_regularization_metrics(prefix):
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
      W = tf.get_default_graph().get_tensor_by_name("Model/"+prefix+"_Output_Layer/dense/kernel/read:0")
      reg_layerOut_L2 = tf.reduce_mean(tf.square(W), axis=1)
      reg_layerOut_L1 = tf.reduce_mean(tf.abs(W), axis=1)
      
      l_regularization = tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)
  #    l_regularization = tf.reduce_mean(0.001*reg_layer1_L1+1000*reg_layer1_L2)
  #    l_regularization += tf.reduce_mean(0.001*reg_layer2_L1+1000*reg_layer2_L2)
      l_regularization += tf.reduce_mean(0.001*reg_layerOut_L1+1*reg_layerOut_L2)
      reg_eval_metric_ops = {
            "l_regularization": tf.metrics.mean(l_regularization)
  #        , "reg_layer1_L1": tf.metrics.mean(reg_layer1_L1)
  #        , "reg_layer1_L2": tf.metrics.mean(reg_layer1_L2)
  #        , "reg_layer2_L1": tf.metrics.mean(reg_layer2_L1)
  #        , "reg_layer2_L2": tf.metrics.mean(reg_layer2_L2)
          , "reg_layerOut_L1": tf.metrics.mean(reg_layerOut_L1)
          , "reg_layerOut_L2": tf.metrics.mean(reg_layerOut_L2)
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
      logits = predictions["logits"]
      logits = tf.reshape(logits, [-1, 49])

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
  
  def create_losses_and_metrics_HH(prefix, predictions, labels, labels2, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

    if mode == tf.estimator.ModeKeys.TRAIN: # or mode == tf.estimator.ModeKeys.EVAL:
      labels, labels2 = add_noise(labels, labels2, 0.01)      

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
      logits = predictions["logits"]
      logits = tf.reshape(logits, [-1, 49])

    with tf.variable_scope("Losses"):
      l_loglike_ev_goals1 = ev_goals_1 - labels_float*tf.log(ev_goals_1)
      l_loglike_ev_goals2 = ev_goals_2 - labels_float2*tf.log(ev_goals_2)
  
      l_diff_ev_goals_L1 = tf.sqrt(tf.abs(labels_float-labels_float2-( ev_goals_1-ev_goals_2)))
  
      z = tf.cast(tf.zeros_like(gs), tf.float32)
      
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
      
      l_regularization, reg_eval_metric_ops = create_model_regularization_metrics(prefix)
      
      #l_gdiff       = tf.nn.softmax_cross_entropy_with_logits(labels=p_gdiff, logits=tf.reshape(logits, [-1,49]))
      #l_tendency       = tf.nn.softmax_cross_entropy_with_logits(labels=p_tendency, logits=tf.reshape(logits, [-1,49]))
  
      loss = l_full # + l_loglike_ev_goals1 + l_loglike_ev_goals2 
#      loss *= tf.where( t_is_home_loss_bool, z+2, z+1) # away win = 4 x
#      loss *= tf.where( t_is_home_win_bool, z+2, z+1)  # home win = 2 x, draw = 1 x
     
      loss = tf.reduce_mean(loss) + 1000*l_regularization
      tf.summary.scalar("loss", loss)
      
      eval_metric_ops = {
            prefix + "_l_loglike_ev_goals1": tf.metrics.mean(l_loglike_ev_goals1)
          , prefix + "_l_loglike_ev_goals2": tf.metrics.mean(l_loglike_ev_goals2)
          , prefix + "_l_diff_ev_goals_L1": tf.metrics.mean(l_diff_ev_goals_L1)
          , prefix + "_l_full": tf.metrics.mean(l_full)
          , prefix + "_metric_ev_goals1_L1": tf.metrics.mean_absolute_error(labels=labels_float, predictions=ev_goals_1 )
          , prefix + "_metric_ev_goals2_L1": tf.metrics.mean_absolute_error(labels=labels_float2, predictions=ev_goals_2 )
          , prefix + "_metric_ev_goals1_L2": tf.metrics.mean_squared_error(labels=labels_float, predictions=ev_goals_1 )
          , prefix + "_metric_ev_goals2_L2": tf.metrics.mean_squared_error(labels=labels_float2, predictions=ev_goals_2 )
          , prefix + "_metric_is_tendency": tf.metrics.mean(is_tendency)
          , prefix + "_metric_is_diff": tf.metrics.mean(is_diff)
          , prefix + "_metric_is_full": tf.metrics.mean(is_full)
      }
      eval_metric_ops.update(reg_eval_metric_ops)
      
    return eval_metric_ops, loss


  def combine1H2H(predictions1H, predictions2H, t_is_home_bool, tc_home_points_i, tc_away_points_i, p_fulltime_index_matrix, calc_poisson_prob):
    with tf.variable_scope("Combined_Prediction"):
      p1 = predictions1H["p_pred_12"]
      p2 = predictions2H["p_pred_12"]
      p2 = tf.transpose(p2, [1,0])
      p3 = tf.gather(p2, tf.maximum(p_fulltime_index_matrix,0), axis=0)
      probMatrix = tf.where(tf.reshape(tf.equal(p_fulltime_index_matrix, -1), [49*49]),
                             tf.zeros(shape=[49*49, tf.shape(p2)[1]]),
                             tf.reshape(p3, [49*49, -1])
                             )
      probMatrix = tf.transpose(probMatrix, [1,0])
      probMatrix = tf.reshape(probMatrix, [-1,49,49])
      p_pred_12_raw = tf.multiply(tf.reshape(p1, [-1,49,1]), probMatrix)
      p_pred_12 = tf.reduce_sum(p_pred_12_raw, axis=1)
      
      # p_pred_12 = (predictions1H["p_pred_12"]+predictions2H["p_pred_12"])/2
      ev_points =  tf.where(t_is_home_bool,
                             tf.matmul(p_pred_12, tc_home_points_i),
                             tf.matmul(p_pred_12, tc_away_points_i))
      
#      a = tf.argmax(tf.reshape(p_pred_12, [-1, 49]), axis=1)
#      a = tf.argmax(tf.reshape(ev_points, [-1, 49]), axis=1)
      a = tf.argmax(tf.reshape(0.1*p_pred_12+ev_points, [-1, 49]), axis=1)
      pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])

      predictions = {
            "p_marg_1":(predictions1H["p_marg_1"]+predictions2H["p_marg_1"])/2, 
            "p_marg_2":(predictions1H["p_marg_2"]+predictions2H["p_marg_2"])/2, 
            "logits": (predictions1H["logits"]+predictions2H["logits"])/2,
            "p_pred_12":p_pred_12, 
            "ev_points":ev_points,  
            "pred_raw": predictions1H["pred"]+predictions2H["pred"],
            "pred": pred,
            "ev_goals_1":predictions1H["ev_goals_1"]+predictions2H["ev_goals_1"],
            "ev_goals_2":predictions1H["ev_goals_2"]+predictions2H["ev_goals_2"],
            "p_poisson_1": calc_poisson_prob(predictions1H["ev_goals_1"]+predictions2H["ev_goals_1"]),
            "p_poisson_2": calc_poisson_prob(predictions1H["ev_goals_2"]+predictions2H["ev_goals_2"]),
            "p3":tf.transpose(p3, [2,0,1]),
            "probMatrix":probMatrix,
            "p_pred_12_raw":p_pred_12_raw,
          }
      for key, value in predictions1H.items():
        predictions['H1_'+key]=value
      for key, value in predictions2H.items():
        predictions['H2_'+key]=value

    return predictions        

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

  def model(features, labels, mode):
    
    tc = constant_tensors()
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    
    with tf.variable_scope("Model"):

      logits1H = buildGraph("1H", features, columns, mode)
      t_is_home_bool = tf.equal(features["Where"] , "Home")
      predictions1H = create_predictions(logits1H, t_is_home_bool, tc)
      
      if mode == tf.estimator.ModeKeys.TRAIN:
        logits2H = buildGraph("2H", features, columns, mode, tf.stack([features["T1_GHT"],features["T2_GHT"]], axis=1))
      else:
        logits2H = buildGraph("2H", features, columns, mode, predictions1H["pred"])
          
      predictions2H = create_predictions(logits2H, t_is_home_bool, tc)
      predictions = combine1H2H(predictions1H, predictions2H, t_is_home_bool, tc_home_points_i, tc_away_points_i, p_fulltime_index_matrix, calc_poisson_prob)

      logits2H_alt = buildGraph("2H", features, columns, mode, predictions1H["alt_pred"])
      if mode == tf.estimator.ModeKeys.PREDICT:
        # Build alternative prediction with 2nd-most likely outcome of 1H
        predictions2H_alt = create_predictions(logits2H_alt, t_is_home_bool, tc)
        predictions_alt = combine1H2H(predictions1H, predictions2H_alt, t_is_home_bool, tc_home_points_i, tc_away_points_i, p_fulltime_index_matrix, calc_poisson_prob)
        for key, value in predictions_alt.items():
          predictions['Alt_'+key]=value
      
      export_outputs = {
          "predictions": tf.estimator.export.RegressionOutput(predictions["p_marg_1"])
      }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    with tf.variable_scope("Evaluation"):
  
      t_is_home_loss_bool = (t_is_home_bool & tf.less(features["T1_GFT"],features["T2_GFT"])) | (tf.logical_not(t_is_home_bool) & tf.greater(features["T1_GFT"],features["T2_GFT"]))
      t_is_home_win_bool = (t_is_home_bool & tf.greater(features["T1_GFT"],features["T2_GFT"])) | (tf.logical_not(t_is_home_bool) & tf.less(features["T1_GFT"],features["T2_GFT"]))
    
      eval_metric_ops_1H, loss_1H = create_losses_and_metrics_HH("1H", predictions1H, features["T1_GHT"], features["T2_GHT"], t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool)
      eval_metric_ops_2H, loss_2H = create_losses_and_metrics_HH("2H", predictions2H, features["T1_GFT"]-features["T1_GHT"], features["T2_GFT"]-features["T2_GHT"], t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool)
    
      eval_metric_ops = eval_metric_ops_1H
      eval_metric_ops.update(eval_metric_ops_2H)
      loss = loss_1H + loss_2H

      # softpoints
      gs = tf.minimum(features["T1_GFT"],6)
      gc = tf.minimum(features["T2_GFT"],6)
      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))                        
      pt_softpoints = tf.reduce_sum(predictions["p_pred_12"] * achievable_points_mask, axis=1)
      eval_metric_ops["pt_softpoints"] = tf.metrics.mean(pt_softpoints)
      loss -= tf.reduce_mean(pt_softpoints) 
      
      result_metrics = create_result_metrics(predictions["pred"][:,0], predictions["pred"][:,1], features["T1_GFT"], features["T2_GFT"], t_is_home_bool)
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

      
