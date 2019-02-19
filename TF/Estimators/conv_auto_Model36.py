# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
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
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import extend_with_decoupled_weight_decay

GLOBAL_REGULARIZER = l2_regularizer(scale=0.1)
MINOR_REGULARIZER = l2_regularizer(scale=0.005)
plot_list = ["cp/", "cp2/", "sm2/", "pg2/", "av/"] # ["pspt/", "cp/", "cp2/", "sp/", "ens/"]
prefix_list = ["pgpt/", "sp/", "cp/", "cp2/", "pg2/", "av/"]
#ens_prefix_list = ["pg/", "pgpt/", "pghb/", "ps/", "pspt/", "sp/", "smpt/", "cp/", "cp2/"]
ens_prefix_list = ["pgpt/", "sp/", "cp/", "cp2/", "pg2/"]
segmentation_strategies = {"cp":"cp2", "pg":"pg2"}
point_scheme = None

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
  
    draw_points = tf.where(is_draw, tf.where(is_full, z+point_scheme[0][1], tf.where(is_diff, z+point_scheme[1][1], z, name="is_diff"), name="is_full"), z, name="is_draw") 
    home_win_points  = tf.where(is_win & is_home & is_tendency, tf.where(is_full, z+point_scheme[0][0], tf.where(is_diff, z+point_scheme[1][0], z+point_scheme[2][0])), z)
    home_loss_points = tf.where(is_loss & is_home & is_tendency,  tf.where(is_full, z+point_scheme[0][2], tf.where(is_diff, z+point_scheme[1][2], z+point_scheme[2][2])), z)
    away_win_points  = tf.where(is_win & is_away & is_tendency,  tf.where(is_full, z+point_scheme[0][2], tf.where(is_diff, z+point_scheme[1][2], z+point_scheme[2][2])), z)
    away_loss_points = tf.where(is_loss & is_away & is_tendency, tf.where(is_full, z+point_scheme[0][0], tf.where(is_diff, z+point_scheme[1][0], z+point_scheme[2][0])), z)
    
    points = tf.cast(draw_points+home_win_points+home_loss_points+away_loss_points+away_win_points, tf.float32)
  return (points, is_tendency, is_diff, is_full,
          draw_points, home_win_points, home_loss_points, away_loss_points, away_win_points)

def collect_summary(scope, name, mode, tensor=None, metric= tf.metrics.mean, reduce=tf.reduce_mean):
  scope_match = scope in ["summary", "losses", "Gradients"]
  name_match = name in ["z_points", "p_points", "metric_is_tendency"]
  if mode == tf.estimator.ModeKeys.TRAIN and not scope_match and not name_match:
    return {}  
  if tensor is not None:
    cache = {scope+"/"+name : (metric(tensor),  reduce(tensor))}
  else:
    cache = {scope+"/"+name : (metric,  None)}
  return cache

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
    p_tendency_mask_f = p_tendency_mask_f / tf.reduce_sum(p_tendency_mask_f, axis=[1], keepdims=True)
    
    p_gdiff_mask_f = tf.cast(tc_home_masks[2], tf.float32)
    p_gdiff_mask_f  = tf.reshape(p_gdiff_mask_f, [49,49])
    p_gdiff_mask_f = p_gdiff_mask_f / tf.reduce_sum(p_gdiff_mask_f, axis=[1], keepdims=True)

    p_fulltime_index_matrix = tf.reshape(tf.constant(d["target"].values, dtype=tf.int64), shape=[49,49])
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

    l = [tc_1d7_goals_f, tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, p_tendency_mask_f, p_gdiff_mask_f]
    l = [tf.cast(x, tf.float32) for x in l]
    tc_1d7_goals_f, tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, p_tendency_mask_f, p_gdiff_mask_f = l
    
    def calc_poisson_prob(lambda0):
      lambda0 = tf.reshape(lambda0, [-1,1]) # make sure that rank=2
      loglambda = tf.log(lambda0)
      x = tf.matmul(loglambda, tc_1d7_goals_f) - tc_1d7_logfactorial_f
      x = tf.exp(x - lambda0)
      return x

    return tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix

#def add_weight_noise(X, stddev_factor):  
#  with tf.variable_scope("Weight_Noise"):
#    mean, variance = tf.nn.moments(X, axes=[0], keepdims=True)
#    noise = tf.random_normal(shape=tf.shape(X), mean=0.0, stddev=variance * stddev_factor, dtype=tf.float32) 
#  return X+noise
#  
def corrcoef(x_t,y_t):
  def t(x): return tf.transpose(x)
  x_t = tf.cast(x_t, tf.float32)
  y_t = tf.cast(y_t, tf.float32)
  x_t = tf.expand_dims(x_t, axis=0)
  y_t = tf.expand_dims(y_t, axis=0)
  xy_t = tf.concat([x_t, y_t], axis=0)
  mean_t = tf.reduce_mean(xy_t, axis=1, keepdims=True)
  cov_t = ((xy_t-mean_t) @ t(xy_t-mean_t))/tf.cast(tf.shape(x_t)[1]-1, x_t.dtype)
  cov2_t = tf.diag(1/tf.sqrt(tf.diag_part(cov_t)+0.000001))
  cor = cov2_t @ cov_t @ cov2_t
  #print(cor)
  return cor[0,1]
  
#from tensorflow.python.framework import ops
#
#@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [tf.clip_by_norm(grad, 20.0, name=op.name+"Grad"), tf.zeros(tf.shape(op.inputs[1]))]

def rnnSample_gradient(op, grad):
    return [0.5*tf.clip_by_norm(grad, 1.0, name=op.name+"Grad"), tf.zeros(tf.shape(op.inputs[1]))]

#
#@ops.RegisterGradient("Multinomial_ST")
def multinomial_ST(op, grad):
  sample_idx = tf.squeeze(op.outputs[0] , axis=1) 
  logits = op.inputs[0]
  with tf.control_dependencies([tf.assert_rank(sample_idx, 1), tf.assert_rank(logits, 2)]):  
#    tf.summary.histogram("gradient/multinomial", grad)
#    tf.summary.scalar("gradient/multinomial_max", tf.reduce_max(grad))
#    tf.summary.scalar("gradient/multinomial_min", tf.reduce_min(grad))
#    tf.summary.scalar("gradient/multinomial_mean", tf.reduce_mean(grad))
    #grad = tf.print(grad, [grad], message="multinomial gradient")
    grad = tf.clip_by_norm(grad, 20.0)
    prob = tf.nn.softmax(logits, dim=1)
    input_gradient = tf.one_hot(sample_idx, 49, dtype=tf.float32) * (grad+0.0) / (prob+0.0002)
    input_gradient = tf.identity(input_gradient, name=op.name+"Grad")
    return [input_gradient, tf.zeros(tf.shape(op.inputs[1]))]
#
#@ops.RegisterGradient("One_Hot_ST")
def one_hot_ST(op, grad):
  with tf.control_dependencies([
      tf.assert_rank(grad, 2),
      tf.assert_equal(grad.shape[1], 49)]):
    sample_idx = tf.cast(op.inputs[0], tf.int32) 
    grad = tf.clip_by_norm(grad, 20.0)
    input_gradient = tf.gather(
        params  = tf.reshape(grad, [-1]), 
        indices = sample_idx + tf.range(start=0, limit=tf.shape(grad)[0]*49, delta=49, dtype=tf.int32) ,
        axis = 0) 
    input_gradient = tf.identity(input_gradient, name=op.name+"Grad")
    return [input_gradient, 
            tf.zeros(tf.shape(op.inputs[1])),
            tf.zeros(tf.shape(op.inputs[2])),
            tf.zeros(tf.shape(op.inputs[3]))]

def multinomialSample(logits):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'MultinomialGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(multinomial_ST) 
    rnd_name2 = 'OneHotGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name2)(one_hot_ST)  
    
    g = tf.get_default_graph()
    with ops.name_scope("MultinomialSample") :
      with g.gradient_override_map({"Multinomial": rnd_name, "OneHot": rnd_name2}):
        samples = tf.multinomial(logits, 1) # seed, name, output_dtype
        print(samples)
        sample_vector = tf.one_hot(samples[:,0], 49, dtype=tf.float32) # on_value=None, off_value=None, axis=None, dtype=None, name=None):
        return sample_vector 
    

def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (identity).
    """
    rnd_name = 'BernoulliGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(bernoulliSample_ST) 

    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": rnd_name}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

def tanhSample(x):
    """
    Uses a tensor whose values are in [-1,1] to sample a tensor with values in {-1, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, tanhSample(x) will be 1 with probability 0.6, and -1 otherwise,
    and the gradient will be pass-through (identity).
    """
    rnd_name = 'RNNGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(rnnSample_gradient) 

    g = tf.get_default_graph()

    with ops.name_scope("TanhSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": rnd_name}):
            return 2*tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)-1




def create_estimator(model_dir, label_column_names, my_feature_columns, thedata, thelabels, save_steps, evaluate_after_steps, max_to_keep, teams_count, use_swa, histograms):    
  
  def variable_summaries(var, name, mode):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      if not histograms:
        return {}
      name = var.name[:-2]+"/"+name
      if mode != tf.estimator.ModeKeys.TRAIN: 
        tf.summary.histogram(name, var)
        # skip histograms for training - improved performance
      eval_metric_ops = {}  
      with tf.name_scope(name):
        eval_metric_ops.update(collect_summary("histogram", name+"_mean", mode, tensor=var))
        eval_metric_ops.update(collect_summary("histogram", name+"_stddev", mode, tensor=tf.sqrt(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))
        eval_metric_ops.update(collect_summary("histogram", name+"_min", mode, tensor=tf.reduce_min(var)))
        eval_metric_ops.update(collect_summary("histogram", name+"_max", mode, tensor=tf.reduce_max(var)))
      return eval_metric_ops
  
  def build_dense_layer(X, output_size, mode, regularizer = None, keep_prob=1.0, batch_norm=True, activation=tf.nn.relu, eval_metric_ops=None, use_bias=None, add_term=None, batch_scale=True):
    if eval_metric_ops is None:
      eval_metric_ops = {}
    eval_metric_ops.update(variable_summaries(X, "Inputs", mode))
  
    W = tf.get_variable(name="W", regularizer = regularizer, shape=[int(X.shape[1]), output_size],
          initializer=tf.contrib.layers.xavier_initializer(uniform=False),
          dtype=X.dtype)
    eval_metric_ops.update(variable_summaries(W, "Weights", mode))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
  
    if batch_norm and activation is None: 
      # normalize the inputs to mean=0 and variance=1
      X = tf.layers.batch_normalization(X, axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
      eval_metric_ops.update(variable_summaries(X, "InputNormalized", mode))
  
    Z = tf.matmul(X, W)
    
    if (use_bias or not batch_norm or (activation is not None and activation!=tf.nn.relu)): 
      # if no batch-norm or non-linear activation function: apply bias
      b = tf.get_variable(name="b", shape=[output_size],
            initializer=tf.zeros_initializer(),
            dtype=X.dtype)
      eval_metric_ops.update(variable_summaries(b, "Bias", mode))
      Z = Z + b
    
    variable_summaries(Z, "Linear", mode)
    
    if add_term is not None:
      Z = Z + add_term
              
    if batch_norm and activation is not None:
      # normalize the intermediate representation to learned parameters mean=gamma and variance=beta
      Z = tf.layers.batch_normalization(Z, axis=-1, momentum=0.99, center=True, scale=batch_scale, training=(mode == tf.estimator.ModeKeys.TRAIN))
      eval_metric_ops.update(variable_summaries(Z, "Normalized", mode))
  
    if activation is not None:
      X = activation(Z) 
    else:
      X = Z
  
    if mode == tf.estimator.ModeKeys.TRAIN:  
      X = tf.nn.dropout(X, keep_prob=keep_prob)
  
    eval_metric_ops.update(variable_summaries(X, "Outputs", mode))
    return X, Z
    
  def build_cond_prob_layer(X, labels, mode, regularizer, keep_prob, eval_metric_ops): 
    X = tf.stop_gradient(X)
    with tf.variable_scope("H1"):
      h1_logits,_ = build_dense_layer(X, 49, mode, regularizer = regularizer, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
      
  
      t_win_map = tf.stack([max(0, (i // 7) - np.mod(i, 7) ) for i in range(49)], name="t_win_map")
      t_loss_map = tf.stack([max(0, np.mod(i, 7)-(i // 7) ) for i in range(49)], name="t_loss_map")
      t_draw_map = tf.stack([1 if i // 7 == np.mod(i, 7) else 0  for i in range(49)], name="t_draw_map")
      t_win2_map = tf.stack([max(0, (i // 7) - np.mod(i, 7)-1)  for i in range(49)], name="t_win2_map")
      t_loss2_map = tf.stack([max(0, np.mod(i, 7)-(i // 7) -1 ) for i in range(49)], name="t_loss2_map")
      t_owngoal_map = tf.stack([i // 7 for i in range(49)], name="t_owngoal_map")
      t_oppgoal_map = tf.stack([np.mod(i, 7)  for i in range(49)], name="t_oppgoal_map")
      t_bothgoal_map = tf.stack([((i // 7)+np.mod(i, 7)) for i in range(49)], name="t_bothgoal_map")
      t_map = tf.stack([t_win_map, t_draw_map, t_loss_map, t_win2_map, t_loss2_map, t_owngoal_map, t_oppgoal_map, t_bothgoal_map], axis=1, name="t_map")
      t_map = tf.cast(t_map, X.dtype)
  
      t_win_mask = tf.stack([0 if (i // 7) >= np.mod(i, 7) else 0 for i in range(49)], name="t_win_mask")
      t_loss_mask = tf.stack([0 if (i // 7) <= np.mod(i, 7) else 0 for i in range(49)], name="t_loss_mask")
      t_draw_mask = tf.stack([0 for i in range(49)], name="t_draw_mask")
      t_win2_mask = tf.stack([0 if (i // 7) > np.mod(i, 7) else 0 for i in range(49)], name="t_win2_mask")
      t_loss2_mask = tf.stack([0 if (i // 7) < np.mod(i, 7) else 0 for i in range(49)], name="t_loss2_mask")
      t_owngoal_mask = tf.stack([1 for i in range(49)], name="t_owngoal_mask")
      t_oppgoal_mask = tf.stack([1 for i in range(49)], name="t_oppgoal_mask")
      t_bothgoal_mask = tf.stack([0 for i in range(49)], name="t_bothgoal_mask")
      t_mask = tf.stack([t_win_mask, t_draw_mask, t_loss_mask, t_win2_mask, t_loss2_mask, t_owngoal_mask, t_oppgoal_mask, t_bothgoal_mask], axis=1, name="t_mask")
      t_mask = tf.cast(t_mask, tf.float32)
  
      p_pred_12_h1 = tf.nn.softmax(h1_logits)
      p_pred_h1 = tf.matmul(p_pred_12_h1, t_map)
  
      if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        T1_GFT = labels[:,0]
        T2_GFT = labels[:,1]
        T1_GHT = labels[:,2]
        T2_GHT = labels[:,3]
        #T1_GH2 = labels[:,16]
        #T2_GH2 = labels[:,17]
        label_oh_h1 = tf.one_hot(tf.cast(T1_GHT*7+T2_GHT, tf.int32), 49, dtype=X.dtype)
        label_features_h1 = tf.matmul(label_oh_h1, t_map)
      else:
        label_features_h1 = None
        
  #    p_pred_win = p_pred_h1[:,0]
  #    p_pred_draw = p_pred_h1[:,1]
  #    p_pred_loss = p_pred_h1[:,2]
  #    p_pred_win2 = p_pred_h1[:,3]
  #    p_pred_loss2 = p_pred_h1[:,4]
  #    p_pred_owngoal = p_pred_h1[:,5]
  #    p_pred_oppgoal = p_pred_h1[:,6]
  #    p_pred_bothgoal = p_pred_h1[:,7]
      
    with tf.variable_scope("H2"):
      
      test_p_pred_12_h2 = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        # use actual half-time score as input for dense layer
        X = tf.concat([X, label_features_h1], axis=1)
        h2_logits,_ = build_dense_layer(X, 49, mode, regularizer = regularizer, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        p_pred_12_h2 = tf.nn.softmax(h2_logits)
        p_pred_h2 = tf.matmul(p_pred_12_h2, t_map)
      else:
        test_label_oh_h1 = tf.one_hot(tf.range(49), 49, dtype=X.dtype)
        test_label_features_h1 = tf.matmul(test_label_oh_h1, t_map)
  
        X3 = tf.concat([tf.map_fn(lambda x: tf.concat([x, test_label_features_h1[i]], axis=0), X) for i in range(49)], axis=0)
        #print(X3) # Tensor("Model/condprob/H2/concat:0", shape=(?, 136), dtype=float32)
  
        # this should find the same dense layer with same weights as in training - because name scope is same
        test_h2_logits,_ = build_dense_layer(X3, 49, mode, regularizer = regularizer, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
  
        test_p_pred_12_h2 = tf.nn.softmax(test_h2_logits)
        test_p_pred_12_h2 = tf.reshape(test_p_pred_12_h2, (49,-1,49), name="test_p_pred_12_h2") # axis 0: H1 scores, 1: batch, 2: H2 scores
        #test_p_pred_12_h2 = tf.print(test_p_pred_12_h2, data=[test_p_pred_12_h2[:,0,:]], message="Individual probabilities")
        
        p_pred_12_h2 = tf.expand_dims(tf.transpose(p_pred_12_h1, (1,0)), axis=2) * test_p_pred_12_h2  # prior * likelyhood
        p_pred_12_h2 = tf.reduce_sum(p_pred_12_h2 , axis=0) # posterior # axis 0: batch, 1: H2 scores
        #p_pred_12_h2 = tf.print(p_pred_12_h2, data=[p_pred_12_h2[0,:]], message="Summarised probability")
        
        p_pred_h2 = tf.matmul(p_pred_12_h2, t_map)
        #print(p_pred_h2)  #Tensor("Model/condprob/H2/MatMul_2:0", shape=(?, 8), dtype=float32)
  
        h2_logits = tf.log((p_pred_12_h2 + 1e-7) )
        
        test_p_pred_12_h2 = tf.transpose(test_p_pred_12_h2, [1,0,2])
         
      if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        label_oh_h2 = tf.one_hot(tf.cast(T1_GFT*7+T2_GFT, tf.int32), 49, dtype=X.dtype)
        label_features_h2 = tf.matmul(label_oh_h2, t_map)
      else:
        label_features_h2 = None
        
    return (h1_logits, h2_logits, p_pred_h1, label_features_h1, p_pred_h2, label_features_h2, t_mask, test_p_pred_12_h2, p_pred_12_h2)
      
  def buildGraph(features, labels, mode, params): 
      print(mode)
      eval_metric_ops = {}
      with tf.variable_scope("Input_Layer"):
        features_newgame = features['newgame']
        match_history_t1 = features['match_history_t1']
        match_history_t2 = features['match_history_t2'] 
        match_history_t12 = features['match_history_t12']
        
        f_date_round = features_newgame[:,4+2*teams_count : 4+2*teams_count+2]
        suppress_team_names = False
        if suppress_team_names:
          features_newgame = tf.concat([features_newgame[:,0:4], features_newgame[:,4+2*teams_count:]], axis=1)
          match_history_t1 = tf.concat([match_history_t1[:,:,0:2], match_history_t1[:,:,2+2*teams_count:]], axis=2)
          match_history_t2 = tf.concat([match_history_t2[:,:,0:2], match_history_t2[:,:,2+2*teams_count:]], axis=2)
          match_history_t12 = tf.concat([match_history_t12[:,:,0:2], match_history_t12[:,:,2+2*teams_count:]], axis=2)
          
        #batch_size = tf.shape(features_newgame)[0]
        num_label_columns = len(label_column_names)
        output_size = num_label_columns

        #match_history_t1_seqlen = 10*features_newgame[:,0]
        #match_history_t2_seqlen = 10*features_newgame[:,2]
        #match_history_t12_seqlen = 10*features_newgame[:,3]
        
        # batch normalization
        #features_newgame = tf.layers.batch_normalization(features_newgame, axis=1, momentum=0.99, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        #match_history_t1 = tf.layers.batch_normalization(match_history_t1, axis=2, momentum=0.99, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        #match_history_t2 = tf.layers.batch_normalization(match_history_t2, axis=2, momentum=0.99, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        #match_history_t12 = tf.layers.batch_normalization(match_history_t12, axis=2, momentum=0.99, center=False, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
        
      with tf.variable_scope("", reuse=True):
        with tf.variable_scope("Input_Layer"):
          eval_metric_ops.update(variable_summaries(features_newgame, "newgame", mode))
          eval_metric_ops.update(variable_summaries(match_history_t1, "match_history_t1", mode))
          eval_metric_ops.update(variable_summaries(match_history_t2, "match_history_t2", mode))
          eval_metric_ops.update(variable_summaries(match_history_t12, "match_history_t12", mode))

      def passThroughSigmoid(x, slope=1):
          """Sigmoid that uses identity function as its gradient"""
          g = tf.get_default_graph()
          with ops.name_scope("PassThroughSigmoid") as name:
              with g.gradient_override_map({"Sigmoid": "Identity"}):
                  return tf.sigmoid(x, name=name)
      
      def binaryStochastic(x):
        if mode == tf.estimator.ModeKeys.TRAIN:
          #print("sigmoid sampling")
          return bernoulliSample(passThroughSigmoid(x))
        else:
          #print("sigmoid smooth")
          return tf.sigmoid(x)

      def tanhStochastic(x):
        if mode == tf.estimator.ModeKeys.TRAIN:
          #print("tanh sampling")
          return tanhSample(passThroughSigmoid(x))
          #return tf.where(tf.less(tf.random_uniform(tf.shape(x)), 0.05), tanhSample(passThroughSigmoid(x)), tf.tanh(x))
        else:
          #print("tanh smooth")
          return tf.tanh(x)

#      def make_rnn_cell():
#        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=64, 
#                                          activation=tanhStochastic,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
#        #rnn_cell = tf.nn.rnn_cell.ResidualWrapper(rnn_cell)
#        if mode == tf.estimator.ModeKeys.TRAIN:
#          # 13.12.2017: was 0.9
#          rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
#               input_keep_prob=0.99, 
#               output_keep_prob=0.99,
#               state_keep_prob=0.99)
#        return rnn_cell
#      
#
#      def make_gru_cell():
#        return tf.nn.rnn_cell.MultiRNNCell([
#            make_rnn_cell(), 
#            tf.nn.rnn_cell.ResidualWrapper(make_rnn_cell())
#          ])
#      
#      def make_rnn(match_history, sequence_length, rnn_cell = make_gru_cell()):
#        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float16)
#        outputs, state = tf.nn.dynamic_rnn(rnn_cell, match_history,
#                                   initial_state=initial_state,
#                                   dtype=tf.float16,
#                                   sequence_length = sequence_length)
#        # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
#        # 'state' is a tensor of shape [batch_size, num_units]
#        eval_metric_ops.update(variable_summaries(outputs, "Intermediate_Outputs", mode))
#        eval_metric_ops.update(variable_summaries(state[1], "States", mode))
#        eval_metric_ops.update(variable_summaries(sequence_length, "Sequence_Length", mode))
#        return state[1] # use upper layer state
#
#      def rnn_histograms():
#        def rnn_histogram(section, num, part, regularizer=None):
#          scope = tf.get_variable_scope().name
#          summary_name = "gru_cell/"+section+num+"/"+part
#          node_name = scope+"/rnn/multi_rnn_cell/cell_"+num+"/gru_cell/"+section+"/"+part+":0"
#          node = tf.get_default_graph().get_tensor_by_name(node_name)
#          eval_metric_ops.update(variable_summaries(node, summary_name, mode))
#          if regularizer is not None:
#            loss = tf.identity(regularizer(node), name=summary_name)
#            ops.add_to_collection(tf.GraphKeys.WEIGHTS, node)
#            if loss is not None:
#              ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)
#
#        rnn_histogram("gates", "0", "kernel", l2_regularizer(scale=1.01))
#        rnn_histogram("gates", "0", "bias")
#        rnn_histogram("candidate", "0", "kernel", l2_regularizer(scale=3.01))
#        rnn_histogram("candidate", "0", "bias")
#        rnn_histogram("gates", "1", "kernel", l2_regularizer(scale=1.01))
#        rnn_histogram("gates", "1", "bias")
#        rnn_histogram("candidate", "1", "kernel", l2_regularizer(scale=3.01))
#        rnn_histogram("candidate", "1", "bias")
#
#      with tf.variable_scope("RNN_1"):
#        shared_rnn_cell = make_gru_cell()
#        history_state_t1 = make_rnn(match_history_t1, sequence_length = match_history_t1_seqlen, rnn_cell=shared_rnn_cell)  
#        rnn_histograms()
#      with tf.variable_scope("RNN_2"):
#        history_state_t2 = make_rnn(match_history_t2, sequence_length = match_history_t2_seqlen, rnn_cell=shared_rnn_cell)  
#      with tf.variable_scope("RNN_12"):
#        history_state_t12 = make_rnn(match_history_t12, sequence_length = match_history_t12_seqlen, rnn_cell=make_gru_cell())  
#        rnn_histograms()
      def conv_layer(X, name, output_channels, keep_prob=1.0):
        X = tf.layers.conv1d(X,
                  filters=output_channels, kernel_size=3, strides=1,
                  padding='valid',
                  data_format='channels_last',
                  dilation_rate=1,
                  activation=None, #tanhStochastic,
                  use_bias=True,
                  kernel_initializer=None,
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=l2_regularizer(scale=0.01),
                  bias_regularizer=None,
                  #activity_regularizer=l2_regularizer(scale=0.01),
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=name,
                  reuse=None
              )  
        eval_metric_ops.update(variable_summaries(X, "conv_outputs", mode))
        #X = tf.layers.batch_normalization(X, momentum=0.99, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
        X = tf.nn.tanh(X)
        if mode == tf.estimator.ModeKeys.TRAIN:  
          X = tf.nn.dropout(X, keep_prob=keep_prob)
        return X

      def avg_pool(X, name):
        eval_metric_ops.update(variable_summaries(X, "pooling_inputs", mode))
        return tf.layers.average_pooling1d(X, pool_size=2, strides=2,
              padding='valid',
              data_format='channels_last',
              name=name
            )
      
      def deconv_layer(X, name, output_width, output_channels, keep_prob=1.0, stride=1, activation=tf.nn.tanh):
        #print("input", X)
        W = tf.get_variable(name=name+"W", dtype=X.dtype,
                            shape=[3, output_channels, int(X.shape[2])],
                            regularizer = l2_regularizer(scale=0.001), 
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            )
        #print(W)
        eval_metric_ops.update(variable_summaries(W, "Weights", mode))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        X = tf.contrib.nn.conv1d_transpose(X, W, 
                   output_shape=[tf.shape(X)[0], output_width, output_channels],
                   stride=stride,
                   padding='VALID',
                   data_format='NWC',
                   name=name)
        #print("deconv", X)
        eval_metric_ops.update(variable_summaries(X, "deconv_outputs", mode))
        if activation is not None:
          X = activation(X)
        if mode == tf.estimator.ModeKeys.TRAIN:  
          X = tf.nn.dropout(X, keep_prob=keep_prob)
        return X

      def auto_encoder(X):
        with tf.variable_scope("enc"):
          X = conv_layer(X, name="conv1", output_channels=32, keep_prob=1.0)
          X = conv_layer(X, name="conv2", output_channels=16, keep_prob=1.0)
          X = avg_pool(X, name="avgpool")
          #print(X)
          shape_after_pooling = tf.shape(X)
          w,c = X.shape[1], X.shape[2]
          X = tf.reshape(X, (-1, w*c))
          #print(X)
          if mode == tf.estimator.ModeKeys.TRAIN:  
            X = tf.nn.dropout(X, keep_prob=1.0)
          hidden,_ = build_dense_layer(X, w*c, mode, regularizer = l2_regularizer(scale=0.01), keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops)
          #print("hidden", hidden)
        
        with tf.variable_scope("dec"):
          ####X = build_dense_layer(hidden, w*c, mode, regularizer = l2_regularizer(scale=0.01), keep_prob=1.0, batch_norm=True, activation=tf.nn.relu, eval_metric_ops=eval_metric_ops)
          X = hidden
          #print(X)
          X = tf.reshape(X, shape_after_pooling )
          #print(X)
          X = deconv_layer(X, "deconv1", 8, output_channels=32, keep_prob=1.0, activation=tf.nn.tanh, stride=2)        
          #print(X)
          X = deconv_layer(X, "deconv2", 10, output_channels=44, keep_prob=1.0, activation=None)        
          #print(X)
          decoded = X
        return tf.stop_gradient(hidden), decoded
      
      with tf.variable_scope("CNN_1"):
        match_history_t1, decode_t1 = auto_encoder(match_history_t1)
      with tf.variable_scope("CNN_2"):
        match_history_t2, decode_t2 = auto_encoder(match_history_t2)
      with tf.variable_scope("CNN_12"):
        match_history_t12, decode_t12 = auto_encoder(match_history_t12)
        
#      with tf.variable_scope("CNN_1"):
#        match_history_t1 = conv_layer(match_history_t1, name="conv1", output_channels=32, keep_prob=0.9)
#        match_history_t1 = conv_layer(match_history_t1, name="conv2", output_channels=16, keep_prob=0.9)
#        match_history_t1 = avg_pool(match_history_t1, name="avgpool")
#        decode_t1 = match_history_t1
#        decode_t1 = deconv_layer(decode_t1, "deconv1", 8, output_channels=32, keep_prob=0.9, stride=2)        
#        decode_t1 = deconv_layer(decode_t1, "deconv2", 10, output_channels=44, keep_prob=1.0, activation=None)        
#      with tf.variable_scope("CNN_2"):
#        match_history_t2 = conv_layer(match_history_t2, name="conv1", output_channels=32, keep_prob=0.9)
#        match_history_t2 = conv_layer(match_history_t2, name="conv2", output_channels=16, keep_prob=0.9)
#        match_history_t2 = avg_pool(match_history_t2, name="avgpool")
#        decode_t2 = match_history_t2
#        decode_t2 = deconv_layer(decode_t2, "deconv1", 8, output_channels=32, keep_prob=0.9, stride=2)        
#        decode_t2 = deconv_layer(decode_t2, "deconv2", 10, output_channels=44, keep_prob=1.0, activation=None)        
#
#      with tf.variable_scope("CNN_12"):
#        match_history_t12 = conv_layer(match_history_t12, name="conv1", output_channels=32, keep_prob=0.9)
#        match_history_t12 = conv_layer(match_history_t12, name="conv2", output_channels=16, keep_prob=1.0)
#        match_history_t12 = avg_pool(match_history_t12, name="avgpool")
#        decode_t12 = match_history_t12
#        decode_t12 = deconv_layer(decode_t12, "deconv1", 8, output_channels=32, keep_prob=0.9, stride=2)        
#        decode_t12 = deconv_layer(decode_t12, "deconv2", 10, output_channels=44, keep_prob=0.9, activation=None)        
        
#      match_history_t1 = 0.0+tf.reshape(match_history_t1, (-1, match_history_t1.shape[1]*match_history_t1.shape[2]))
#      match_history_t2 = 0.0+tf.reshape(match_history_t2, (-1, match_history_t2.shape[1]*match_history_t2.shape[2]))
#      match_history_t12 = 0.0+tf.reshape(match_history_t12, (-1, match_history_t12.shape[1]*match_history_t12.shape[2]))
#      
#      match_history_t1 = tf.stop_gradient(match_history_t1)
#      match_history_t2 = tf.stop_gradient(match_history_t2)
#      match_history_t12 = tf.stop_gradient(match_history_t12)
      
      with tf.variable_scope("Combine"):
        X = tf.concat([features_newgame, match_history_t1, match_history_t2, match_history_t12], axis=1)
        
      with tf.variable_scope("Layer0"):
          X0,Z0 = build_dense_layer(X, 128, mode, regularizer = l2_regularizer(scale=1.0), keep_prob=1.0, batch_norm=True, activation=None, eval_metric_ops=eval_metric_ops)
      
      with tf.variable_scope("Layer1"):
        X1,Z1 = build_dense_layer(X, 128, mode, regularizer = l2_regularizer(scale=0.3), keep_prob=0.85, batch_norm=False, activation=binaryStochastic, eval_metric_ops=eval_metric_ops)
      with tf.variable_scope("Layer2"):
        X2,Z2 = build_dense_layer(X1, 128, mode, add_term = X0*2.0, regularizer = l2_regularizer(scale=0.3), keep_prob=0.85, batch_norm=True, activation=binaryStochastic, eval_metric_ops=eval_metric_ops, batch_scale=False)

      X = X2 # shortcut connection bypassing two non-linear activation functions
      #X = 0.55*X2 + X0 # shortcut connection bypassing two non-linear activation functions
      
#      with tf.variable_scope("Layer3"):
#        #X = tf.stop_gradient(X)
#        X,Z = build_dense_layer(X+0.001, 64, mode, regularizer = l2_regularizer(scale=1.0), keep_prob=0.90, batch_norm=True, activation=binaryStochastic, eval_metric_ops=eval_metric_ops)
      #X = tf.layers.batch_normalization(X, momentum=0.99, center=True, scale=True, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
#        eval_metric_ops.update(variable_summaries(X, "Normalized", mode))
      hidden_layer = X
      
#      with tf.variable_scope("Skymax"):
#        sk_logits,_ = build_dense_layer(X, 49, mode, regularizer = l2_regularizer(scale=1.2), keep_prob=0.8, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)

      with tf.variable_scope("condprob"):
        cond_probs = build_cond_prob_layer(X, labels, mode, regularizer = l2_regularizer(scale=0.009), keep_prob=1.0, eval_metric_ops=eval_metric_ops) 
        #cb1_logits,_ = build_dense_layer(X, 49, mode, regularizer = l2_regularizer(scale=1.2), keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)

      with tf.variable_scope("Softpoints"):
        sp_logits,_ = build_dense_layer(X, 49, mode, 
                                      #regularizer = None, 
                                      regularizer = l2_regularizer(scale=0.009), 
                                      keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        
      with tf.variable_scope("Poisson"):
        outputs,Z = build_dense_layer(X, output_size, mode, 
                                regularizer = l2_regularizer(scale=0.003), 
                                keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        #outputs, index = harmonize_outputs(outputs, label_column_names)
        #eval_metric_ops.update(variable_summaries(outputs, "Outputs_harmonized", mode))

      return outputs, sp_logits, hidden_layer, eval_metric_ops, cond_probs, f_date_round, decode_t1, decode_t2, decode_t12 
        
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
    tc_1d7_goals_f = tf.constant([0,1,2,3,4,5,6], dtype=lambda1.dtype, shape=[1,7])
    tc_1d7_logfactorial_f = tf.constant(logfactorial, dtype=lambda1.dtype, shape=[1,7])

    lambda1 = tf.reshape(lambda1, [-1,1]) # make sure that rank=2
    lambda2 = tf.reshape(lambda2, [-1,1]) # make sure that rank=2
    loglambda1 = tf.log(lambda1)
    loglambda2 = tf.log(lambda2)
    x1 = tf.matmul(loglambda1, tc_1d7_goals_f) - tc_1d7_logfactorial_f # shape=(-1, 7)
    x2 = tf.matmul(loglambda2, tc_1d7_goals_f) - tc_1d7_logfactorial_f # shape=(-1, 7)
    x = tf.concat([x1,x2], axis=1) # shape=(-1, 14)
    matrix = [[1 if j==i1 or j==7+i2 else 0 for j in range(14)] for i1 in range(7) for i2 in range(7) ] # shape=(14, 49)
    t_matrix = tf.constant(matrix, dtype=lambda1.dtype, shape=[49,14])
    t_matrix = tf.transpose(t_matrix)
    x = tf.matmul(x, t_matrix)
    x = tf.exp(x)
    normalization_factor = tf.exp(-lambda1-lambda2) 
    return x, normalization_factor 

  def apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = None):
    if predictions is None:
      predictions = {}
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
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

    predictions.update({
      "p_marg_1":p_marg_1, 
      "p_marg_2":p_marg_2, 
      "p_pred_12":p_pred_12, 
      "ev_points":ev_points, 
      "ev_goals_1":ev_goals_1,
      "ev_goals_2":ev_goals_2, 
      "p_poisson_1":p_poisson_1, 
      "p_poisson_2":p_poisson_2, 
    })
    return predictions
      
  def create_predictions_from_ev_goals(ev_goals_1, ev_goals_2, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
        
        ev_goals_1 = tf.stop_gradient(ev_goals_1)
        ev_goals_2 = tf.stop_gradient(ev_goals_2)
        
        p_pred_12, normalization_factor = calc_joint_poisson_prob(ev_goals_1, ev_goals_2)
        p_pred_12 *= normalization_factor # normalize to 1

        predictions = apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = None)
        ev_points =  predictions["ev_points"]
        a = tf.argmax(ev_points, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
        predictions.update({"pred":pred})
        return predictions
  
  def create_hierarchical_predictions(outputs, logits, t_is_home_bool, tc, mode, p_pred_12 = None, apply_point_scheme=True):
    with tf.variable_scope("Prediction_softpoints"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        if p_pred_12 is None:
          p_pred_12 = tf.nn.softmax(tf.reshape(logits, [-1, 49]))
        
        t_win_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)], name="t_win_mask")
        t_loss_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)], name="t_loss_mask")
        t_draw_mask = tf.stack([1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)], name="t_draw_mask")
        t_tendency_mask = tf.stack([t_win_mask, t_draw_mask, t_loss_mask], axis=1, name="t_tendency_mask")
        t_win_mask = tf.cast(t_win_mask, outputs.dtype)
        t_loss_mask = tf.cast(t_loss_mask, outputs.dtype)
        t_draw_mask = tf.cast(t_draw_mask, outputs.dtype)
        t_tendency_mask = tf.cast(t_tendency_mask, outputs.dtype)
        
#        p_pred_win = tf.reduce_sum(tf.stack([p_pred_12[:,i] for i in range(49) if i // 7 > np.mod(i, 7)], axis=1, name="p_win"), axis=1, name="p_win_sum")
#        p_pred_loss = tf.reduce_sum(tf.stack([p_pred_12[:,i] for i in range(49) if i // 7 < np.mod(i, 7)], axis=1, name="p_loss"), axis=1, name="p_loss_sum")
#        p_pred_draw = tf.reduce_sum(tf.stack([p_pred_12[:,i] for i in range(49) if i // 7 == np.mod(i, 7)], axis=1, name="p_draw"), axis=1, name="p_draw_sum")
#        p_pred_tendency = tf.stack([p_pred_win, p_pred_draw, p_pred_loss], axis=1, name="p_pred_tendency")
#        
        p_pred_tendency = tf.matmul(p_pred_12, t_tendency_mask)
        p_pred_win = p_pred_tendency[:,0]
        p_pred_draw = p_pred_tendency[:,1]
        p_pred_loss = p_pred_tendency[:,2]
        predictions = apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = None)
        
        
        t_pred_tendency_weight = tf.gather(tf.stack([tf.constant(point_scheme[3][::-1]),
                                          tf.constant(point_scheme[3])], axis=0),
                                      tf.cast(t_is_home_bool, tf.int32))
        
#        p_pred_gdiff = []
#        for j in range(13):
#          gdiff = j-6
#          t1 = tf.stack([p_pred_12[:,i] for i in range(49) if (i // 7 - np.mod(i, 7))==gdiff], axis=1, name="p_gdiff"+str(gdiff))
#          t1 = tf.reduce_sum(t1, axis=1, name="p_gdiff_sum"+str(gdiff))
#          p_pred_gdiff.append(t1)
#        p_pred_gdiff = tf.stack(p_pred_gdiff, axis=1, name="p_pred_gdiff")

        t_gdiff_mask = []
        for j in range(13):
          gdiff = j-6
          t1 = tf.stack([1.0 if (i // 7 - np.mod(i, 7))==gdiff else 0.0 for i in range(49) ], name="t_gdiff_mask_"+str(gdiff))
          t_gdiff_mask.append(t1)
        t_gdiff_mask = tf.stack(t_gdiff_mask, axis=1)
        t_gdiff_mask = tf.cast(t_gdiff_mask, p_pred_12.dtype)
        p_pred_gdiff = tf.matmul(p_pred_12, t_gdiff_mask)
        
        p_pred_gtotal = []
        for j in range(13):
          gdiff = j-6
          mask = [1.0 if ((i // 7 - np.mod(i, 7))==gdiff) else 0.0 for i in range(49) ]
          t1 = tf.stack([mask[i] for i in range(49) ], axis=0, name="p_gdiff_goals"+str(gdiff))
          p_pred_gtotal.append(t1)
        p_pred_gtotal = tf.stack(p_pred_gtotal, axis=0, name="p_pred_gtotal")

        t_pred_tendency_weight = tf.cast(t_pred_tendency_weight, p_pred_tendency.dtype)
        if apply_point_scheme:
          pred_tendency = tf.argmax(p_pred_tendency * t_pred_tendency_weight, axis=1, name="pred_tendency_weighted")
          #pred_tendency = tf.argmin(t_pred_tendency_weight/(p_pred_tendency+1e-7), axis=1, name="pred_tendency_weighted")
        else:
          pred_tendency = tf.argmax(p_pred_tendency , axis=1, name="pred_tendency")

        pred_gdiff = tf.where(tf.equal(pred_tendency,2),
                              tf.argmax(p_pred_gdiff[:,:6], axis=1, name="pred_gdiff_loss") - 6,
                              tf.where(tf.equal(pred_tendency,0),
                                       tf.argmax(p_pred_gdiff[:,7:], axis=1, name="p_pred_gdiff_win") + 1,
                                       tf.argmax(p_pred_gdiff[:,6:7], axis=1, name="p_pred_gdiff_draw"))
                              , name="pred_gdiff")
        
        tgather = tf.gather(p_pred_gtotal, pred_gdiff+6, axis=0)
        tgather = tf.cast(tgather, p_pred_12.dtype)
        pred_gtotal = tf.argmax(p_pred_12 * tgather, axis=1, name="pred_gtotal")
        pred = tf.reshape(tf.stack([pred_gtotal // 7, tf.mod(pred_gtotal, 7)], axis=1), (-1,2))
        pred = tf.cast(pred, tf.int32)
        
        if True:
          # try fixed scheme based on probabilities
          pred = create_fixed_scheme_prediction(outputs, p_pred_12, t_is_home_bool, mode)

        predictions.update({
          "logits":logits,
          "pred":pred,
          "p_pred_win":p_pred_win, 
          "p_pred_loss":p_pred_loss, 
          "p_pred_draw":p_pred_draw, 
          "p_pred_gdiff":p_pred_gdiff, 
          "pred_tendency":pred_tendency, 
          "pred_gdiff":pred_gdiff, 
          "pred_gtotal":pred_gtotal, 
        })
        return predictions
  
  def create_fixed_scheme_prediction(outputs, p_pred_12, t_is_home_bool, mode):
        t_win_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)], name="t_win_mask")
        t_loss_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)], name="t_loss_mask")
        t_draw_mask = tf.stack([1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)], name="t_draw_mask")
        t_both_teams_score_mask = tf.stack([1.0 if (i // 7) > 0 and np.mod(i, 7) > 0 else 0.0  for i in range(49)], name="t_both_teams_score_mask")
        t_tendency_mask = tf.stack([t_win_mask, t_draw_mask, t_loss_mask, t_both_teams_score_mask], axis=1, name="t_tendency_mask")

        t_win_mask = tf.cast(t_win_mask, outputs.dtype)
        t_loss_mask = tf.cast(t_loss_mask, outputs.dtype)
        t_draw_mask = tf.cast(t_draw_mask, outputs.dtype)
        t_tendency_mask = tf.cast(t_tendency_mask, outputs.dtype)
        t_both_teams_score_mask = tf.cast(t_both_teams_score_mask, outputs.dtype)
        
        #T1_GFT = tf.exp(outputs[:,0])
        #T2_GFT = tf.exp(outputs[:,1])
        
        p_pred_tendency = tf.matmul(p_pred_12, t_tendency_mask)
        p_pred_win = p_pred_tendency[:,0]
        #p_pred_draw = p_pred_tendency[:,1]
        p_pred_loss = p_pred_tendency[:,2]
        p_pred_both = p_pred_tendency[:,3]
        #hg = tf.where(t_is_home_bool, T1_GFT, T2_GFT)
        #ag = tf.where(t_is_home_bool, T2_GFT, T1_GFT)

        with tf.variable_scope("diff_p"):
          diffp = p_pred_loss - p_pred_win
          diffp = tf.where(t_is_home_bool, diffp, -diffp)
          diffp = tf.layers.batch_normalization(tf.expand_dims(diffp, axis=1), axis=1, scale=True, center=True, momentum=0.99, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
          diffp = tf.squeeze(diffp, axis=1)
          tf.summary.histogram("diffp", diffp)
        
        with tf.variable_scope("p_both"):
          p_pred_both = tf.layers.batch_normalization(tf.expand_dims(p_pred_both, axis=1) , axis=1, momentum=0.99, center=True, scale=True, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
          p_pred_both = tf.squeeze(p_pred_both, axis=1)
          tf.summary.histogram("p_pred_both", p_pred_both)

        pred = tf.stack([0*p_pred_loss, 0*p_pred_win ], axis=1)
        if point_scheme[0][0]==5: # Sky
          pred = tf.cast(pred, tf.int32)+tf.constant([[0,3]])
#          pred = tf.where(diffp< 0.708641107, pred*0+tf.constant([[0,2]]), pred)
#          pred = tf.where(diffp< 0.329863088, tf.where(p_pred_both<0.525, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
#          pred = tf.where(diffp< 0.001331164, tf.where(p_pred_both<0.525, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
#          pred = tf.where(diffp<-0.489031706, pred*0+tf.constant([[2,0]]), pred)
#          pred = tf.where(diffp<-0.601062863, pred*0+tf.constant([[3,0]]), pred)
#          pred = tf.where(diffp<-0.789108038, pred*0+tf.constant([[4,0]]), pred)

          pred = tf.where(diffp< 2.0, pred*0+tf.constant([[0,2]]), pred)
          pred = tf.where(diffp< 1.0, tf.where(p_pred_both<-0.1, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
          pred = tf.where(diffp< 0.001, tf.where(p_pred_both<-0.1, pred*0+tf.constant([[0,0]]), pred*0+tf.constant([[1,1]])), pred) # ag<1.1
          pred = tf.where(diffp< -0.001, tf.where(p_pred_both<-0.1, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
          pred = tf.where(diffp<-1.0, pred*0+tf.constant([[2,0]]), pred)
          pred = tf.where(diffp<-2.0, pred*0+tf.constant([[3,0]]), pred)
          pred = tf.where(diffp<-2.5, pred*0+tf.constant([[4,0]]), pred)

          pred = tf.where(t_is_home_bool, pred, pred[:,::-1])
        else:
          pred = tf.cast(pred, tf.int32)+tf.constant([[0,2]])
          pred = tf.where(diffp< 0.32753985, tf.where(p_pred_both<0.525, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
          pred = tf.where(diffp< 0.10371171, tf.where(p_pred_both<0.54, pred*0+tf.constant([[0,0]]), pred*0+tf.constant([[1,1]])), pred)# tf.logical_and(ag<1.3, hg<1.3)
          pred = tf.where(diffp<-0.01979582, tf.where(p_pred_both<0.525, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
          pred = tf.where(diffp<-0.48798187, pred*0+tf.constant([[2,0]]), pred)
          pred = tf.where(t_is_home_bool, pred, pred[:,::-1])
        return pred

  def create_fixed_scheme_prediction_new(p_pred_12, t_is_home_bool, mode):
        t_win_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)], name="t_win_mask")
        t_loss_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)], name="t_loss_mask")
        t_draw_mask = tf.stack([1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)], name="t_draw_mask")
        t_both_teams_score_mask = tf.stack([1.0 if (i // 7) > 0 and np.mod(i, 7) > 0 else 0.0  for i in range(49)], name="t_both_teams_score_mask")
        t_tendency_mask = tf.stack([t_win_mask, t_draw_mask, t_loss_mask, t_both_teams_score_mask], axis=1, name="t_tendency_mask")
        
        t_win_mask = tf.cast(t_win_mask, p_pred_12.dtype)
        t_loss_mask = tf.cast(t_loss_mask, p_pred_12.dtype)
        t_draw_mask = tf.cast(t_draw_mask, p_pred_12.dtype)
        t_tendency_mask = tf.cast(t_tendency_mask, p_pred_12.dtype)
        t_both_teams_score_mask = tf.cast(t_both_teams_score_mask, p_pred_12.dtype)

        #T1_GFT = tf.exp(outputs[:,0])
        #T2_GFT = tf.exp(outputs[:,1])
        
        p_pred_tendency = tf.matmul(p_pred_12, t_tendency_mask)
        p_pred_win = p_pred_tendency[:,0]
        #p_pred_draw = p_pred_tendency[:,1]
        p_pred_loss = p_pred_tendency[:,2]
        p_pred_both = p_pred_tendency[:,3]
        #hg = tf.where(t_is_home_bool, T1_GFT, T2_GFT)
        #ag = tf.where(t_is_home_bool, T2_GFT, T1_GFT)
        diffp = tf.where(t_is_home_bool, p_pred_loss - p_pred_win, p_pred_win - p_pred_loss)
        with tf.variable_scope("diffp"):
          diffp = tf.layers.batch_normalization(tf.expand_dims(diffp, axis=1) , axis=1, momentum=0.99, center=True, scale=True, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
          diffp = tf.squeeze(diffp, axis=1)
          tf.summary.histogram("diffp", diffp)
        with tf.variable_scope("p_both"):
          p_pred_both = tf.layers.batch_normalization(tf.expand_dims(p_pred_both, axis=1) , axis=1, momentum=0.99, center=True, scale=True, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
          p_pred_both = tf.squeeze(p_pred_both, axis=1)
          tf.summary.histogram("p_pred_both", p_pred_both)
        
        pred = tf.stack([0*p_pred_loss, 0*p_pred_win ], axis=1)
        if point_scheme[0][0]==5: # Sky
#          pred = tf.cast(pred, tf.int32)+tf.constant([[0,4]])
#          pred = tf.where(diffp< 0.65, pred*0+tf.constant([[0,3]]), pred)
#          pred = tf.where(diffp< 0.60, pred*0+tf.constant([[0,2]]), pred)
#          pred = tf.where(diffp< 0.52, tf.where(p_pred_both<0.525, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
#          pred = tf.where(diffp< 0.02, tf.where(p_pred_both<0.54, pred*0+tf.constant([[0,0]]), pred*0+tf.constant([[1,1]])), pred)# tf.logical_and(ag<1.3, hg<1.3)
#          pred = tf.where(diffp< 0.00, tf.where(p_pred_both<0.525, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
#          pred = tf.where(diffp<-0.30, pred*0+tf.constant([[2,0]]), pred)
#          pred = tf.where(diffp<-0.60, pred*0+tf.constant([[3,0]]), pred)
#          pred = tf.where(diffp<-0.80, pred*0+tf.constant([[4,0]]), pred)
#          pred = tf.where(t_is_home_bool, pred, pred[:,::-1])
          
          if False:
            # use the settings previously found in dynamic cutoff strategy
            pred = tf.cast(pred, tf.int32)+tf.constant([[0,4]])
            pred = tf.where(diffp< 0.95, pred*0+tf.constant([[0,3]]), pred)
            pred = tf.where(diffp< 0.90, pred*0+tf.constant([[0,2]]), pred)
            pred = tf.where(diffp< 0.52, tf.where(p_pred_both<0.525, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
            pred = tf.where(diffp< 0.07, tf.where(p_pred_both<0.54, pred*0+tf.constant([[0,0]]), pred*0+tf.constant([[1,1]])), pred)# tf.logical_and(ag<1.3, hg<1.3)
            pred = tf.where(diffp< 0.06, tf.where(p_pred_both<0.525, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
            pred = tf.where(diffp<-0.56, pred*0+tf.constant([[2,0]]), pred)
            pred = tf.where(diffp<-0.70, pred*0+tf.constant([[3,0]]), pred)
            pred = tf.where(diffp<-0.95, pred*0+tf.constant([[4,0]]), pred)
            pred = tf.where(t_is_home_bool, pred, pred[:,::-1])
          else:
            pred = tf.cast(pred, tf.int32)+tf.constant([[0,4]])
            pred = tf.where(diffp< 2.5, pred*0+tf.constant([[0,3]]), pred)
            pred = tf.where(diffp< 2.0, pred*0+tf.constant([[0,2]]), pred)
            pred = tf.where(diffp< 1.0, tf.where(p_pred_both<-0.1, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
            pred = tf.where(diffp< 0.01, tf.where(p_pred_both<-0.1, pred*0+tf.constant([[0,0]]), pred*0+tf.constant([[1,1]])), pred)# tf.logical_and(ag<1.3, hg<1.3)
            pred = tf.where(diffp< -0.01, tf.where(p_pred_both<-0.1, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
            pred = tf.where(diffp<-1.0, pred*0+tf.constant([[2,0]]), pred)
            pred = tf.where(diffp<-1.7, pred*0+tf.constant([[3,0]]), pred)
            pred = tf.where(diffp<-2.3, pred*0+tf.constant([[4,0]]), pred)
            pred = tf.where(t_is_home_bool, pred, pred[:,::-1])
        else:
          # Pistor
          pred = tf.cast(pred, tf.int32)+tf.constant([[0,4]])
          pred = tf.where(diffp< 0.67, pred*0+tf.constant([[0,3]]), pred)
          pred = tf.where(diffp< 0.60, pred*0+tf.constant([[0,2]]), pred)
          pred = tf.where(diffp< 0.52, tf.where(p_pred_both<0.525, pred*0+tf.constant([[0,1]]), pred*0+tf.constant([[1,2]])), pred) # hg<1.1
          pred = tf.where(diffp< 0.11, tf.where(p_pred_both<0.54, pred*0+tf.constant([[0,0]]), pred*0+tf.constant([[1,1]])), pred)# tf.logical_and(ag<1.3, hg<1.3)
          pred = tf.where(diffp<-0.04, tf.where(p_pred_both<0.525, pred*0+tf.constant([[1,0]]), pred*0+tf.constant([[2,1]])), pred) # ag<1.1
          pred = tf.where(diffp<-0.50, pred*0+tf.constant([[2,0]]), pred)
          pred = tf.where(diffp<-0.70, pred*0+tf.constant([[3,0]]), pred)
          pred = tf.where(diffp<-0.75, pred*0+tf.constant([[4,0]]), pred)
          pred = tf.where(t_is_home_bool, pred, pred[:,::-1])
        return pred


  def create_ensemble_predictions(ensemble_logits, predictions, t_is_home_bool, tc):
    with tf.variable_scope("Prediction"):
#        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        p_ensemble = tf.nn.softmax(tf.reshape(ensemble_logits, [-1, len(ens_prefix_list)*2]))
        # limit each strategies contribution to a range of 5% and 15%
        p_ensemble = tf.maximum(p_ensemble, 0.15) 
        p_ensemble = tf.minimum(p_ensemble, 0.05) 
        p_ensemble = p_ensemble / tf.reduce_sum(p_ensemble, axis=1, keepdims=True) # normalize to 100%
        
        #prefix_list = ["p1/", "p2/", "p3/", "p4/", "p5/", "p7/", "sp/", "sm/", "p1pt/", "p2pt/", "p4pt/", "sppt/", "smpt/"]
        t_home_preds = tf.stack([predictions[p+"pred"][0::2] for p in ens_prefix_list], axis=1)
        t_away_preds = tf.stack([predictions[p+"pred"][1::2] for p in ens_prefix_list], axis=1)
        t_away_preds = tf.stack([t_away_preds[:,:,1], t_away_preds[:,:,0]], axis=2) # swap home and away goals
        t_preds = tf.concat([t_home_preds, t_away_preds], axis=1)
        # expand p_ensemble and t_preds from half lenght to full length - else TF will complain in PREDICT mode
        index = tf.range(0, 2 * tf.shape(p_ensemble)[0])
        index = index // 2 # every index element repeats twice
        p_ensemble = tf.gather(p_ensemble, index)
        t_preds = tf.gather(t_preds, index)
        
        # majority vote of the results - weighted by p_ensemble
        t_allvotes = tf.stack([tf.cast(tf.equal(t_preds[:,:,0]*7+t_preds[:,:,1], i), tf.float32) for i in range(49)], axis=2) # 0/1 array of predictions - axes: 0=batch, 1=strategy, 2=predicted score
        t_allvotes = t_allvotes * tf.expand_dims(p_ensemble, axis=2) # multiply with strategy weights
        t_votes_per_score = tf.reduce_sum(t_allvotes, axis=1)        
        pred = tf.argmax(t_votes_per_score, axis=1, output_type=tf.int32)
        
        # how did the strategies contribute to this particular score?
        # pick the highest weight among the contributors
        pick_index = tf.range(0, tf.shape(t_allvotes)[0])*49+pred # axis0: batch index 0..n-1, axis1: the predicted score (chosen by majority voting)
        t_pick_matrix = tf.reshape(tf.transpose(t_allvotes, [0,2,1]), [tf.shape(t_allvotes)[0]*49, tf.shape(t_allvotes)[1]]) # axes: 0=batch * predicted score, 1=strategy
        t_contributions = tf.gather(t_pick_matrix, pick_index) # axis0: batch index 0..n-1, axis1: the strategy's weight towards the predicted score 
        selected_strategy = tf.argmax(t_contributions, axis=1, output_type=tf.int32)
        # convert prediction index back to home goal / away goals
        pred = tf.stack([pred//7, tf.mod(pred, 7)], axis=1)
        
        predictions = {
          "ensemble_logits":tf.gather(ensemble_logits, index),
          "selected_strategy":selected_strategy, 
          "pred":pred,
          "p_ensemble":p_ensemble, 
        }
        return predictions
  
  def point_maximization_layer(outputs, X, prefix, t_is_home_bool, tc, mode):
      with tf.variable_scope("PointMax_"+prefix):
          tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
          X = tf.stop_gradient(X)
          with tf.variable_scope("Layer1"):
            X,_ = build_dense_layer(X, output_size=10, mode=mode, regularizer = None, keep_prob=0.5, batch_norm=True, activation=tf.nn.relu) #, eval_metric_ops=None, use_bias=None, add_term=None, batch_scale=True)        
          with tf.variable_scope("Layer2"):
            X,_ = build_dense_layer(X, output_size=10, mode=mode, regularizer = None, keep_prob=0.5, batch_norm=True, activation=tf.nn.relu) #, eval_metric_ops=None, use_bias=None, add_term=None, batch_scale=True)        
          with tf.variable_scope("Layer3"):
            X,_ = build_dense_layer(X, output_size=49, mode=mode, regularizer = None, keep_prob=1.0, batch_norm=False, activation=None)
          
          logits=X
          predictions = create_hierarchical_predictions(outputs, logits, t_is_home_bool, tc, mode)
          #predictions = create_predictions(logits, logits, t_is_home_bool, tc, True)

          p_pred_12 = tf.nn.softmax(logits)
          predictions = apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = predictions)
          
          if True:
            # try fixed scheme based on probabilities
            pred = create_fixed_scheme_prediction(outputs, p_pred_12, t_is_home_bool, mode)
            predictions.update({"pred":pred})

          # this should be filled from underlying strategy
          predictions.pop("ev_goals_1")
          predictions.pop("ev_goals_2")

          return predictions
  
  def create_model_regularization_metrics(eval_metric_ops, predictions, labels, mode):
    metrics = {}
    for w in tf.get_collection(tf.GraphKeys.WEIGHTS):
      wname = w.name[6:-2]
      with tf.variable_scope("regularization"):
        metrics.update(variable_summaries(w, wname, mode))
      metrics.update(collect_summary("regularization", wname+"_L1", mode, tensor=tf.abs(w)))
      metrics.update(collect_summary("regularization", wname+"_L2", mode, tensor=tf.square(w)))

    l_tendencies = [tf.reduce_mean(predictions[p + "is_tendency"]) for p in prefix_list]
    t_tendencies = tf.stack(l_tendencies)
    current_tendency = tf.reduce_max(t_tendencies)
    metrics.update(collect_summary("regularization", "current_tendency", mode, tensor=current_tendency))

    is_win  = labels[:,18]
    is_loss = labels[:,20]
    pred_win  = tf.exp(predictions["outputs_poisson"][:,18])
    pred_loss = tf.exp(predictions["outputs_poisson"][:,20])
    
    current_winloss_corr = corrcoef(is_win-is_loss, pred_win-pred_loss) 
    metrics.update(collect_summary("regularization", "current_winloss_corr", mode, tensor=current_winloss_corr))
    
    pred_gs = predictions["outputs_poisson"][::2, 0]
    gs_mean, gs_variance = tf.nn.moments(pred_gs, axes=[0])
    # variance should be in the order of 1.0
    #base_noise_factor /= tf.sqrt(tf.minimum(gs_variance, 0.1))
    #base_noise_factor *= 0.01
    #base_noise_factor *= 0.1
    metrics.update(collect_summary("regularization", "gs_variance", mode, tensor=gs_variance))
    return metrics

  def create_poisson_correlation_metrics(outputs, t_labels, mode, mask = None, section="poisson"):
    metrics={}
    for i,col in enumerate(label_column_names):
      x = tf.exp(outputs[:,i])
      y = t_labels[:,i]
      if mask is not None:
        # mean imputation
        x = tf.where(mask[:,i]==0.0, tf.zeros_like(x)+tf.reduce_mean(x), x)
        y = tf.where(mask[:,i]==0.0, tf.zeros_like(y)+tf.reduce_mean(y), y)
      metrics.update(collect_summary(section, col.replace(":", "_"), mode, tensor=corrcoef(x, y)))
    return metrics
  
  def create_autoencoder_losses(loss, decode_t1, decode_t2, decode_t12, features, t_labels, mode ):
      ncol = int(t_labels.shape[1])
      maxseqlen = int(features["match_history_t1"].shape[1])
      match_history_t1_seqlen =  10*features['newgame'][:,0]
      match_history_t2_seqlen =  10*features['newgame'][:,2]
      match_history_t12_seqlen = 10*features['newgame'][:,3]

      poisson_column_weights = tf.ones(shape=[1, 1, ncol], dtype=t_labels.dtype)
#      poisson_column_weights = tf.concat([
#          poisson_column_weights[:,:,0:4] *3,
#          poisson_column_weights[:,:,4:16],
#          poisson_column_weights[:,:,16:21] *3,
#          poisson_column_weights[:,:,21:],
#          ], axis=2)
      
      # do not include null sequence positions into the loss
      def sequence_len_mask(l):
        return tf.expand_dims(1.0-tf.sequence_mask(maxseqlen-l, maxlen=maxseqlen, dtype=t_labels.dtype), axis=2)

      #eval_ae_loss_ops={}
      m1 = sequence_len_mask(match_history_t1_seqlen)
      m2 = sequence_len_mask(match_history_t2_seqlen)
      m12 = sequence_len_mask(match_history_t12_seqlen)
      m_total = tf.reduce_sum(m1+m2+m12, axis=1)
      m_total = tf.reduce_mean(m_total/30) # normalization constant

      all_outputs = tf.concat([tf.reshape(decode_t1, [-1,ncol]),
                               tf.reshape(decode_t2, [-1,ncol]),
                               tf.reshape(decode_t12, [-1,ncol])], axis=0) 
      all_labels = tf.concat([tf.reshape(features["match_history_t1"][:,:,-ncol:], [-1,ncol]),
                              tf.reshape(features["match_history_t2"][:,:,-ncol:], [-1,ncol]),
                              tf.reshape(features["match_history_t12"][:,:,-ncol:], [-1,ncol])], axis=0)
      
      all_masks = tf.concat([tf.reshape(m1, [-1,ncol]),
                               tf.reshape(m2, [-1,ncol]),
                               tf.reshape(m12, [-1,ncol])], axis=0) 
      eval_ae_loss_ops = create_poisson_correlation_metrics(outputs=all_outputs, t_labels=all_labels, mode=mode, mask = all_masks, section="autoencoder")
      
      l_ae_loglike_poisson = \
          sequence_len_mask(match_history_t1_seqlen) * tf.nn.log_poisson_loss(targets=features["match_history_t1"][:,:,-ncol:], log_input=decode_t1) + \
          sequence_len_mask(match_history_t2_seqlen) * tf.nn.log_poisson_loss(targets=features["match_history_t2"][:,:,-ncol:], log_input=decode_t2) + \
          sequence_len_mask(match_history_t12_seqlen) * tf.nn.log_poisson_loss(targets=features["match_history_t12"][:,:,-ncol:], log_input=decode_t12)
      l_ae_loglike_poisson = tf.reduce_sum(l_ae_loglike_poisson, axis=1)
      l_ae_loglike_poisson *= poisson_column_weights
      l_ae_loglike_poisson /= m_total
      loss += tf.reduce_mean(l_ae_loglike_poisson)
      eval_ae_loss_ops.update(collect_summary("losses", "l_ae_loglike_poisson", mode, tensor=l_ae_loglike_poisson))
      return eval_ae_loss_ops, loss 

  def create_losses_RNN(outputs, sp_logits, cond_probs, t_labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool, eval_metric_ops):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    h1_logits, h2_logits, p_pred_h1, label_features_h1, p_pred_h2, label_features_h2, t_mask, test_p_pred_12_h2, p_pred_12_h2 = cond_probs
    
    with tf.variable_scope("Prediction"):
      labels_float  = t_labels[:,0]
      labels_float2 = t_labels[:,1]
      labels = tf.cast(labels_float, tf.int32)
      labels2 = tf.cast(labels_float2, tf.int32)
      # reduce to 6 goals max. for training
      gs = tf.minimum(labels,6)
      gc = tf.minimum(labels2,6)
  
      labels_float_1h  = t_labels[:,2]
      labels_float2_1h = t_labels[:,3]
      labels_1h = tf.cast(labels_float_1h, tf.int32)
      labels2_1h = tf.cast(labels_float2_1h, tf.int32)
      # reduce to 6 goals max. for training
      gs_1h = tf.minimum(labels_1h,6)
      gc_1h = tf.minimum(labels2_1h,6)
    
    with tf.variable_scope("Losses"):
      outputs = tf.clip_by_value(outputs, -10, 10)
      sp_logits = tf.clip_by_value(sp_logits, -10, 10)
      
      match_date = features["newgame"][:,4]
      sequence_length = features['newgame'][:,0]
      #t_weight = tf.exp(0.5*match_date) * (sequence_length + 0.05) # sequence_length ranges from 0 to 1 - depending on the number of prior matches of team 1
      t_weight = 1.0 + 0.0 * tf.exp(0.5*match_date) * (sequence_length + 0.05) # sequence_length ranges from 0 to 1 - depending on the number of prior matches of team 1

      # result importance = log(1/frequency) - mean adjusted to 1.0
      # 1:1 has low importance, 3:2 much higher, 5:0 overvalued but rare
#      result_importance = tf.constant([0.610699463302517, 0.636624134693228, 0.688478206502995, 0.861174384773783, 1.09447971749187, 1.45916830182096, 1.45916830182096, 0.565089593736748, 0.490552492978211, 0.620077094945206, 0.760233563222618, 0.992062709888034, 1.31889788143772, 1.49409788361176, 0.584692048161506, 0.540777816952666, 0.664928108155794, 0.870617157643794, 1.10781260749576, 1.4022220211579, 1.53541081027154, 0.690646584367668, 0.698403607367345, 0.828902824753833, 0.983969215334149, 1.45916830182096, 1.58597374606561, 1.90009939460063, 0.888152492546633, 0.865846586541708, 1.00917224994398, 1.33703505934424, 1.45916830182096, 1.65116070787927, 0.226593757678754, 1.12197933011558, 1.11478527326307, 1.30210547755345, 1.45916830182096, 1.74303657033312, 0.226593757678754, 0.226593757678754, 1.30210547755345, 1.4022220211579, 1.4289109217981, 1.65116070787927, 0.226593757678754, 0.226593757678754, 0.226593757678754], dtype=t_weight.dtype)
#      row_weight = tf.gather(result_importance, gs*7+gc, name="select_importance")
#      t_weight = t_weight * row_weight

      # result importance = corrcoef 
      # 1:1 has low importance as it is difficult to predict, 3:0 much higher
      result_importance = tf.constant([0.048, 0.058, 0.140, 0.125, 0.1, 0.1, 0.1,
                                       0.057, 0.046, 0.055, 0.102, 0.1, 0.1, 0.1,
                                       0.139, 0.053, 0.012, 0.025, 0.1, 0.1, 0.1,
                                       0.125, 0.103, 0.025, 0.001, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.001, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.001, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001,
                                       ], dtype=t_weight.dtype)
      result_importance = result_importance * 10.0 # scaling to mean about 1.0
      row_weight = tf.gather(result_importance, gs*7+gc, name="select_importance")
      
      # draws are weighed only with 20%
      row_weight = tf.where(gs==gc, row_weight*0.0+0.2, row_weight*0.0+1.0)
      t_weight = t_weight * row_weight
      t_weight_total = tf.reduce_mean(t_weight)
      t_weight = t_weight / t_weight_total

      l_loglike_poisson = tf.expand_dims(t_weight, axis=1) * tf.nn.log_poisson_loss(targets=t_labels, log_input=outputs)
      poisson_column_weights = tf.ones(shape=[1, t_labels.shape[1]], dtype=t_weight.dtype)
      poisson_column_weights = tf.concat([
          poisson_column_weights[:,0:4] *3,
          poisson_column_weights[:,4:16],
          poisson_column_weights[:,16:20] *3,
          poisson_column_weights[:,20:],
          ], axis=1)
      l_loglike_poisson *= poisson_column_weights
    
      l_loglike_poisson = tf.expand_dims(t_weight, axis=1) * tf.nn.log_poisson_loss(targets=t_labels, log_input=outputs)
      
      l_softmax_1h = t_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gs_1h*7+gc_1h, logits=h1_logits)
      l_softmax_2h = t_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gs*7+gc, logits=h2_logits)
      p_full    = tf.one_hot(gs*7+gc, 49, dtype=t_weight.dtype)

      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    
      
      t_is_draw = tf.logical_and(tf.logical_not(t_is_home_win_bool), tf.logical_not(t_is_home_loss_bool), name="t_is_draw")
      t_is_draw_f = tf.cast(t_is_draw, dtype=t_weight.dtype)
#      t_is_win_f = tf.cast(tf.greater(gs, gc), tf.float32)
#      t_is_loss_f = tf.cast(tf.less(gs, gc), tf.float32)
      epsilon = 1e-7
      l_tendency = -t_weight * tf.where(t_is_home_bool,
                                       point_scheme[3][0]*tf.log(predictions["sp/p_pred_win"]+epsilon)*tf.cast(t_is_home_win_bool, t_weight.dtype) + 
                                       point_scheme[3][1]*tf.log(predictions["sp/p_pred_draw"]+epsilon)*t_is_draw_f + 
                                       point_scheme[3][2]*tf.log(predictions["sp/p_pred_loss"]+epsilon)*tf.cast(t_is_home_loss_bool, t_weight.dtype) ,
                                       
                                       point_scheme[3][0]*tf.log(predictions["sp/p_pred_loss"]+epsilon)*tf.cast(t_is_home_win_bool, t_weight.dtype) + 
                                       point_scheme[3][1]*tf.log(predictions["sp/p_pred_draw"]+epsilon)*t_is_draw_f + 
                                       point_scheme[3][2]*tf.log(predictions["sp/p_pred_win"]+epsilon)*tf.cast(t_is_home_loss_bool, t_weight.dtype) 
                                       )

      t_gdiff = tf.one_hot(gs-gc+6, 13, name="t_gdiff", dtype=t_weight.dtype)
      t_ignore_draw_mask = tf.stack([1.0]*6+[0.0]+[1.0]*6)
      t_ignore_draw_mask = tf.cast(t_ignore_draw_mask, dtype=t_weight.dtype)
      l_gdiff = t_weight * tf.reduce_sum(-tf.log(predictions["sp/p_pred_gdiff"]+epsilon) * t_gdiff * t_ignore_draw_mask, axis=1) 
      
      t_gdiff_select_mask = []
      for k in range(7):
        t1 = tf.stack([tf.cast(tf.logical_and(i//7==k, tf.equal(gs-gc, i // 7 - np.mod(i, 7))), t_weight.dtype) for i in range(49) ], name="t_gdiff_select_mask_"+str(k), axis=1)
#            1.0 if (i // 7 - np.mod(i, 7))==gs-gc & i//7==k else 0.0 for i in range(49) ], name="t_gdiff_select_mask_"+str(k))
        t_gdiff_select_mask.append(t1)
      t_gdiff_select_mask = tf.stack(t_gdiff_select_mask, axis=0)
      t_gdiff_select_mask = tf.reduce_sum(t_gdiff_select_mask, axis=0, name="t_gdiff_select_mask")
      t_gdiff_select_mask = tf.cast(t_gdiff_select_mask, dtype=t_weight.dtype)
      
      def softmax_loss_masked(probs, labels, mask, weight):
        p_pred_goals_abs = probs * mask # select only scores within the correct goal difference, ignore all others
        p_pred_goals_abs = p_pred_goals_abs / (tf.reduce_sum(p_pred_goals_abs, axis=1, keepdims=True)+1e-7) # normalize sum = 1
        return (weight * tf.reduce_sum(-tf.log(p_pred_goals_abs+1e-7) * labels, axis=1)) 

      # select only scores within the correct goal difference, ignore all others
      l_gfull = softmax_loss_masked(predictions["sp/p_pred_12"], p_full, t_gdiff_select_mask, t_weight) 

      reg_eval_metric_ops = create_model_regularization_metrics(eval_metric_ops, predictions, t_labels, mode)
      
      l_loglike_poisson = tf.reduce_sum(l_loglike_poisson, axis=1)
      loss = tf.reduce_mean(l_loglike_poisson)
      
      loss += 3.1*tf.reduce_mean(l_softmax_1h) 
      loss += 30.3*tf.reduce_mean(l_softmax_2h) 
      
      t_win_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)], name="t_win_mask")
      t_loss_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)], name="t_loss_mask")
      t_draw_mask = tf.stack([1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)], name="t_draw_mask")
      t_win_mask = tf.cast(t_win_mask, dtype=t_weight.dtype)
      t_loss_mask = tf.cast(t_loss_mask, dtype=t_weight.dtype)
      t_draw_mask = tf.cast(t_draw_mask, dtype=t_weight.dtype)
      
      loss += tf.reduce_mean(2*l_tendency)  
      loss += tf.reduce_mean(0.5*l_gdiff)  
      loss += tf.reduce_mean(0.5*l_gfull)  
      
      # draws have only 7 points to contribute, wins and losses have 21 points each
      t_draw_bias_mask = tf.stack([3.0 if i // 7 == np.mod(i, 7) else 1.0  for i in range(49)], name="t_draw_bias_mask")
      t_draw_bias_mask = tf.expand_dims(t_draw_bias_mask, axis=0)
      t_draw_bias_mask = tf.cast(t_draw_bias_mask, dtype=t_weight.dtype)
      
      pd.DataFrame({
          "win":[1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)],
          "loss":[1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)],
          "draw":[1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)]}).values
      
      
      t_lt2_mask = tf.cast(tf.stack([1.0 if i // 7 + np.mod(i, 7) < 2 else 0.0  for i in range(49)], name="t_lt2_mask"), dtype=t_weight.dtype)
      t_eq2_mask = tf.cast(tf.stack([1.0 if i // 7 + np.mod(i, 7) == 2 else 0.0  for i in range(49)], name="t_eq2_mask"), dtype=t_weight.dtype)
      t_gt2_mask = tf.cast(tf.stack([1.0 if (i // 7 + np.mod(i, 7)) in [3,4] and (i // 7 in [1,2]) and (np.mod(i, 7) in [1,2]) else 0.0  for i in range(49)], name="t_gt2_mask"), dtype=t_weight.dtype)
      t_gt3_mask = tf.cast(tf.stack([1.0 if i // 7  >= 3 or np.mod(i, 7) >= 3 else 0.0  for i in range(49)], name="t_gt3_mask"), dtype=t_weight.dtype)
      
      t_wdl_mask = tf.cast(tf.stack([t_win_mask, t_draw_mask, t_loss_mask], axis=1, name="t_wdl_mask"), dtype=t_weight.dtype)
      t_leg2_mask = tf.cast(tf.stack([t_lt2_mask, t_eq2_mask, t_gt2_mask, t_gt3_mask ], axis=1, name="t_leg2_mask"), dtype=t_weight.dtype)
      
      t_wdl_labels = tf.matmul(p_full, t_wdl_mask)   
      t_leg2_labels = tf.matmul(p_full, t_leg2_mask)   
      #t_wdl_mask = tf.print(t_wdl_mask, [t_wdl_mask])
#      a0 = tf.Assert(tf.reduce_any(tf.equal(tf.reduce_sum(p_full, axis=1, name="test_p_full"), 1)), [p_full], name="assert_p_full")
#      a1 = tf.Assert(tf.reduce_any(tf.equal(tf.reduce_sum(t_wdl_mask, axis=1, name="test_t_wdl_mask"), 1)), [t_wdl_mask ], name="assert_t_wdl_mask")
#      a2 = tf.Assert(tf.reduce_any(tf.equal(tf.reduce_sum(t_leg2_mask, axis=1, name="test_t_leg2_mask"), 1)), [p_full], name="assert_t_leg2_mask")
#      a3 = tf.Assert(tf.reduce_any(tf.equal(tf.reduce_sum(t_wdl_labels, axis=1, name="test_wdl"), 1)), [t_wdl_labels], name="assert_wdl")
#      a4 = tf.Assert(tf.reduce_any(tf.equal(tf.reduce_sum(t_leg2_labels, axis=1, name="test_leg2"), 1)), [t_leg2_labels], name="assert_leg2")
#
#      ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, a0) # control dependencies
#      ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, a1) # control dependencies
#      ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, a2) # control dependencies
#      ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, a3) # control dependencies
#      ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, a4) # control dependencies
#      ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, t_wdl_mask ) # control dependencies
      t_pred_tendency_weight = tf.gather(tf.stack([tf.constant(point_scheme[3][::-1], dtype=t_weight.dtype),
                                        tf.constant(point_scheme[3], dtype=t_weight.dtype)], axis=0),
                                    tf.cast(t_is_home_bool, tf.int32))

      def weighted_softmax_cross_entropy(labels_weighted, p):
        return tf.reduce_sum(-tf.log(p+epsilon) * labels_weighted, axis=1)
      
      def softpoint_loss(p):
        t_wdl_pred = tf.matmul(p, t_wdl_mask)   
        t_leg2_pred = tf.matmul(p, t_leg2_mask)   
        l_softmax1 = t_weight * weighted_softmax_cross_entropy(t_wdl_labels*t_pred_tendency_weight, t_wdl_pred) 
        l_softmax2 = t_weight * weighted_softmax_cross_entropy(t_leg2_labels, t_leg2_pred) 
        #softpoints = t_weight * weighted_softmax_cross_entropy(-achievable_points_mask*(p+0.2), p)
        softpoints = t_weight * weighted_softmax_cross_entropy(-achievable_points_mask, p * t_draw_bias_mask)
        
        return 0.05*l_softmax1 + 0.2*l_softmax2, softpoints
      
      pt_pgpt_sm_loss, pt_pgpt_softpoints = softpoint_loss(predictions["pgpt/p_pred_12"])

      loss -= 0.05*tf.reduce_mean(pt_pgpt_softpoints)

      loss += 0.05*tf.reduce_mean(pt_pgpt_sm_loss)

      #print(tf.get_collection(tf.GraphKeys.WEIGHTS))
      reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      #reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
      #print(reg_variables)
      l_regularization = tf.zeros(shape=(), dtype=t_weight.dtype)
      if (reg_variables):
        #reg_term = tf.contrib.layers.apply_regularization(GLOBAL_REGULARIZER, reg_variables)
        for r in reg_variables:
          reg_eval_metric_ops.update(collect_summary("regularization", r.name[6:-2], mode, tensor=r))
          l_regularization += r

#        #print(reg_term)
#        tf.summary.scalar("regularization/reg_term", reg_term)
#        loss += reg_term
#      for r in reg_variables:
#        tf.summary.scalar("regularization/"+r.name[6:-2], tf.contrib.layers.apply_regularization(GLOBAL_REGULARIZER, r))
#        
      if False:
        loss += l_regularization
      
#      ### ensemble
#      t_home_points = tf.stack([predictions[p+"z_points"][0::2] for p in ens_prefix_list], axis=1)
#      t_away_points = tf.stack([predictions[p+"z_points"][1::2] for p in ens_prefix_list], axis=1)
#      t_points = tf.concat([t_home_points, t_away_points], axis=1)
#      t_ensemble_softpoints = t_weight[::2] * tf.reduce_sum(predictions["ens/p_ensemble"][0::2]*t_points, axis=1)

      #loss -= tf.reduce_mean(30*t_ensemble_softpoints)
      # 30.1.2018
#      loss -= tf.reduce_mean(10*t_ensemble_softpoints)
      
      loss = tf.identity(loss, "loss")
      #tf.summary.scalar("loss", loss)
      
      eval_metric_ops = reg_eval_metric_ops
      eval_metric_ops.update(collect_summary("losses", "l_loglike_poisson", mode, tensor=l_loglike_poisson))
      eval_metric_ops.update(collect_summary("losses", "l_softmax_1h", mode, tensor=l_softmax_1h))
      eval_metric_ops.update(collect_summary("losses", "l_softmax_2h", mode, tensor=l_softmax_2h))
 
      eval_metric_ops.update(collect_summary("losses", "loss", mode, tensor=loss))
      eval_metric_ops.update(collect_summary("summary", "loss", mode, tensor=loss))

      eval_metric_ops.update(collect_summary("losses", "pt_pgpt_softpoints", mode, tensor=pt_pgpt_softpoints))
      eval_metric_ops.update(collect_summary("losses", "pt_pgpt_sm_loss", mode, tensor=pt_pgpt_sm_loss))

#      eval_metric_ops.update(collect_summary("losses", "t_ensemble_softpoints", mode, tensor=t_ensemble_softpoints))
      eval_metric_ops.update(collect_summary("losses", "l_regularization", mode, tensor=l_regularization))
      eval_metric_ops.update(collect_summary("summary", "l_regularization", mode, tensor=l_regularization))
      eval_metric_ops.update(collect_summary("losses", "l_tendency", mode, tensor=l_tendency))
      eval_metric_ops.update(collect_summary("losses", "l_gdiff", mode, tensor=l_gdiff))
      eval_metric_ops.update(collect_summary("losses", "l_gfull", mode, tensor=l_gfull))
      
    return eval_metric_ops, loss

  def create_result_metrics(prefix, predictions, labels, labels2, t_is_home_bool, achievable_points_mask, tc, mode):      
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    pGS = predictions[prefix+"pred"][:,0]
    pGC = predictions[prefix+"pred"][:,1]
    probs = predictions[prefix+"p_pred_12"]
    ev_goals_1 = predictions[prefix+"ev_goals_1"]
    ev_goals_2 = predictions[prefix+"ev_goals_2"]
    
    dtype = predictions[prefix+"p_pred_12"].dtype
    with tf.variable_scope(prefix+"Metrics"):
      point_summary = calc_points(pGS,pGC, labels, labels2, t_is_home_bool)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary

      predictions[prefix+"z_points"] = pt_actual_points
      predictions[prefix+"is_tendency"] = tf.cast(is_tendency, dtype)
      predictions[prefix+"is_full"] = tf.cast(is_full, dtype)
      predictions[prefix+"is_diff"] = tf.cast(is_diff, dtype)
      
      pred_draw = tf.cast(tf.equal(pGS,pGC), dtype)
      pred_home_win = tf.cast((tf.greater(pGS, pGC) & t_is_home_bool) | (tf.less(pGS, pGC) & tf.logical_not(t_is_home_bool)), dtype)
      pred_away_win = tf.cast((tf.less(pGS, pGC) & t_is_home_bool) | (tf.greater(pGS, pGC) & tf.logical_not(t_is_home_bool)), dtype)

      labels_float  = tf.cast(labels, dtype)
      labels_float2 = tf.cast(labels2, dtype)
      
      l_diff_ev_goals_L1 = tf.abs(labels_float-labels_float2-( ev_goals_1-ev_goals_2))

      pt_softpoints = tf.reduce_sum(probs * achievable_points_mask, axis=1)

      prefix = prefix[:-1] # cut off trailing "/"
      eval_metric_ops = {}
      eval_metric_ops.update(collect_summary(prefix, "z_points", mode, tensor=pt_actual_points))
      eval_metric_ops.update(collect_summary(prefix, "metric_is_tendency", mode, tensor=tf.cast(is_tendency, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_is_diff", mode, tensor=tf.cast(is_diff, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_is_full", mode, tensor=tf.cast(is_full, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_home_win", mode, tensor=pred_home_win))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_away_win", mode, tensor=pred_away_win))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_draw", mode, tensor=pred_draw))
      eval_metric_ops.update(collect_summary(prefix, "pt_softpoints", mode, tensor=pt_softpoints))
      eval_metric_ops.update(collect_summary(prefix, "metric_ev_goals_diff_L1", mode, tensor=l_diff_ev_goals_L1))
      eval_metric_ops.update(collect_summary(prefix, "metric_cor_diff", mode, tensor=corrcoef(ev_goals_1-ev_goals_2, labels_float-labels_float2)))
    return eval_metric_ops

  def create_ensemble_result_metrics(predictions, labels, labels2, achievable_points_mask, tc, mode):      
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    pGS = predictions["ens/pred"][0::2, 0]
    pGC = predictions["ens/pred"][0::2, 1]
    labels = labels[0::2]
    labels2 = labels2[0::2]
    achievable_points_mask = achievable_points_mask[0::2]
    
    with tf.variable_scope("ensMetrics"):
      point_summary = calc_points(pGS,pGC, labels, labels2, True)
      pt_actual_points, is_tendency, is_diff, is_full, pt_draw_points, pt_home_win_points, pt_home_loss_points, pt_away_loss_points, pt_away_win_points = point_summary

      predictions["ens/z_points"] = pt_actual_points
      predictions["ens/is_tendency"] = tf.cast(is_tendency, tf.float32)

      pred_draw = tf.cast(tf.equal(pGS,pGC), tf.float32)
      pred_home_win = tf.cast(tf.greater(pGS, pGC)  , tf.float32)
      pred_away_win = tf.cast(tf.less(pGS, pGC) , tf.float32)
      
      eval_metric_ops = {}
      eval_metric_ops.update(collect_summary("ens", "z_points", mode, tensor=pt_actual_points))
      eval_metric_ops.update(collect_summary("ens", "metric_is_tendency", mode, tensor=tf.cast(is_tendency, tf.float32)))
      eval_metric_ops.update(collect_summary("ens", "metric_is_diff", mode, tensor=tf.cast(is_diff, tf.float32)))
      eval_metric_ops.update(collect_summary("ens", "metric_is_full", mode, tensor=tf.cast(is_full, tf.float32)))
      eval_metric_ops.update(collect_summary("ens", "metric_pred_home_win", mode, tensor=pred_home_win))
      eval_metric_ops.update(collect_summary("ens", "metric_pred_away_win", mode, tensor=pred_away_win))
      eval_metric_ops.update(collect_summary("ens", "metric_pred_draw", mode, tensor=pred_draw))

      prefix_list = [p[:-1] for p in ens_prefix_list]
      prefix_list = [p+"_home" for p in prefix_list]+[p+"_away" for p in prefix_list]
      
#      prefix_list = ["p1_home", "p1_away", "p2_home", "p2_away", 
#                     "p3_home", "p3_away", "p4_home", "p4_away", 
#                     "p5_home", "p5_away", "p6_home", "p6_away", 
#                     "sp_home", "sp_away", 
#                     "sm_home", "sm_away", "smhb_home", "smhb_away"] 

      for i,p in enumerate(prefix_list):
        eval_metric_ops.update(collect_summary("ens", p, mode, tensor=tf.cast(tf.equal(predictions["ens/selected_strategy"],i), tf.float32)))

    return eval_metric_ops

  def decode_home_away_matches(x):
    if x is None: 
      return x
    r = len(x.shape)# tf.rank(x)
    s = x.shape # tf.shape(x)
#    print(s)
#    print(type(s))
#    for i in s:
#      print(i)
#      print(type(i))
    if type(s)==tf.TensorShape:
      s = [int(i.value) if (i.value is not None) and (i is not tf.Dimension(None)) else -1 for i in s]
    else:
      s = [int(i) if (i is not None) and (i is not tf.Dimension(None)) else -1 for i in s]
#    print(s)  
#    print(type(s))
    p0 = np.arange(r)
    p = list(p0[1:-1])+[0]+[p0[-1]] # rotate first dimension in second last position
#    print(p)
    x_new = tf.transpose(x, perm=p)
#    print(x_new)
    s_new =  s[1:-1] + [-1] # keep all dimensions except first and last
    x_new = tf.reshape(x_new, s_new)
#    print(x_new)
    p = [p0[-2]]+list(p0[0:-2]) # rotate last dimension in second first position
#    print(p)
    x_new = tf.transpose(x_new, perm=p)
#    print(x_new)
    return x_new

  def encode_home_away_match_predictions(s, wrapper_fun=None):
    s = {k:tf.stack([v[0::2], v[1::2]], axis=-1) for k,v in s.items()}
    if wrapper_fun is not None:
      s = {k:wrapper_fun(v) for k,v in s.items()}
    return s

  def model(features, labels, mode, params):
#    print("labels", labels)
    tc = constant_tensors()
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    
#    print("features", features)
    fc = [tf.feature_column.numeric_column(key=k, shape=v.shape[1:].as_list(), dtype=v.dtype) for k, v in features.items()]
    #fc = {k:tf.feature_column.numeric_column(key=k, shape=fc.shape, dtype=fc.dtype) for k,fc in zip(features.keys(), params['feature_columns'])}
#    print(fc)
    #features = tf.feature_column.input_layer(features, fc)
    #features = {k:tf.reshape(tf.feature_column.input_layer(features, fc), [-1]+[x for x in fc.shape]) for k,fc in zip(features.keys(), params['feature_columns']) if k != 'match_input_layer'}
    features = {k:tf.feature_column.input_layer(features, f) for k,f in zip(features.keys(), fc) if k != 'match_input_layer'}
#    print(features)
    
    #alldata_placeholder = tf.placeholder(params["data"].dtype, shape=params["data"].shape, name="alldata")
    #alllabels_placeholder = tf.placeholder(params["labels"].dtype, shape=params["labels"].shape, name="alllabels")
#    alllabels_placeholder = tf.get_default_graph().get_tensor_by_name("alllabels:0")
#    alldata_placeholder = tf.get_default_graph().get_tensor_by_name("alldata:0")

    alldata_placeholder = tf.placeholder(thedata.dtype, shape=thedata.shape, name="alldata")
    alllabels_placeholder = tf.placeholder(thelabels.dtype, shape=thelabels.shape, name="alllabels")

#    print(alldata_placeholder)
#    print(alllabels_placeholder)
    selected_batch = tf.cast(features['gameindex'], tf.int32)
#    paired_batch = tf.floordiv(selected_batch, 2)+1-tf.floormod(selected_batch, 2)
#    selected_batch = tf.concat([selected_batch, paired_batch], axis=0) # add home to away and vice versa
    
#    print_op = tf.print("selected batch: ", selected_batch[0:5], output_stream=sys.stdout)
#    with tf.control_dependencies([print_op]):
    features["newgame"] = tf.squeeze(tf.gather(alldata_placeholder, selected_batch), axis=1)
    features["newgame"] = tf.cast(features["newgame"], tf.float32)
    labels = tf.squeeze(tf.gather(alllabels_placeholder, selected_batch), axis=1)
    labels = tf.cast(labels, tf.float32)
    #labels = tf.cast(labels, tf.float32)
    alldata0 = tf.concat([alldata_placeholder[0:1]*0.0, alldata_placeholder], axis=0)
    alllabels0 = tf.concat([alllabels_placeholder[0:1]*0.0, alllabels_placeholder], axis=0)
    def build_history_input(name):
      hist_idx = tf.cast(features[name], tf.int32)
#      print_op = tf.print(name, hist_idx[0:5], output_stream=sys.stdout)
#      with tf.control_dependencies([print_op]):
      hist_idx = hist_idx+1
      data = tf.gather(params=alldata0, indices=hist_idx)
      labels = tf.gather(params=alllabels0, indices=hist_idx)
      features[name] = tf.concat([data, labels], axis=2)
      features[name] = tf.cast(features[name], tf.float32)

    build_history_input("match_history_t1")
    build_history_input("match_history_t2")
    build_history_input("match_history_t12")
    
    with tf.variable_scope("Model"):

      #features = {k:decode_home_away_matches(f) for k,f in features.items() }
      #labels   = decode_home_away_matches(labels)

      graph_outputs = buildGraph(features, labels, mode, params)
      outputs, sp_logits, hidden_layer, eval_metric_ops, cond_probs, f_date_round, decode_t1, decode_t2, decode_t12  = graph_outputs
      t_is_home_bool = tf.equal(features["newgame"][:,1] , 1)
#      t_is_train_bool = tf.equal(features["Train"] , True)

      def apply_prefix(predictions, prefix):
        return {prefix+k:v for k,v in predictions.items() }

      
      with tf.variable_scope("sp"):
          predictions = create_hierarchical_predictions(outputs, sp_logits, t_is_home_bool, tc, mode, apply_point_scheme=False)
      #predictions = create_predictions(outputs, sp_logits, t_is_home_bool, tc)
      predictions = apply_prefix(predictions, "sp/")

      h1_logits, h2_logits, p_pred_h1, label_features_h1, p_pred_h2, label_features_h2, t_mask, test_p_pred_12_h2, p_pred_12_h2 = cond_probs
      
      predictions["test_p_pred_12_h2"]=test_p_pred_12_h2
      predictions["p_pred_12_h2"]=p_pred_12_h2
      
      if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        predictions["cp1/logits"] = h1_logits
        predictions["cp/logits"] = h2_logits
        predictions["cp1/outputs"] = p_pred_h1
        predictions["cp/outputs"] = p_pred_h2
        predictions["cp1/labels"] = label_features_h1
        predictions["cp/labels"] = label_features_h2
            
      with tf.variable_scope("cp"):
          cp_predictions = create_hierarchical_predictions(outputs, h2_logits, t_is_home_bool, tc, mode)
      predictions.update(apply_prefix(cp_predictions, "cp/"))
      cp1_predictions = create_predictions(outputs, h1_logits, t_is_home_bool, tc, False)
      predictions.update(apply_prefix(cp1_predictions, "cp1/"))

      avg_logits = sp_logits + h2_logits # averaging of sp and cp strategies in logit space
      with tf.variable_scope("av"):
#        avg_predictions = create_predictions(outputs, avg_logits, t_is_home_bool, tc, False)
#        avg_pred = create_fixed_scheme_prediction_new(avg_predictions["p_pred_12"], t_is_home_bool, mode)
#        avg_predictions.update({"pred":avg_pred})
#        predictions.update(apply_prefix(avg_predictions, "av/"))
        avg_predictions = create_hierarchical_predictions(outputs, avg_logits, t_is_home_bool, tc, mode)
        predictions.update(apply_prefix(avg_predictions, "av/"))

      T1_GFT = tf.exp(outputs[:,0])
      T2_GFT = tf.exp(outputs[:,1])
      T1_GHT = tf.exp(outputs[:,2])
      T2_GHT = tf.exp(outputs[:,3])
      T1_GH2 = tf.exp(outputs[:,16])
      T2_GH2 = tf.exp(outputs[:,17])
      T1_GFT_est = (T1_GFT+T1_GHT+T1_GH2)/2
      T2_GFT_est = (T2_GFT+T2_GHT+T2_GH2)/2
      epsilon = 1e-7
      predictions_poisson_FT = create_predictions_from_ev_goals(T1_GFT_est, T2_GFT_est, t_is_home_bool, tc)
      
      p1pt_predictions = {k:v for k,v in predictions_poisson_FT.items() } # copy
      p1pt_predictions.update(point_maximization_layer(outputs, tf.stack([tf.log(T1_GFT_est+epsilon), tf.log(T2_GFT_est+epsilon)], axis=1), "pgpt", t_is_home_bool, tc, mode))
      p1pt_predictions = apply_prefix(p1pt_predictions, "pgpt/" )
      predictions.update(apply_prefix(predictions_poisson_FT, "pg/"))
      predictions.update(p1pt_predictions)

      predictions["outputs_poisson"] = outputs
      #predictions["index"] = index

      for k,v in segmentation_strategies.items():
        with tf.variable_scope(k):
          segm_pred = create_fixed_scheme_prediction_new(predictions[k+"/p_pred_12"], t_is_home_bool, mode)
          segm_predictions = {k2:v2 for k2,v2 in cp_predictions.items() } # copy
          segm_predictions.update({"pred":segm_pred})
          predictions.update(apply_prefix(segm_predictions, v+"/"))
      

#      with tf.variable_scope("Ensemble"):
#        all_predictions = [predictions[p+"pred"] for p in ens_prefix_list]
#        all_predictions = tf.stack(all_predictions, axis=1)
#        goal_diffs = tf.expand_dims(all_predictions[:,:,0] - all_predictions[:,:,1], axis=2)
#        tendencies = tf.expand_dims(tf.sign(all_predictions[:,:,0] - all_predictions[:,:,1]), axis=2)
#        all_predictions = tf.concat([all_predictions, goal_diffs, tendencies], axis=2)
#        all_predictions = tf.reshape(all_predictions, (-1, int(all_predictions.shape[1])*int(all_predictions.shape[2]) ))
#        all_predictions = tf.cast(all_predictions, tf.float32)  
#        X = tf.concat([all_predictions[0::2], all_predictions[1::2], hidden_layer[0::2], hidden_layer[1::2], f_date_round[0::2] ], axis=1)
#        print(X)
#        X = tf.stop_gradient(X)
#        ensemble_logits,_ = build_dense_layer(X, len(ens_prefix_list)*2, mode, regularizer = l2_regularizer(scale=0.01), keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
# 
#      predictions_ensemble = create_ensemble_predictions(ensemble_logits, predictions, t_is_home_bool, tc)
#      predictions_ensemble = apply_prefix(predictions_ensemble, "ens/")
#      predictions.update(predictions_ensemble)
      
      #export_outputs = { "predictions": tf.estimator.export.ClassificationOutput(predictions["sp/p_pred_12"])}
      export_outputs = { p[:-1]: tf.estimator.export.PredictOutput(predictions[p+"p_pred_12"]) for p in prefix_list}
      export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]=tf.estimator.export.PredictOutput(predictions["sp/p_pred_12"])
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode, 
          predictions    = predictions, 
          export_outputs = export_outputs)
          #wrapper_fun=tf.estimator.export.ClassificationOutput)
#      return tf.estimator.EstimatorSpec(
#          mode=mode, 
#          predictions    = encode_home_away_match_predictions(predictions), 
#          export_outputs = encode_home_away_match_predictions(export_outputs, wrapper_fun=tf.estimator.export.ClassificationOutput))

    with tf.variable_scope("Evaluation"):

      t_goals_1  = tf.cast(labels[:, 0], dtype=tf.int32)
      t_goals_2  = tf.cast(labels[:, 1], dtype=tf.int32)
      t_goals = tf.stack([t_goals_1,t_goals_2], axis=1)
      
      t_is_home_loss_bool = (t_is_home_bool & tf.less(t_goals_1, t_goals_2)) | (tf.logical_not(t_is_home_bool) & tf.greater(t_goals_1, t_goals_2))
      t_is_home_win_bool = (t_is_home_bool & tf.greater(t_goals_1, t_goals_2)) | (tf.logical_not(t_is_home_bool) & tf.less(t_goals_1, t_goals_2))
#      t_is_draw_bool = tf.equal(t_goals_1, t_goals_2)
      
      eval_metric_ops.update(create_poisson_correlation_metrics(outputs, labels, mode))
      
      # prepare derived data from labels
      gs = tf.minimum(t_goals_1,6)
      gc = tf.minimum(t_goals_2,6)
      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    

      def append_result_metrics(result_metrics, prefix):
        result_metrics_new = create_result_metrics(prefix, predictions, t_goals[:,0], t_goals[:,1], t_is_home_bool, achievable_points_mask, tc, mode)
        #result_metrics_new = {prefix+"/"+k:v for k,v in result_metrics_new.items() }
        result_metrics.update(result_metrics_new)

      result_metrics = {}
      for p in prefix_list:
        append_result_metrics(result_metrics, p)
#      ensemble_metrics = create_ensemble_result_metrics(predictions, t_goals[:,0], t_goals[:,1], achievable_points_mask, tc, mode)
#      #result_metrics.update({"ens/"+k:v for k,v in ensemble_metrics.items() })
#      result_metrics.update(ensemble_metrics)
      
    eval_metric_ops.update(result_metrics)

    eval_loss_ops, loss = create_losses_RNN(outputs, sp_logits, cond_probs, labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool, eval_metric_ops)
    eval_metric_ops.update(eval_loss_ops)
    eval_ae_loss_ops, loss = create_autoencoder_losses(loss, decode_t1, decode_t2, decode_t12, features, labels, mode)  
    eval_metric_ops.update(eval_ae_loss_ops)
    eval_metric_ops.update({"summary/"+k:v for k,v in eval_metric_ops.items() if "z_points" in k })
    eval_metric_ops.update({"summary/"+k:v for k,v in eval_metric_ops.items() if "is_tendency" in k })

    if mode == tf.estimator.ModeKeys.EVAL:
#      for key, value in eval_metric_ops.items():
#        tf.summary.scalar(key, value[1])
      #summary_op=tf.summary.merge_all()
      return tf.estimator.EstimatorSpec(
          mode=mode, # predictions=predictions, 
          loss= loss, eval_metric_ops={k:v[0] for k,v in eval_metric_ops.items()})
  
  
    global_step = tf.train.get_global_step()
    tf.summary.scalar("global_step/global_step", global_step)
    #optimizer = tf.train.GradientDescentOptimizer(1e-4)
    learning_rate = 3e-3 # 1e-3 -> 1e-2 on 4.1.2018 and back 1e-4, 3e-4
    #learning_rate = 2e-3 
    #learning_rate = 2e-4 
    #learning_rate = 1e-3
    print("Learning rate = {}".format(learning_rate))

#    decay_steps=200
#    learning_rate = tf.train.cosine_decay(learning_rate,
#                          global_step=tf.mod(global_step-1, decay_steps),
#                          decay_steps=decay_steps,
#                          alpha=0.03)
    tf.summary.scalar('global_step/learning_rate', learning_rate)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
#    optimizer_class = extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
#    optimizer = optimizer_class(weight_decay=0.01, learning_rate=learning_rate)
    
    #print("WEIGHTS: ", tf.get_collection(tf.GraphKeys.WEIGHTS))
    #print("REGULARIZATION_LOSSES", tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    gradients, variables = zip(*optimizer.compute_gradients(loss))
#    print(gradients)
#    print("gradient variables: ", variables)

    # handle regularization weight-decay apart from other weights
    # AdamOptimizer shall not include regularization in its momentum
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.add_n(reg_variables)
    #print("reg_loss", reg_loss)
    #tf.summary.scalar('losses/regularization_loss', reg_loss)
    
    reg_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    reg_gradients, reg_variables = zip(*reg_optimizer.compute_gradients(reg_loss))
    
#    print(reg_gradients)
#    print("reg gradient variables: ", reg_variables)

    #exclude_list = ["CNN", "RNN", "Layer0", "Layer1", "Layer2", "Poisson"]
    exclude_list = []
    def skip_gradients(gradients, variables, exclude_list): 
      gradvars = [(g,v) for g,v in zip(gradients, variables) if g is not None and not any(s in v.name for s in exclude_list)]      
      gradients, variables = zip(*gradvars)
      print(variables)
      print(gradients)
      return gradients, variables
      
    reg_gradients, reg_variables = skip_gradients(reg_gradients, reg_variables, exclude_list)
    gradients, variables = skip_gradients(gradients, variables, exclude_list)
    
    # handle model upgrades gently
    if True:
      print("Auto-Encoder: using gradient reduction")
      print("Small gradients", [(v,g) for g,v in zip(gradients, variables) if g is not None and "CNN_" in v.name] )
      variables, gradients = zip(*[(v,g if "CNN_" not in v.name else 0.01*g*0.0001) for g,v in zip(gradients, variables) if g is not None])
      reg_variables, reg_gradients = zip(*[(v,g if "CNN_" not in v.name else 0.01*g*0.00001) for g,v in zip(reg_gradients, reg_variables) if g is not None])

    print(variables)
    print(gradients)
    # set NaN gradients to zero
    gradients = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in gradients]

    #gradients, _ = tf.clip_by_global_norm(gradients, 1000.0, use_norm=global_norm)
    gradients = [tf.clip_by_norm(g, 100.0, name=g.name[:-2]) for g in gradients]
    #print(gradients)
    
    if True:
      for g,v in zip(gradients, variables):
        eval_metric_ops.update(collect_summary("Gradients", v.name[:-2], mode, tensor=tf.global_norm([g])))

    global_norm = tf.global_norm([tf.cast(g, dtype=tf.float32) for g in gradients])
    eval_metric_ops.update(collect_summary("Gradients", "global_norm", mode, tensor=global_norm))

    #print("gradient variables", variables)
    train_op = optimizer.apply_gradients(zip(gradients, variables), 
                                         global_step=global_step, name="ApplyGradients") 
                                         #,decay_var_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
    with tf.control_dependencies([train_op, tf.assign_add(global_step, -1)]):
      train_op = reg_optimizer.apply_gradients(zip(reg_gradients, reg_variables), 
                                         global_step=global_step , name="ApplyRegularizationGradients") 
#    print(train_op)

#    reset_node = tf.get_default_graph().get_tensor_by_name("Model/HomeBias/HomeBias/bias:0")
#    reset_op = tf.assign(reset_node, reset_node - 0.5)
    #reset_node = tf.get_default_graph().get_tensor_by_name("Model/home_bias:0")
    #reset_op = tf.assign(reset_node, tf.minimum(reset_node, -1.0))
#    reset_node = tf.group(tf.get_default_graph().get_tensor_by_name("Model/Softpoints/b/Assign:0"),
#                           tf.get_default_graph().get_tensor_by_name("Model/Softpoints/W/Assign:0"))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #print(update_ops)
    with tf.control_dependencies(update_ops):
      # Ensures that we execute the update_ops before performing the train_step
      if use_swa:
        train = tf.assign_add(global_step, 1) # no gradient descent, only adjust batch normalization weights from train data
      else:
        train = tf.group( train_op) #, tf.assign_add(global_step, 1))
    
    # keep only summary-level metrics for training
    eval_metric_ops = {k:v for k,v in eval_metric_ops.items() if 
                       "summary/" in k or 
                       "losses/" in k or 
                       "Gradients/" in k } 
                       # or "histogram/" in k}
    
    for key, value in eval_metric_ops.items():
      tf.summary.scalar(key, value[1])
        #tf.summary.scalar(key, value[0][1])

#    summary_op=tf.summary.merge_all()
#    summary_hook = tf.train.SummarySaverHook(save_steps=100,
#                                       output_dir=model_dir+"/train",
#                                       scaffold=None,
#                                       summary_op=summary_op)
#    
#    checkpoint_hook = tf.train.CheckpointSaverHook(model_dir, 
#                                            save_steps=save_steps, 
#                                            saver = tf.train.Saver(max_to_keep=max_to_keep))
    
#    print(eval_metric_ops["summary/ens/z_points"][0][1])
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
#                                               "ens" : eval_metric_ops["summary/ens/z_points"][0][1],
                                               "cp" : eval_metric_ops["summary/cp/z_points"][0][1],
                                               "cp2" : eval_metric_ops["summary/cp2/z_points"][0][1], 
                                               "pgpt" : eval_metric_ops["summary/pgpt/z_points"][0][1],
                                               "pg2" : eval_metric_ops["summary/pg2/z_points"][0][1],
                                               "sp" : eval_metric_ops["summary/sp/z_points"][0][1]}, 
      every_n_iter=25)
    
    return tf.estimator.EstimatorSpec(mode=mode #, predictions=predictions 
                                      , loss= loss+reg_loss, train_op=train
                                      , eval_metric_ops={k:v[0] for k,v in eval_metric_ops.items()}
                                      , training_hooks=[logging_hook]
                                      , evaluation_hooks=[logging_hook]
                                      , prediction_hooks=[logging_hook]
                                      )# , training_hooks = [summary_hook, checkpoint_hook]  )
  if use_swa:  
    model_dir = model_dir+"/swa"
    
  return tf.estimator.Estimator(model_fn=model, model_dir=model_dir,
                                params={'feature_columns': my_feature_columns,
                                        #"data":thedata, 'labels':thelabels
                                        },
                                config = tf.estimator.RunConfig(
                                    save_checkpoints_steps=save_steps,
                                    save_summary_steps=100,
                                    keep_checkpoint_max=max_to_keep,
                                    log_step_count_steps=100),
                              )

