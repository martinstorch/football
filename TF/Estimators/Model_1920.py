# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
#from pathlib import Path
#import os
#import sys
#from tensorflow.python.training.session_run_hook import SessionRunHook
#from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
#from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import check_ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops.losses import losses
from tensorflow.contrib.layers import l2_regularizer, l1_regularizer
from tensorflow.contrib.metrics import f1_score
from tensorflow.metrics import precision, recall

#from tensorflow.contrib import rnn
#from tensorflow.python import debug as tf_debug
#from tensorflow.contrib.opt.python.training.weight_decay_optimizers import extend_with_decoupled_weight_decay

GLOBAL_REGULARIZER = l2_regularizer(scale=0.1)
MINOR_REGULARIZER = l2_regularizer(scale=0.005)
plot_list = ["cp/", "cp2/", "cpmx/", "sp/", "pg2/", "av/", "avmx/", "cbsp/", "xpt/"] # ["pspt/", "cp/", "cp2/", "sp/", "ens/"]
prefix_list = ["pgpt/", "sp/", "cp/", "cp2/", "cpmx/", "pg2/", "av/", "avmx/", "cbsp/", "xpt/"]
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
  
    if point_scheme[0][0]==-1: # Goal Difference
      gsDiff = tf.abs(gs-pGS)
      gcDiff = tf.abs(gc-pGC)
      diff_points = 10-tf.minimum(gsDiff+gcDiff, 5)
      draw_points = tf.where(is_draw & is_tendency, diff_points, z, name="is_draw") 
      home_win_points  = tf.where(is_win & is_home & is_tendency, diff_points, z)
      home_loss_points = tf.where(is_loss & is_home & is_tendency, diff_points , z)
      away_win_points  = tf.where(is_win & is_away & is_tendency, diff_points , z)
      away_loss_points = tf.where(is_loss & is_away & is_tendency, diff_points, z)
    else:  
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

    features = features["match_input_layer"]
    labels = labels[features[:,1]==1] # Home only
    features = features[features[:,1]==1] # Home only
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

def laplacian_matrix():
  m = [[-1 if abs(i-i2+j-j2)==1 and abs(i-i2-j+j2)==1 else \
         2 if (i,j)==(i2,j2) and (i,j) in [(0,0), (0,6), (6,0), (6,6)] else \
         3 if (i,j)==(i2,j2) and (i in [0,6] or j in [0,6]) else \
         4 if (i,j)==(i2,j2) else \
         0 
         for i in range(7) for j in range(7)] for i2 in range(7) for j2 in range(7)]   
  t_laplacian = tf.constant(m, dtype=tf.float32)
  return t_laplacian

def create_laplacian_loss(p_pred_12, alpha=1.0):
  laplm = laplacian_matrix()
  lp = tf.matmul(p_pred_12, laplm)
  laplacian_loss = (lp ** 2) / 2
  laplacian_loss = tf.reduce_sum(laplacian_loss, axis=1)
  laplacian_loss = tf.multiply(alpha, tf.reduce_mean(laplacian_loss), name="laplacian") 
  ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, laplacian_loss)

def kernel_smoothing_matrix(sigma = 1.0):
  m = [[np.exp(-(abs(i-i2)+abs(j-j2))/sigma)
         for i in range(7) for j in range(7)] for i2 in range(7) for j2 in range(7)]   
  t_kernel_smoothing_matrix = tf.constant(m, dtype=tf.float32)
  return t_kernel_smoothing_matrix

def apply_kernel_smoothing(p_pred_12, sigma = 1.0):
  ksm = kernel_smoothing_matrix(sigma)
  p_pred_12 = tf.matmul(p_pred_12, ksm)
  p_pred_total = tf.reduce_sum(p_pred_12, axis=1, keepdims=True)
  p_pred_12 = p_pred_12 / p_pred_total 
  return p_pred_12 
  
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


def kl_divergence(p, p_hat):
        epsilon = 1e-8
        return tf.reduce_sum(p * tf.log(p+epsilon) - p * tf.log(p_hat+epsilon) + (1 - p) * tf.log(1 - p + epsilon) - (1 - p) * tf.log(1 - p_hat + epsilon), name="KL_divergence")


def create_estimator(model_dir, label_column_names, my_feature_columns, thedata, thelabels, save_steps, evaluate_after_steps, max_to_keep, teams_count, use_swa, histograms, target_distr):    
  
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
    
  def build_cond_prob_layer(X, labels, mode, regularizer1, regularizer2, keep_prob, eval_metric_ops): 
    #X = tf.stop_gradient(X)
    with tf.variable_scope("H1"):
      h1_logits,_ = build_dense_layer(X, 49, mode, regularizer = regularizer1, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
      
  
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
      create_laplacian_loss(p_pred_12_h1, alpha=0.1) # 100
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
      label_features_h1_est = tf.matmul(p_pred_12_h1, t_map)
        
  #    p_pred_win = p_pred_h1[:,0]
  #    p_pred_draw = p_pred_h1[:,1]
  #    p_pred_loss = p_pred_h1[:,2]
  #    p_pred_win2 = p_pred_h1[:,3]
  #    p_pred_loss2 = p_pred_h1[:,4]
  #    p_pred_owngoal = p_pred_h1[:,5]
  #    p_pred_oppgoal = p_pred_h1[:,6]
  #    p_pred_bothgoal = p_pred_h1[:,7]
      
    with tf.variable_scope("H2", reuse=tf.AUTO_REUSE):

      test_p_pred_12_h2 = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        # use actual half-time score as input for dense layer
        X4 = tf.concat([X, label_features_h1], axis=1)
        h2_logits,_ = build_dense_layer(X4, 49, mode, regularizer = regularizer2, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        p_pred_12_h2 = tf.nn.softmax(h2_logits)
        create_laplacian_loss(p_pred_12_h2, alpha=0.1) # 100
        p_pred_h2 = tf.matmul(p_pred_12_h2, t_map)
        
        # loss will be linked to h2_logits only
        # reestimate p_pred_12_h2 from p_pred_12_h1_est
        X2 = tf.concat([X, label_features_h1_est], axis=1)
        h2_logits_est,_ = build_dense_layer(X2, 49, mode, regularizer = regularizer2, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        p_pred_12_h2 = tf.nn.softmax(h2_logits_est)
        
      else:  
        # this should find the same dense layer with same weights as in training - because name scope is same
        test_label_oh_h1 = tf.one_hot(tf.range(49), 49, dtype=X.dtype)
        test_label_features_h1 = tf.matmul(test_label_oh_h1, t_map)
        
        X3 = tf.concat([
            tf.tile(X, [49,1]),
            tf.reshape(tf.tile(test_label_features_h1, [1,tf.shape(X)[0]]), (tf.shape(X)[0]*49, test_label_features_h1.shape[1])),
            ], axis=1)
        
        #X3 = tf.concat([tf.map_fn(lambda x: tf.concat([x, test_label_features_h1[i]], axis=0), X) for i in range(49)], axis=0)
        test_h2_logits,_ = build_dense_layer(X3, 49, mode, regularizer = regularizer2, keep_prob=keep_prob, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        test_p_pred_12_h2 = tf.nn.softmax(test_h2_logits)
        test_p_pred_12_h2 = tf.reshape(test_p_pred_12_h2, (49,-1,49), name="test_p_pred_12_h2") # axis 0: H1 scores, 1: batch, 2: H2 scores
        #test_p_pred_12_h2 = tf.print(test_p_pred_12_h2, data=[test_p_pred_12_h2[:,0,:]], message="Individual probabilities")
            
        p_pred_12_h2 = tf.expand_dims(tf.transpose(p_pred_12_h1, (1,0)), axis=2) * test_p_pred_12_h2  # prior * likelyhood
        p_pred_12_h2 = tf.reduce_sum(p_pred_12_h2 , axis=0) # posterior # axis 0: batch, 1: H2 scores
        #p_pred_12_h2 = tf.print(p_pred_12_h2, data=[p_pred_12_h2[0,:]], message="Summarised probability")
        test_p_pred_12_h2 = tf.transpose(test_p_pred_12_h2, [1,0,2])
      
        p_pred_h2 = tf.matmul(p_pred_12_h2, t_map)
        #print(p_pred_h2)  #Tensor("Model/condprob/H2/MatMul_2:0", shape=(?, 8), dtype=float32)
        h2_logits = tf.log((p_pred_12_h2 + 1e-7) )
        
      if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        label_oh_h2 = tf.one_hot(tf.cast(T1_GFT*7+T2_GFT, tf.int32), 49, dtype=X.dtype)
        label_features_h2 = tf.matmul(label_oh_h2, t_map)
      else:
        label_features_h2 = None
        
    return (h1_logits, h2_logits, p_pred_h1, label_features_h1, p_pred_h2, label_features_h2, t_mask, test_p_pred_12_h2, p_pred_12_h2)
      
  def buildGraph(features, labels, mode, params, t_is_home_bool): 
      print(mode)
      eval_metric_ops = {}
      with tf.variable_scope("Input_Layer"):
        features_newgame = features['newgame']
        
        match_history_t1 = features['match_history_t1']
        match_history_t2 = features['match_history_t2'] 
        match_history_t12 = features['match_history_t12']
        
#        suppress_team_names = True
#        if suppress_team_names:
#          features_newgame = tf.concat([features_newgame[:,0:4], features_newgame[:,4+2*teams_count:]], axis=1)
#          match_history_t1 = tf.concat([match_history_t1[:,:,0:2], match_history_t1[:,:,2+2*teams_count:]], axis=2)
#          match_history_t2 = tf.concat([match_history_t2[:,:,0:2], match_history_t2[:,:,2+2*teams_count:]], axis=2)
#          match_history_t12 = tf.concat([match_history_t12[:,:,0:2], match_history_t12[:,:,2+2*teams_count:]], axis=2)
          
        batch_size = tf.shape(features_newgame)[0]
        num_label_columns = len(label_column_names)
        print(label_column_names)
        output_size = num_label_columns

        match_history_t1_seqlen = 10*features_newgame[:,0]
        match_history_t2_seqlen = 10*features_newgame[:,2]
        match_history_t12_seqlen = 10*features_newgame[:,3]
        
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
        if False and mode == tf.estimator.ModeKeys.TRAIN:
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

#        
#        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=8, 
#                                          activation=None, #tanhStochastic,
#                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
#        #rnn_cell = tf.nn.rnn_cell.ResidualWrapper(rnn_cell)
#        if mode == tf.estimator.ModeKeys.TRAIN:
#          # 13.12.2017: was 0.9
#          rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
#               input_keep_prob=0.99, 
#               output_keep_prob=0.99,
#               state_keep_prob=0.99)
#        return rnn_cell
      

#      def make_gru_cell():
#        return tf.nn.rnn_cell.MultiRNNCell([
#            make_rnn_cell(), 
##            tf.nn.rnn_cell.ResidualWrapper(make_rnn_cell()),
##            tf.nn.rnn_cell.ResidualWrapper(make_rnn_cell())
#          ])
      
#      def make_rnn(match_history, sequence_length, rnn_cell = make_gru_cell()):
#        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
#        outputs, state = tf.nn.dynamic_rnn(rnn_cell, match_history,
#                                   initial_state=initial_state,
#                                   dtype=tf.float32,
#                                   sequence_length = sequence_length)
#        # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
#        # 'state' is a tensor of shape [batch_size, num_units]
#        eval_metric_ops.update(variable_summaries(outputs, "Intermediate_Outputs", mode))
##        eval_metric_ops.update(variable_summaries(state[1], "States", mode))
#        eval_metric_ops.update(variable_summaries(sequence_length, "Sequence_Length", mode))
#        return state[-1] # use upper layer state
#
      def make_rnn_cell():
        return tf.keras.layers.SimpleRNNCell(units=8, 
                                          activation=None,
                                          recurrent_regularizer = l2_regularizer(scale=0.1))

      def make_rnn(match_history, sequence_length, rnn_cell):
          def extract_BW_xG_WDL(y):
              return tf.concat([y[:,:,4:7], y[:,:,-2:], y[:,:,-30:-27]], axis=2)
          match_history = extract_BW_xG_WDL(match_history)
          print(match_history)

          if mode == tf.estimator.ModeKeys.TRAIN:
            match_history = tf.cond(tf.random.uniform((1,))[0]>0.2, 
                        lambda: match_history, 
                        lambda: tf.transpose(tf.random.shuffle(tf.transpose(match_history, (1,0,2))), (1,0,2))
                        )

          rnn_output = tf.keras.layers.RNN(cell = rnn_cell,
                                      return_sequences=False,
                                          return_state=False,
                                      unroll=True)(match_history, mask=None, 
                                                 training=(mode==tf.estimator.ModeKeys.TRAIN))

          # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
          # 'state' is a tensor of shape [batch_size, num_units]
          #eval_metric_ops.update(variable_summaries(outputs, "Intermediate_Outputs", mode))
          eval_metric_ops.update(variable_summaries(rnn_output, "rnn_output", mode))
          eval_metric_ops.update(variable_summaries(sequence_length, "Sequence_Length", mode))
          return rnn_output 

      
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
      with tf.variable_scope("RNN_1"):
        shared_rnn_cell = make_rnn_cell()
        history_state_t1 = make_rnn(match_history_t1, sequence_length = match_history_t1_seqlen, rnn_cell=shared_rnn_cell)  
#        rnn_histograms()
      with tf.variable_scope("RNN_2"):
        history_state_t2 = make_rnn(match_history_t2, sequence_length = match_history_t2_seqlen, rnn_cell=shared_rnn_cell)  
      with tf.variable_scope("RNN_12"):
        history_state_t12 = make_rnn(match_history_t12, sequence_length = match_history_t12_seqlen, rnn_cell=make_rnn_cell())  
#        rnn_histograms()
      
      def reshuffle(x, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
          x = tf.cond(tf.random.uniform((1,))[0]>0.2, 
                      lambda: x, 
                      lambda: tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0]))))
        return x
      
#      history_state_t1 = reshuffle(history_state_t1 , mode)
#      history_state_t2 = reshuffle(history_state_t2 , mode)
#      history_state_t12 = reshuffle(history_state_t12 , mode)
     
         #X = features_newgame
#        with tf.variable_scope("MH1"):
#          mh1,_ = build_dense_layer(match_history_t1[:,-1], 8, mode, 
#                                        regularizer = l2_regularizer(scale=3.0), # 2.0
#                                        keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
#        with tf.variable_scope("MH2"):
#          mh2,_ = build_dense_layer(match_history_t2[:,-1], 8, mode, 
#                                        regularizer = l2_regularizer(scale=3.0), # 2.0
#                                        keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
#        with tf.variable_scope("MH12"):
#          mh12,_ = build_dense_layer(match_history_t12[:,-1], 8, mode, 
#                                        regularizer = l2_regularizer(scale=3.0), # 2.0
#                                        keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
#
#        #X = features_newgame
#        X = tf.concat([mh1, mh2, mh12], axis=1)
#        if mode == tf.estimator.ModeKeys.TRAIN:
#          X = tf.nn.dropout(X, keep_prob=0.99)
#        X = tf.concat([features_newgame, X], axis=1)
      def extract_BW_xG_WDL(y, step=-1):
        return tf.concat([y[:,step,4:7], y[:,step,-2:], y[:,step,-30:-27]], axis=1)
      
      X = features_newgame
#      X = tf.concat([X, extract_BW_xG_WDL(match_history_t1), extract_BW_xG_WDL(match_history_t2), extract_BW_xG_WDL(match_history_t12)], axis=1)
#      X = tf.concat([X, extract_BW_xG_WDL(match_history_t1, -2), extract_BW_xG_WDL(match_history_t2, -2)], axis=1)
#      X = tf.concat([X, extract_BW_xG_WDL(match_history_t1, -3), extract_BW_xG_WDL(match_history_t2, -3)], axis=1)
#      X = tf.concat([X, extract_BW_xG_WDL(match_history_t1, -4), extract_BW_xG_WDL(match_history_t2, -4)], axis=1)
#      X = tf.concat([X, extract_BW_xG_WDL(match_history_t1, -5), extract_BW_xG_WDL(match_history_t2, -5)], axis=1)
      BWdata = X[:,4:7]
      X= tf.concat([X[:,0:4], X[:,7:]], axis=1)
      if mode == tf.estimator.ModeKeys.TRAIN:
          #X = tf.nn.dropout(X, keep_prob=0.5)
          BWdata  = tf.cond(tf.random.uniform((1,))[0]>0.2, lambda: BWdata, lambda: tf.random.shuffle(BWdata))
      
      X = tf.concat([X, BWdata], axis=1)

      X = tf.concat([X, history_state_t1, history_state_t2, history_state_t12], axis=1)
      
#      if mode == tf.estimator.ModeKeys.TRAIN:
#          X= tf.cond(tf.random.uniform((1,))[0]>0.05, lambda: X, lambda: tf.random.shuffle(X))
      
        
#      with tf.variable_scope("Layer0H"):
#          X0H,Z0H = build_dense_layer(X, 128, mode, 
#                                    regularizer = l2_regularizer(scale=300.0), # 100.0
#                                    keep_prob=0.96, 
#                                    batch_norm=True, 
#                                    activation=None, 
#                                    eval_metric_ops=eval_metric_ops)
#      
#      with tf.variable_scope("Layer0A"):
#          X0A,Z0A = build_dense_layer(X, 128, mode, 
#                                    regularizer = l2_regularizer(scale=300.0), # 100.0
#                                    keep_prob=0.96, 
#                                    batch_norm=True, 
#                                    activation=None, 
#                                    eval_metric_ops=eval_metric_ops)

      with tf.variable_scope("Layer0H"):
          X0H,Z0H = build_dense_layer(X, 64, mode, # 32
                                    regularizer = l1_regularizer(scale=0.5), # 0.7 -> 0.4 
                                    keep_prob=1.0, 
                                    batch_norm=True, # True
                                    activation=None, 
                                    eval_metric_ops=eval_metric_ops)
      
      with tf.variable_scope("Layer0A"):
          X0A,Z0A = build_dense_layer(X, 64, mode, 
                                    regularizer = l1_regularizer(scale=0.5), # 100.0
                                    keep_prob=1.0, 
                                    batch_norm=True, # True
                                    activation=None, 
                                    eval_metric_ops=eval_metric_ops)
      
      X0 = tf.where(t_is_home_bool, X0H, X0A)    
      
      if False:
        with tf.variable_scope("Layer1"):
          X1,Z1 = build_dense_layer(X, 32, mode, 
                                    regularizer = l1_regularizer(scale=0.2), 
                                    keep_prob=1.0, #0.95, 
                                    batch_norm=False, # True
                                    activation=tanhStochastic, 
                                    eval_metric_ops=eval_metric_ops)
        
        with tf.variable_scope("Layer2"):
          X2,Z2 = build_dense_layer(X1, 32, mode, 
                                    #add_term = X0*2.0, 
                                    regularizer = l1_regularizer(scale=0.1), 
                                    keep_prob=1.0, #0.95, 
                                    batch_norm=False, # True
                                    activation=tanhStochastic, 
                                    eval_metric_ops=eval_metric_ops, 
                                    batch_scale=False)

        #X = 0.55*X2 + X0 # shortcut connection bypassing two non-linear activation functions
        X = X0 
      else:
        X = X0 # shortcut connection bypassing two non-linear activation functions
      
#      with tf.variable_scope("Layer3"):
#        #X = tf.stop_gradient(X)
#        X,Z = build_dense_layer(X+0.001, 64, mode, regularizer = l2_regularizer(scale=1.0), keep_prob=0.90, batch_norm=True, activation=binaryStochastic, eval_metric_ops=eval_metric_ops)
      #X = tf.layers.batch_normalization(X, momentum=0.99, center=True, scale=True, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
#        eval_metric_ops.update(variable_summaries(X, "Normalized", mode))
      hidden_layer = X
      
#      with tf.variable_scope("Skymax"):
#        sk_logits,_ = build_dense_layer(X, 49, mode, regularizer = l2_regularizer(scale=1.2), keep_prob=0.8, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)

      with tf.variable_scope("condprob"):
        cond_probs = build_cond_prob_layer(X, labels, mode, 
                                           regularizer1 = l2_regularizer(scale=0.4), 
                                           regularizer2 = l2_regularizer(scale=0.2), 
                                           keep_prob=1.0, eval_metric_ops=eval_metric_ops) 
        #cb1_logits,_ = build_dense_layer(X, 49, mode, regularizer = l2_regularizer(scale=1.2), keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)

      with tf.variable_scope("Softpoints"):
        with tf.variable_scope("WDL"):
          sp_logits_1,_ = build_dense_layer(X, 3, mode, 
                                        regularizer = l2_regularizer(scale=0.6), # 2.0
                                        keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        with tf.variable_scope("GD"):
          sp_logits_2,_ = build_dense_layer(X, 11, mode, 
                                        regularizer = l2_regularizer(scale=0.200002), # 2.0
                                        keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
  
        with tf.variable_scope("FS"):
          sp_logits_3,_ = build_dense_layer(X, 49, mode, 
                                        #regularizer = None, 
                                        regularizer = l2_regularizer(scale=1.200002), # 2.0
                                        keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        sp_logits = (sp_logits_1, sp_logits_2, sp_logits_3)

      with tf.variable_scope("Poisson"):
        outputs,Z = build_dense_layer(X, output_size, mode, 
                                regularizer = l2_regularizer(scale=2.2002), #2.0
                                keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        #outputs, index = harmonize_outputs(outputs, label_column_names)
        #eval_metric_ops.update(variable_summaries(outputs, "Outputs_harmonized", mode))
      
      with tf.variable_scope("cbsp"):
        cb_h2_logits = cond_probs[8]
        print(cb_h2_logits)
        cb_h2_logits = tf.stop_gradient(cb_h2_logits)
        cb_h2_logits = tf.concat([X, cb_h2_logits], axis=1)
        cb_h2_logits = tf.stop_gradient(cb_h2_logits)
        cbsp_logits,_ = build_dense_layer(cb_h2_logits, 49, mode, 
                                      regularizer = l2_regularizer(scale=0.20002), # 2.0
                                      keep_prob=1.0, batch_norm=False, activation=None, eval_metric_ops=eval_metric_ops, use_bias=True)
        
      
      return outputs, sp_logits, hidden_layer, eval_metric_ops, cond_probs, cbsp_logits #, decode_t1, decode_t2, decode_t12 
        
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
      
  def calc_probabilities(p_pred_12, predictions):
        t_win_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)], name="t_win_mask")
        t_draw_mask = tf.stack([1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)], name="t_draw_mask")
        t_loss_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)], name="t_loss_mask")
        t_win2_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7)+1 else 0.0  for i in range(49)], name="t_win2_mask")
        t_loss2_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7)-1 else 0.0  for i in range(49)], name="t_loss2_mask")
        t_win3_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7)+2 else 0.0  for i in range(49)], name="t_win3_mask")
        t_loss3_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7)-2 else 0.0  for i in range(49)], name="t_loss3_mask")
        t_zero_mask = tf.stack([1.0 if (i // 7) == 0 or np.mod(i, 7) == 0 else 0.0  for i in range(49)], name="t_zero_mask")

        t_all_mask = tf.stack([t_win_mask, t_draw_mask, t_loss_mask, t_win2_mask, t_loss2_mask, t_win3_mask, t_loss3_mask, t_zero_mask], axis=1, name="t_all_mask")
        t_all_mask = tf.cast(t_all_mask, p_pred_12.dtype)
        
        p_pred_all = tf.matmul(p_pred_12, t_all_mask)
        p_pred_win = p_pred_all[:,0]
        p_pred_draw = p_pred_all[:,1]
        p_pred_loss = p_pred_all[:,2]
        p_pred_win2 = p_pred_all[:,3]
        p_pred_loss2 = p_pred_all[:,4]
        p_pred_win3 = p_pred_all[:,5]
        p_pred_loss3 = p_pred_all[:,6]
        p_pred_zero = p_pred_all[:,7]
        predictions.update({
          "p_pred_win":p_pred_win, 
          "p_pred_draw":p_pred_draw, 
          "p_pred_loss":p_pred_loss, 
          "p_pred_win2":p_pred_win2, 
          "p_pred_loss2":p_pred_loss2, 
          "p_pred_win3":p_pred_win3, 
          "p_pred_loss3":p_pred_loss3, 
          "p_pred_zero":p_pred_zero, 
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
        predictions = calc_probabilities(p_pred_12, predictions)
        ev_points =  predictions["ev_points"]
        a = tf.argmax(ev_points, axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        
        predictions.update({"pred":pred})
        return predictions

  def get_gdiff(p_pred_12) :
        t_gdiff_mask = []
        for j in range(13):
          gdiff = j-6
          t1 = tf.stack([1.0 if (i // 7 - np.mod(i, 7))==gdiff else 0.0 for i in range(49) ], name="t_gdiff_mask_"+str(gdiff))
          t_gdiff_mask.append(t1)
        t_gdiff_mask = tf.stack(t_gdiff_mask, axis=1)
        t_gdiff_mask = tf.cast(t_gdiff_mask, p_pred_12.dtype)
        p_pred_gdiff = tf.matmul(p_pred_12, t_gdiff_mask)
        return p_pred_gdiff
        
  def create_hierarchical_predictions(outputs, logits, t_is_home_bool, tc, mode, prefix, p_pred_12 = None, apply_point_scheme=True):
    with tf.variable_scope("Prediction_softpoints"):
        tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc

        if p_pred_12 is None:
          p_pred_12 = tf.nn.softmax(tf.reshape(logits, [-1, 49]))
        
        predictions = apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = None)
        predictions = calc_probabilities(p_pred_12, predictions)
        
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

#        t_gdiff_mask = []
#        for j in range(13):
#          gdiff = j-6
#          t1 = tf.stack([1.0 if (i // 7 - np.mod(i, 7))==gdiff else 0.0 for i in range(49) ], name="t_gdiff_mask_"+str(gdiff))
#          t_gdiff_mask.append(t1)
#        t_gdiff_mask = tf.stack(t_gdiff_mask, axis=1)
#        t_gdiff_mask = tf.cast(t_gdiff_mask, p_pred_12.dtype)
#        p_pred_gdiff = tf.matmul(p_pred_12, t_gdiff_mask)
#        
        p_pred_gdiff = get_gdiff(p_pred_12)
        p_pred_tendency = tf.stack([predictions["p_pred_win"], predictions["p_pred_draw"], predictions["p_pred_loss"]], axis=1)
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
        
        if False:
          # try fixed scheme based on probabilities
          pred = create_quantile_scheme_prediction(outputs, p_pred_12, t_is_home_bool, mode, prefix)

        predictions.update({
          "logits":logits,
          "pred":pred,
          "p_pred_gdiff":p_pred_gdiff, 
          "pred_tendency":pred_tendency, 
          "pred_gdiff":pred_gdiff, 
          "pred_gtotal":pred_gtotal, 
        })
        return predictions
  
  def create_quantile_scheme_prediction(outputs, logits, t_is_home_bool, tc, mode, prefix, p_pred_12=None):
        if p_pred_12 is None:
          p_pred_12 = tf.nn.softmax(tf.reshape(logits, [-1, 49]))
        
        p_pred_12 = tf.stop_gradient(p_pred_12)
        
        p_pred_gdiff = get_gdiff(p_pred_12)
        
        predictions = apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = None)
        predictions = calc_probabilities(p_pred_12, predictions)

        target_distr0 = target_distr[prefix] # pick the correct distribution for the prefix
        
        with tf.variable_scope("diff_p"):
          diffp = predictions["p_pred_loss"] - predictions["p_pred_win"]
          diffp = tf.where(t_is_home_bool, diffp, -diffp)

          t_is_home_float = tf.cast(t_is_home_bool, tf.float32)
          
          p_win1 = tf.stack([predictions["p_pred_win"]*t_is_home_float, predictions["p_pred_loss"]*(1.0-t_is_home_float)], axis=1)
          p_win2 = tf.stack([predictions["p_pred_win2"]*t_is_home_float, predictions["p_pred_loss2"]*(1.0-t_is_home_float)], axis=1)
          p_win3 = tf.stack([predictions["p_pred_win3"]*t_is_home_float, predictions["p_pred_loss3"]*(1.0-t_is_home_float)], axis=1)
          p_loss1 = tf.stack([predictions["p_pred_loss"]*t_is_home_float, predictions["p_pred_win"]*(1.0-t_is_home_float)], axis=1)
          p_loss2 = tf.stack([predictions["p_pred_loss2"]*t_is_home_float, predictions["p_pred_win2"]*(1.0-t_is_home_float)], axis=1)
          p_loss3 = tf.stack([predictions["p_pred_loss3"]*t_is_home_float, predictions["p_pred_win3"]*(1.0-t_is_home_float)], axis=1)

          
          q_loss3 = target_distr0[2][2] # 0:3 -> 0:2
          q_loss2 = target_distr0[2][2]+target_distr0[2][1] # 0:2 -> 0:1
          q_loss1 = target_distr0[2][2]+target_distr0[2][1]+target_distr0[2][0] # 0:1 -> 0:0
          q_win3 = target_distr0[0][0] # 3:0 -> 2:0 
          q_win2 = target_distr0[0][0]+target_distr0[0][1] # 2:0 -> 1:0
          q_win1 = target_distr0[0][0]+target_distr0[0][1]+target_distr0[0][2] # 1:0 -> 0:0
          print("quantiles: ", prefix, [q_loss3, q_loss2, q_loss1, q_win1, q_win2, q_win3])
          cutpoint_loss3 = tf.contrib.distributions.percentile(p_loss3, 100-0.5*q_loss3, name="cutpoint_loss3", axis=0)
          cutpoint_loss2 = tf.contrib.distributions.percentile(p_loss2, 100-0.5*q_loss2, name="cutpoint_loss2", axis=0)
          cutpoint_loss1 = tf.contrib.distributions.percentile(p_loss1, 100-0.5*q_loss1, name="cutpoint_loss1", axis=0)
          cutpoint_win3 = tf.contrib.distributions.percentile(p_win3, 100-0.5*q_win3, name="cutpoint_win3", axis=0)
          cutpoint_win2 = tf.contrib.distributions.percentile(p_win2, 100-0.5*q_win2, name="cutpoint_win2", axis=0)
          cutpoint_win1 = tf.contrib.distributions.percentile(p_win1, 100-0.5*q_win1, name="cutpoint_win1", axis=0)
          
          cutpoints = [cutpoint_loss3, cutpoint_loss2, cutpoint_loss1, cutpoint_win1, cutpoint_win2, cutpoint_win3]
          #diffp = tf.layers.batch_normalization(tf.expand_dims(diffp, axis=1), axis=1, scale=True, center=True, momentum=0.99, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
          #diffp = tf.squeeze(diffp, axis=1)
          
#          t_cutpoint_loss3 = tf.get_variable("t_cutpoint_loss3", dtype=tf.float32, initializer=tf.zeros_initializer())
#          t_cutpoint_win3 = tf.get_variable("t_cutpoint_win3", dtype=tf.float32, initializer=tf.zeros_initializer())
          
          #tf.summary.histogram("diffp", diffp)

        with tf.variable_scope("p_both"):
          
          q_00 = target_distr0[3][0] # 1:1 -> 0:0 
          q_10 = target_distr0[3][1] # 2:1 -> 1:0 
          q_20 = target_distr0[3][2] # 3:1 -> 2:0 
          
          p_pred_both = 1.0-predictions["p_pred_zero"]
          cutpoint_00 = tf.contrib.distributions.percentile(p_pred_both, q_00, name="cutpoint_00")
          cutpoint_10 = tf.contrib.distributions.percentile(p_pred_both, q_10, name="cutpoint_10")
          cutpoint_20 = tf.contrib.distributions.percentile(p_pred_both, q_20, name="cutpoint_20")

          cutpoints_00 = [cutpoint_00, cutpoint_10, cutpoint_20]
          #p_pred_both = tf.layers.batch_normalization(tf.expand_dims(p_pred_both, axis=1) , axis=1, momentum=0.99, center=True, scale=True, epsilon=0.0001, training=(mode == tf.estimator.ModeKeys.TRAIN))
          #p_pred_both = tf.squeeze(p_pred_both, axis=1)
          #tf.summary.histogram("p_pred_both", p_pred_both)

        ema = tf.train.ExponentialMovingAverage(decay=0.98)
        ema_training_op = ema.apply(cutpoints+cutpoints_00)
        ops.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_training_op) # control dependencies

        for v in cutpoints_00:
          tf.summary.scalar(v.name[:-2], v)
        for v in cutpoints:
          tf.summary.scalar(v.name[:-2]+"_home", v[0])
          tf.summary.scalar(v.name[:-2]+"_away", v[1])
          #tf.summary.scalar(v.name[:-2]+"_avg", ema.average(v))
        
        pred = tf.cast(tf.stack([0*p_pred_both, 0*p_pred_both ], axis=1), dtype=tf.int32)
        p0 = tf.zeros_like(pred, dtype=tf.int32)
        
        pred = tf.where(p_pred_both<ema.average(cutpoint_00), p0+tf.constant([[0,0]]), p0+tf.constant([[1,1]])) # start with draw

        t_is_home_int = 1-tf.cast(t_is_home_bool, tf.int32) # home=0, away=1
        def make_avg(cutpoint_var):
          return tf.gather(ema.average(cutpoint_var), t_is_home_int)

        p_win1 = tf.where(t_is_home_bool, predictions["p_pred_win"], predictions["p_pred_loss"])
        p_win2 = tf.where(t_is_home_bool, predictions["p_pred_win2"], predictions["p_pred_loss2"])
        p_win3 = tf.where(t_is_home_bool, predictions["p_pred_win3"], predictions["p_pred_loss3"])
        p_loss1 = tf.where(t_is_home_bool, predictions["p_pred_loss"], predictions["p_pred_win"])
        p_loss2 = tf.where(t_is_home_bool, predictions["p_pred_loss2"], predictions["p_pred_win2"])
        p_loss3 = tf.where(t_is_home_bool, predictions["p_pred_loss3"], predictions["p_pred_win3"])

        pred = tf.where(p_win1 > make_avg(cutpoint_win1), tf.where(p_pred_both<ema.average(cutpoint_10), p0+tf.constant([[1,0]]), p0+tf.constant([[2,1]])), pred) 
        pred = tf.where(p_win2 > make_avg(cutpoint_win2), tf.where(p_pred_both<ema.average(cutpoint_20), p0+tf.constant([[2,0]]), p0+tf.constant([[3,1]])), pred)
        pred = tf.where(p_win3 > make_avg(cutpoint_win3), pred*0+tf.constant([[3,0]]), pred)

        pred = tf.where(p_loss1 > make_avg(cutpoint_loss1), tf.where(p_pred_both<ema.average(cutpoint_10), p0+tf.constant([[0,1]]), p0+tf.constant([[1,2]])), pred) 
        pred = tf.where(p_loss2 > make_avg(cutpoint_loss2), tf.where(p_pred_both<ema.average(cutpoint_20), p0+tf.constant([[0,2]]), p0+tf.constant([[1,3]])), pred) 
        pred = tf.where(p_loss3 > make_avg(cutpoint_loss3), p0+tf.constant([[0,3]]), pred) 


        pred = tf.where(t_is_home_bool, pred, pred[:,::-1])

        predictions.update({
          "pred":pred,
          "diffp":diffp, 
          #"p_pred_tendency":p_pred_tendency, 
          "p_pred_gdiff":p_pred_gdiff, 
          "p_pred_both":p_pred_both, 
        })
        return predictions

  def point_maximization_layer(outputs, X, prefix, t_is_home_bool, tc, mode, use_max_points=False):
      with tf.variable_scope("PointMax_"+prefix):
          tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
          X = tf.stop_gradient(X)
          with tf.variable_scope("Layer1"):
            X,_ = build_dense_layer(X, output_size=10, mode=mode, regularizer = None, keep_prob=0.95, batch_norm=True, activation=tf.nn.relu) #, eval_metric_ops=None, use_bias=None, add_term=None, batch_scale=True)        
            X0 = X
          with tf.variable_scope("Layer2"):
            X,_ = build_dense_layer(X, output_size=10, mode=mode, regularizer = None, keep_prob=0.95, batch_norm=True, activation=tf.nn.relu) #, eval_metric_ops=None, use_bias=None, add_term=None, batch_scale=True)        
            X = X + X0 # apply skip connections
          with tf.variable_scope("Layer3"):
            X,_ = build_dense_layer(X, output_size=49, mode=mode, regularizer = None, keep_prob=1.0, batch_norm=False, activation=None)
          
          logits=X
          #predictions = create_hierarchical_predictions(outputs, logits, t_is_home_bool, tc, mode, prefix)
          predictions = create_predictions(logits, logits, t_is_home_bool, tc, use_max_points)

          p_pred_12 = tf.nn.softmax(logits)
          create_laplacian_loss(p_pred_12, alpha=0.01)
          
          predictions = apply_poisson_summary(p_pred_12, t_is_home_bool, tc, predictions = predictions)
          predictions = calc_probabilities(p_pred_12, predictions)
          
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

    is_win  = labels[:,20]
    is_loss = labels[:,18]
    pred_win  = tf.exp(predictions["outputs_poisson"][:,20])
    pred_loss = tf.exp(predictions["outputs_poisson"][:,18])
    
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
  

  def create_hierarchical_losses_and_predictions(sp_logits, t_labels, features, t_is_home_bool, mode, tc, eval_metric_ops):
    
    sp_logits_1, sp_logits_2, sp_logits_3 = sp_logits    
    sp_logits_1 = tf.clip_by_value(sp_logits_1, -10000, 1e10)
    sp_logits_2 = tf.clip_by_value(sp_logits_2, -10000, 1e10)
    sp_logits_3 = tf.clip_by_value(sp_logits_3, -10000, 1e10)
    
    with tf.variable_scope("Prediction"):
      tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
      
      t_goals_1  = tf.cast(t_labels[:, 0], dtype=tf.int32)
      t_goals_2  = tf.cast(t_labels[:, 1], dtype=tf.int32)
        
      t_is_home_loss_bool = (t_is_home_bool & tf.less(t_goals_1, t_goals_2)) | (tf.logical_not(t_is_home_bool) & tf.greater(t_goals_1, t_goals_2))
      t_is_home_win_bool = (t_is_home_bool & tf.greater(t_goals_1, t_goals_2)) | (tf.logical_not(t_is_home_bool) & tf.less(t_goals_1, t_goals_2))
      t_is_draw = tf.logical_and(tf.logical_not(t_is_home_win_bool), tf.logical_not(t_is_home_loss_bool), name="t_is_draw")
  
      t_is_home_win_bool_f = tf.cast(t_is_home_win_bool, dtype=sp_logits_1.dtype)
      t_is_home_loss_bool_f = tf.cast(t_is_home_loss_bool, dtype=sp_logits_1.dtype)
      t_is_draw_f = tf.cast(t_is_draw, dtype=sp_logits_1.dtype)
      
      # reduce to 6 goals max. for training
      gs = tf.minimum(t_goals_1,6)
      gc = tf.minimum(t_goals_2,6)
    
    sp_labels_1 = 1+tf.sign(gs-gc)
    sp_labels_2 = tf.maximum(0, tf.minimum(10, 5+gs-gc))
    sp_labels_3 = gs*7+gc
    
    t_expand_WDL_GDiff_mask = tf.constant([[i<5 for i in range(11)], [i==5 for i in range(11)], [i>5 for i in range(11)]], dtype=tf.float32, name="t_expand_WDL_GDiff_mask")
    t_expand_GDiff_FS_mask = tf.constant([[ 1.0 if (j//7-np.mod(j,7))==(i-5) or (j==6 and i==0) or (j==42 and i==10) else 0.0 for j in range(49)] for i in range(11)], dtype=tf.float32, name="t_expand_GDiff_FS_mask")

    l_factor = tf.where(t_is_home_bool,
                                     point_scheme[3][0]*t_is_home_win_bool_f + 
                                     point_scheme[3][1]*t_is_draw_f + 
                                     point_scheme[3][2]*t_is_home_loss_bool_f,
                                     
                                     point_scheme[3][0]*t_is_home_loss_bool_f + 
                                     point_scheme[3][1]*t_is_draw_f + 
                                     point_scheme[3][2]*t_is_home_win_bool_f 
                                     )
    l_tendency = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sp_labels_1, logits=sp_logits_1, name="sp_WDL_loss")
    l_tendency *= l_factor
    pred_WDL = tf.argmax(sp_logits_1, axis=1, name="pred_WDL")

    t_actual_GF_mask = tf.matmul(tf.one_hot(sp_labels_1, 3), t_expand_WDL_GDiff_mask, name="t_actual_GF_mask")           
    t_pred_GF_mask   = tf.matmul(tf.one_hot(pred_WDL   , 3), t_expand_WDL_GDiff_mask, name="t_pred_GF_mask")           
    
    sp_logits_2_masked_act = (sp_logits_2 + 10000.0)*t_actual_GF_mask
    sp_logits_2_masked_pred = (sp_logits_2 + 10000.0)*t_pred_GF_mask
    # normalize active logits
    #sp_logits_2_masked_pred = sp_logits_2_masked_pred / tf.reduce_sum(sp_logits_2_masked_pred, axis=1, keepdims=True)
    
    l_gdiff = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sp_labels_2, logits=sp_logits_2_masked_act, name="sp_GD_loss")
    pred_GDiff = tf.argmax(sp_logits_2_masked_pred, axis=1, name="pred_GDiff")
    
    t_actual_FS_mask = tf.matmul(tf.one_hot(sp_labels_2, 11), t_expand_GDiff_FS_mask, name="t_actual_FS_mask")           
    t_pred_FS_mask   = tf.matmul(tf.one_hot(pred_GDiff , 11), t_expand_GDiff_FS_mask, name="t_pred_FS_mask")           
    
    sp_logits_3_masked_act = (sp_logits_3 + 10000.0)*t_actual_FS_mask
    sp_logits_3_masked_pred = (sp_logits_3 + 10000.0)*t_pred_FS_mask
    # normalize active logits
    #sp_logits_3_masked_pred = sp_logits_3_masked_pred / tf.reduce_sum(sp_logits_3_masked_pred, axis=1, keepdims=True)
    
    l_gfull = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sp_labels_3, logits=sp_logits_3_masked_act, name="sp_FS_loss")
    #pred_FS = tf.argmax((sp_logits_3 + 10.0)*t_pred_FS_mask, axis=1, name="pred_FS")
    
    predictions = create_predictions(sp_logits_3_masked_pred, sp_logits_3_masked_pred, t_is_home_bool, tc, use_max_points=False)

    loss = tf.reduce_mean(2*l_tendency)  
    loss += tf.reduce_mean(1.0*l_gdiff)  
    loss += tf.reduce_mean(0.5*l_gfull)  

    eval_metric_ops.update(collect_summary("losses", "l_tendency", mode, tensor=l_tendency))
    eval_metric_ops.update(collect_summary("losses", "l_gdiff", mode, tensor=l_gdiff))
    eval_metric_ops.update(collect_summary("losses", "l_gfull", mode, tensor=l_gfull))
    
    p_pred_WDL = tf.nn.softmax(sp_logits_1, axis=1)
    p_pred_GDiff = tf.nn.softmax(sp_logits_2, axis=1) * tf.matmul(p_pred_WDL, t_expand_WDL_GDiff_mask)
    p_pred_GDiff = p_pred_GDiff / tf.reduce_sum(p_pred_GDiff, axis=1, keepdims=True)
    p_pred_FS = tf.nn.softmax(sp_logits_3, axis=1) * tf.matmul(p_pred_GDiff, t_expand_GDiff_FS_mask)
    p_pred_FS = p_pred_FS / tf.reduce_sum(p_pred_FS, axis=1, keepdims=True)
    
    t_zero_mask = tf.transpose(tf.stack([[1.0 if (i // 7) == 0 or np.mod(i, 7) == 0 else 0.0  for i in range(49)]], name="t_zero_mask"))

    p_pred_zero = tf.matmul(p_pred_FS, t_zero_mask)

    apply_poisson_summary(p_pred_FS, t_is_home_bool, tc, predictions)
    
    predictions.update({
      "p_pred_win":p_pred_WDL[:,2], 
      "p_pred_draw":p_pred_WDL[:,1], 
      "p_pred_loss":p_pred_WDL[:,0], 
      "p_pred_win2":p_pred_GDiff[:,8], 
      "p_pred_loss2":p_pred_GDiff[:,4], 
      "p_pred_win3":p_pred_GDiff[:,9], 
      "p_pred_loss3":p_pred_GDiff[:,3], 
      "p_pred_zero":p_pred_zero, 
    })

    # add laplacian loss to sp_logits_2, to make sure that GDiff=0 estimate fits properly in range between -1 and +1
    laplacian_matrix_GDiff = [[-1 if abs(i-j)==1 else 2 if i==j else 0 for i in range(11)] for j in range(11)]   
    t_laplacian_matrix_GDiff = tf.constant(laplacian_matrix_GDiff, dtype=tf.float32)

    lp = tf.matmul(sp_logits_2, t_laplacian_matrix_GDiff)
    laplacian_loss = (lp ** 2) / 2
    laplacian_loss = tf.reduce_sum(laplacian_loss, axis=1)
    laplacian_loss = tf.multiply(0.1, tf.reduce_mean(laplacian_loss), name="laplacian_gdiff") 
    ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, laplacian_loss)

    if False:
      # apply laplacian regularization to upper right diagonal of the final score matrix, in order to given 0:6 and 6:0 and proper value    
      laplacian_matrix_FS =  [[-1 if abs(i-i2+j-j2)==1 and abs(i-i2-j+j2)==1 and (i+j>=5) and (i2+j2>=5) else \
           2 if (i,j)==(i2,j2) and (i,j) in [(0,6), (6,0), (6,6), (5,0), (0,5), (1,4), (4,1), (2,3), (3,2)] else \
           3 if (i,j)==(i2,j2) and (i==6 or j==6) else \
           4 if (i,j)==(i2,j2) and (i+j>=5) and (i2+j2>=5) else \
           0 
           for i in range(7) for j in range(7)] for i2 in range(7) for j2 in range(7)]   
      t_laplacian_matrix_FS = tf.constant(laplacian_matrix_FS, dtype=tf.float32)
  
      lp = tf.matmul(sp_logits_3, t_laplacian_matrix_FS)
      laplacian_loss = (lp ** 2) / 2
      laplacian_loss = tf.reduce_sum(laplacian_loss, axis=1)
      laplacian_loss = tf.multiply(0.001, tf.reduce_mean(laplacian_loss), name="laplacian_fs") 
      ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, laplacian_loss)

    return predictions, loss


  def create_losses_RNN(outputs, softpoint_loss, cond_probs, t_labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool, eval_metric_ops):
    tc_1d1_goals_f, tc_home_points_i, tc_away_points_i, calc_poisson_prob, p_tendency_mask_f, p_gdiff_mask_f, p_fulltime_index_matrix = tc
    h1_logits, h2_logits, p_pred_h1, label_features_h1, p_pred_h2, label_features_h2, t_mask, test_p_pred_12_h2, p_pred_12_h2 = cond_probs
    
    loss = softpoint_loss
    
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
      
#      match_date = features["newgame"][:,4]
#      sequence_length = features['newgame'][:,0]
#      #t_weight = tf.exp(0.5*match_date) * (sequence_length + 0.05) # sequence_length ranges from 0 to 1 - depending on the number of prior matches of team 1
#      t_weight = 1.0 + 0.0 * tf.exp(0.5*match_date) * (sequence_length + 0.05) # sequence_length ranges from 0 to 1 - depending on the number of prior matches of team 1
      t_weight = 1.0+0.0*features['newgame'][:,0]
      # result importance = log(1/frequency) - mean adjusted to 1.0
      # 1:1 has low importance, 3:2 much higher, 5:0 overvalued but rare
#      result_importance = tf.constant([0.610699463302517, 0.636624134693228, 0.688478206502995, 0.861174384773783, 1.09447971749187, 1.45916830182096, 1.45916830182096, 0.565089593736748, 0.490552492978211, 0.620077094945206, 0.760233563222618, 0.992062709888034, 1.31889788143772, 1.49409788361176, 0.584692048161506, 0.540777816952666, 0.664928108155794, 0.870617157643794, 1.10781260749576, 1.4022220211579, 1.53541081027154, 0.690646584367668, 0.698403607367345, 0.828902824753833, 0.983969215334149, 1.45916830182096, 1.58597374606561, 1.90009939460063, 0.888152492546633, 0.865846586541708, 1.00917224994398, 1.33703505934424, 1.45916830182096, 1.65116070787927, 0.226593757678754, 1.12197933011558, 1.11478527326307, 1.30210547755345, 1.45916830182096, 1.74303657033312, 0.226593757678754, 0.226593757678754, 1.30210547755345, 1.4022220211579, 1.4289109217981, 1.65116070787927, 0.226593757678754, 0.226593757678754, 0.226593757678754], dtype=t_weight.dtype)
#      row_weight = tf.gather(result_importance, gs*7+gc, name="select_importance")
#      t_weight = t_weight * row_weight

      # result importance = corrcoef 
      # 1:1 has low importance as it is difficult to predict, 3:0 much higher
      if False:
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
        
      
#      t_weight_total = tf.reduce_mean(t_weight)
#      t_weight = t_weight / t_weight_total

      l_loglike_poisson = tf.expand_dims(t_weight, axis=1) * tf.nn.log_poisson_loss(targets=t_labels[:,:-2], log_input=outputs[:,:-2])
      
#      poisson_column_weights = tf.ones(shape=[1, t_labels.shape[1]], dtype=t_weight.dtype)
#      poisson_column_weights = tf.concat([
#          poisson_column_weights[:,0:4] *3,
#          poisson_column_weights[:,4:16],
#          poisson_column_weights[:,16:21] *3,
#          poisson_column_weights[:,21:-2],
#          ], axis=1)
#      l_loglike_poisson *= poisson_column_weights
#
      l_xg_mse = tf.expand_dims(t_weight, axis=1) * tf.losses.mean_squared_error(labels=t_labels[:,-2:], predictions=outputs[:,-2:])
      # disable MSE loss if xGoals not present - not needed because xGoals have been set to actual goals if nothing else was available
#      l_xg_mse  *= \
#          tf.cast(tf.logical_not(tf.equal(0.0, t_labels[:,-2:-1])), tf.float32) * \
#          tf.cast(tf.logical_not(tf.equal(0.0, t_labels[:,-1:  ])), tf.float32) \
#          #* poisson_column_weights 
      
      l_softmax_1h = t_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gs_1h*7+gc_1h, logits=h1_logits)
      l_softmax_2h = t_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gs*7+gc, logits=h2_logits)
      p_full    = tf.one_hot(gs*7+gc, 49, dtype=t_weight.dtype)

      achievable_points_mask = tf.where(t_is_home_bool, 
        tf.gather(tc_home_points_i, gs*7+gc),
        tf.gather(tc_away_points_i, gs*7+gc))    
      
#      t_is_draw = tf.logical_and(tf.logical_not(t_is_home_win_bool), tf.logical_not(t_is_home_loss_bool), name="t_is_draw")
#      t_is_draw_f = tf.cast(t_is_draw, dtype=t_weight.dtype)
#      t_is_win_f = tf.cast(tf.greater(gs, gc), tf.float32)
#      t_is_loss_f = tf.cast(tf.less(gs, gc), tf.float32)
#      epsilon = 1e-7
#      l_tendency = -t_weight * tf.where(t_is_home_bool,
#                                       point_scheme[3][0]*tf.log(predictions["sp/p_pred_win"]+epsilon)*tf.cast(t_is_home_win_bool, t_weight.dtype) + 
#                                       point_scheme[3][1]*tf.log(predictions["sp/p_pred_draw"]+epsilon)*t_is_draw_f + 
#                                       point_scheme[3][2]*tf.log(predictions["sp/p_pred_loss"]+epsilon)*tf.cast(t_is_home_loss_bool, t_weight.dtype) ,
#                                       
#                                       point_scheme[3][0]*tf.log(predictions["sp/p_pred_loss"]+epsilon)*tf.cast(t_is_home_win_bool, t_weight.dtype) + 
#                                       point_scheme[3][1]*tf.log(predictions["sp/p_pred_draw"]+epsilon)*t_is_draw_f + 
#                                       point_scheme[3][2]*tf.log(predictions["sp/p_pred_win"]+epsilon)*tf.cast(t_is_home_loss_bool, t_weight.dtype) 
#                                       )
#
#      t_gdiff = tf.one_hot(gs-gc+6, 13, name="t_gdiff", dtype=t_weight.dtype)
#      t_ignore_draw_mask = tf.stack([1.0]*6+[0.0]+[1.0]*6)
#      t_ignore_draw_mask = tf.cast(t_ignore_draw_mask, dtype=t_weight.dtype)
#      l_gdiff = t_weight * tf.reduce_sum(-tf.log(predictions["sp/p_pred_gdiff"]+epsilon) * t_gdiff * t_ignore_draw_mask, axis=1) 
#      
#      t_gdiff_select_mask = []
#      for k in range(7):
#        t1 = tf.stack([tf.cast(tf.logical_and(i//7==k, tf.equal(gs-gc, i // 7 - np.mod(i, 7))), t_weight.dtype) for i in range(49) ], name="t_gdiff_select_mask_"+str(k), axis=1)
##            1.0 if (i // 7 - np.mod(i, 7))==gs-gc & i//7==k else 0.0 for i in range(49) ], name="t_gdiff_select_mask_"+str(k))
#        t_gdiff_select_mask.append(t1)
#      t_gdiff_select_mask = tf.stack(t_gdiff_select_mask, axis=0)
#      t_gdiff_select_mask = tf.reduce_sum(t_gdiff_select_mask, axis=0, name="t_gdiff_select_mask")
#      t_gdiff_select_mask = tf.cast(t_gdiff_select_mask, dtype=t_weight.dtype)
#      
#      def softmax_loss_masked(probs, labels, mask, weight):
#        p_pred_goals_abs = probs * mask # select only scores within the correct goal difference, ignore all others
#        p_pred_goals_abs = p_pred_goals_abs / (tf.reduce_sum(p_pred_goals_abs, axis=1, keepdims=True)+1e-7) # normalize sum = 1
#        return (weight * tf.reduce_sum(-tf.log(p_pred_goals_abs+1e-7) * labels, axis=1)) 
#
#      # select only scores within the correct goal difference, ignore all others
#      l_gfull = softmax_loss_masked(predictions["sp/p_pred_12"], p_full, t_gdiff_select_mask, t_weight) 

      reg_eval_metric_ops = create_model_regularization_metrics(eval_metric_ops, predictions, t_labels, mode)
      
      l_loglike_poisson = tf.reduce_sum(l_loglike_poisson, axis=1)
      loss += tf.reduce_mean(l_loglike_poisson)
      
      l_xg_mse = tf.reduce_sum(l_xg_mse, axis=1)
      loss += tf.reduce_mean(l_xg_mse)
      
      loss += 5.0*tf.reduce_mean(l_softmax_1h) # 13
      loss += 50.0*tf.reduce_mean(l_softmax_2h) # 130
      
      t_win_mask = tf.stack([1.0 if i // 7 > np.mod(i, 7) else 0.0  for i in range(49)], name="t_win_mask")
      t_loss_mask = tf.stack([1.0 if i // 7 < np.mod(i, 7) else 0.0  for i in range(49)], name="t_loss_mask")
      t_draw_mask = tf.stack([1.0 if i // 7 == np.mod(i, 7) else 0.0  for i in range(49)], name="t_draw_mask")
      t_win_mask = tf.cast(t_win_mask, dtype=t_weight.dtype)
      t_loss_mask = tf.cast(t_loss_mask, dtype=t_weight.dtype)
      t_draw_mask = tf.cast(t_draw_mask, dtype=t_weight.dtype)
      
#      loss += tf.reduce_mean(2*l_tendency)  
#      loss += tf.reduce_mean(1.0*l_gdiff)  
#      loss += tf.reduce_mean(0.5*l_gfull)  
      
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
        return tf.reduce_sum(-tf.log(p+1e-7) * labels_weighted, axis=1)
      
      def softpoint_loss(p):
        t_wdl_pred = tf.matmul(p, t_wdl_mask)   
        t_leg2_pred = tf.matmul(p, t_leg2_mask)   
        l_softmax1 = t_weight * weighted_softmax_cross_entropy(t_wdl_labels*t_pred_tendency_weight, t_wdl_pred) 
        l_softmax2 = t_weight * weighted_softmax_cross_entropy(t_leg2_labels, t_leg2_pred) 
        softpoints = t_weight * weighted_softmax_cross_entropy(-achievable_points_mask, p * t_draw_bias_mask)
        
        return 0.05*l_softmax1 + 0.2*l_softmax2, softpoints
      
      pt_pgpt_sm_loss, pt_pgpt_softpoints = softpoint_loss(predictions["pgpt/p_pred_12"])
      #pt_pgpt_softpoints = tf.reduce_sum(predictions["pgpt/p_pred_12"] * achievable_points_mask, axis=1)

      loss -= 0.05*tf.reduce_mean(pt_pgpt_softpoints)

      loss += 0.05*tf.reduce_mean(pt_pgpt_sm_loss)


      pt_cbsp_softpoints = tf.reduce_sum(tf.minimum(0.2, predictions["cbsp/p_pred_12"]) * achievable_points_mask, axis=1)
      loss -= 10*tf.reduce_mean(pt_cbsp_softpoints)
      
      xpt_softpoints = tf.reduce_sum(tf.minimum(0.2, predictions["xpt/p_pred_12"]) * achievable_points_mask, axis=1)
      loss -= 10*tf.reduce_mean(xpt_softpoints)

      with tf.variable_scope("cbsp"):
        create_laplacian_loss(predictions["cbsp/p_pred_12"], alpha=0.001)
      #create_laplacian_loss(predictions["sp/p_pred_12"], alpha=1.0) # 100
      
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
      
      eval_metric_ops.update(collect_summary("losses", "l_xg_mse", mode, tensor=l_xg_mse))
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

      eval_metric_ops.update(collect_summary("losses", "pt_cbsp_softpoints", mode, tensor=pt_cbsp_softpoints))
      eval_metric_ops.update(collect_summary("losses", "xpt_softpoints", mode, tensor=xpt_softpoints))
      
    return eval_metric_ops, loss

  def create_f1_metrics(prefix, predictions, labels, labels2, t_is_home_bool, mode):      
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      return {}  

    dtype = predictions[prefix+"p_pred_12"].dtype
    
    pGS = predictions[prefix+"pred"][:,0]
    pGC = predictions[prefix+"pred"][:,1]

    is_draw = tf.equal(labels, labels2)
    p_pred_draw = predictions[prefix+"p_pred_draw"]
    #pred_draw = tf.equal(pGS, pGC)
    pred_draw = tf.cast(tf.equal(pGS,pGC), dtype)
    is_zero = tf.equal(labels, 0) | tf.equal(labels2, 0)
    p_pred_zero = predictions[prefix+"p_pred_zero"]
    pred_zero = tf.equal(pGS, 0) | tf.equal(pGC, 0)
    
    is_win = tf.greater(labels, labels2) 
    is_loss = tf.less(labels, labels2) 
    is_win, is_loss = tf.where(t_is_home_bool, is_win, is_loss), tf.where(t_is_home_bool, is_loss, is_win)
    p_pred_win = tf.where(t_is_home_bool, predictions[prefix+"p_pred_win"], predictions[prefix+"p_pred_loss"])
    p_pred_loss = tf.where(t_is_home_bool, predictions[prefix+"p_pred_loss"], predictions[prefix+"p_pred_win"])
    pred_win = tf.greater(pGS, pGC) 
    pred_loss = tf.less(pGS, pGC) 
    pred_win, pred_loss = tf.where(t_is_home_bool, pred_win, pred_loss), tf.where(t_is_home_bool, pred_loss, pred_win)

    is_win2 = tf.greater(labels, labels2+1) 
    is_loss2 = tf.less(labels, labels2-1) 
    is_win2, is_loss2 = tf.where(t_is_home_bool, is_win2, is_loss2), tf.where(t_is_home_bool, is_loss2, is_win2)
    p_pred_win2 = tf.where(t_is_home_bool, predictions[prefix+"p_pred_win2"], predictions[prefix+"p_pred_loss2"])
    p_pred_loss2 = tf.where(t_is_home_bool, predictions[prefix+"p_pred_loss2"], predictions[prefix+"p_pred_win2"])
    pred_win2 = tf.greater(pGS, pGC+1) 
    pred_loss2 = tf.less(pGS, pGC-1) 
    pred_win2, pred_loss2 = tf.where(t_is_home_bool, pred_win2, pred_loss2), tf.where(t_is_home_bool, pred_loss2, pred_win2)

    is_win3 = tf.greater(labels, labels2+2) 
    is_loss3 = tf.less(labels, labels2-2) 
    is_win3, is_loss3 = tf.where(t_is_home_bool, is_win3, is_loss3), tf.where(t_is_home_bool, is_loss3, is_win3)
    p_pred_win3 = tf.where(t_is_home_bool, predictions[prefix+"p_pred_win3"], predictions[prefix+"p_pred_loss3"])
    p_pred_loss3 = tf.where(t_is_home_bool, predictions[prefix+"p_pred_loss3"], predictions[prefix+"p_pred_win3"])
    pred_win3 = tf.greater(pGS, pGC+2) 
    pred_loss3 = tf.less(pGS, pGC-2) 
    pred_win3, pred_loss3 = tf.where(t_is_home_bool, pred_win3, pred_loss3), tf.where(t_is_home_bool, pred_loss3, pred_win3)
    
    def f1_tensor(actual, predicted, weights=1.0):
        actual = tf.cast(actual, tf.float32)
        predicted = tf.cast(predicted, tf.float32)
        weights = tf.cast(weights, tf.float32)
        TP = tf.count_nonzero(weights * predicted * actual)
        #TN = tf.count_nonzero(weights * (predicted - 1) * (actual - 1))
        FP = tf.count_nonzero(weights * predicted * (actual - 1))
        FN = tf.count_nonzero(weights * (predicted - 1) * actual)
        precision = tf.divide(TP, TP + FP)
        recall = tf.divide(TP, TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
        
    with tf.variable_scope(prefix+"F1"):
      eval_metric_ops = {}
      if False:
        eval_metric_ops[prefix+"f1_draw_opt"]=(f1_score(is_draw, p_pred_draw), None)
        #eval_metric_ops[prefix+"f1_draw_act"]=(f1_score(is_draw, tf.cast(pred_draw, dtype), num_thresholds=200), None)
        #tf.summary.histogram("pred_draw", tf.cast(pred_draw, dtype))
        #eval_metric_ops[prefix+"f1_zero_opt_diff"]=(f1_score(is_zero, 1.0-p_pred_win-p_pred_loss), None)
        eval_metric_ops[prefix+"f1_zero_opt"]=(f1_score(is_zero, p_pred_zero), None)
        #eval_metric_ops[prefix+"f1_zero_act"]=(f1_score(is_zero, tf.cast(pred_zero, dtype), num_thresholds=200), None)
  
        #eval_metric_ops[prefix+"f1_win_opt_diff"]=(f1_score(is_win, p_pred_win-p_pred_loss), None)
        eval_metric_ops[prefix+"f1_win_opt"]=(f1_score(is_win, p_pred_win), None)
        #eval_metric_ops[prefix+"f1_win_act"]=(f1_score(is_win, tf.cast(pred_win, dtype), num_thresholds=200), None)
        #eval_metric_ops[prefix+"f1_win2_opt_diff"]=(f1_score(is_win2, p_pred_win-p_pred_loss), None)
        eval_metric_ops[prefix+"f1_win2_opt"]=(f1_score(~is_win2, 1.0-p_pred_win2, weights=tf.cast(is_win, tf.float32)), None)
        #eval_metric_ops[prefix+"f1_win2_act"]=(f1_score(is_win2, tf.cast(pred_win2, dtype), num_thresholds=200), None)
        #eval_metric_ops[prefix+"f1_win3_opt_diff"]=(f1_score(is_win3, p_pred_win-p_pred_loss), None)
        eval_metric_ops[prefix+"f1_win3_opt"]=(f1_score(~is_win3, 1.0-p_pred_win3, weights=tf.cast(is_win2, tf.float32)), None)
        #eval_metric_ops[prefix+"f1_win3_act"]=(f1_score(is_win3, tf.cast(pred_win3, dtype), num_thresholds=200), None)
  
        #eval_metric_ops[prefix+"f1_loss_opt_diff"]=(f1_score(is_loss, p_pred_loss-p_pred_win), None)
        eval_metric_ops[prefix+"f1_loss_opt"]=(f1_score(is_loss, p_pred_loss), None)
        #eval_metric_ops[prefix+"f1_loss_act"]=(f1_score(is_loss, tf.cast(pred_loss, dtype), num_thresholds=200), None)
        #eval_metric_ops[prefix+"f1_loss2_opt_diff"]=(f1_score(is_loss2, p_pred_loss-p_pred_win), None)
        eval_metric_ops[prefix+"f1_loss2_opt"]=(f1_score(~is_loss2, 1.0-p_pred_loss2, weights=tf.cast(is_loss, tf.float32)), None)
        #eval_metric_ops[prefix+"f1_loss2_act"]=(f1_score(is_loss2, tf.cast(pred_loss2, dtype), num_thresholds=200), None)
        #eval_metric_ops[prefix+"f1_loss3_opt_diff"]=(f1_score(is_loss3, p_pred_loss-p_pred_win), None)
        eval_metric_ops[prefix+"f1_loss3_opt"]=(f1_score(~is_loss3, 1.0-p_pred_loss3, weights=tf.cast(is_loss2, tf.float32)), None)
        #eval_metric_ops[prefix+"f1_loss3_act"]=(f1_score(is_loss3, tf.cast(pred_loss3, dtype), num_thresholds=200), None)
        
        eval_metric_ops[prefix+"f1_draw_prec"]=(precision(is_draw, pred_draw), None)
        eval_metric_ops[prefix+"f1_zero_prec"]=(precision(is_zero, pred_zero), None)
        eval_metric_ops[prefix+"f1_win_prec"]=(precision(is_win, pred_win), None)
        eval_metric_ops[prefix+"f1_win2_prec"]=(precision(~is_win2, ~pred_win2, weights=tf.cast(is_win, tf.float32)), None)
        eval_metric_ops[prefix+"f1_win3_prec"]=(precision(~is_win3, ~pred_win3, weights=tf.cast(is_win2, tf.float32)), None)
        eval_metric_ops[prefix+"f1_loss_prec"]=(precision(is_loss, pred_loss), None)
        eval_metric_ops[prefix+"f1_loss2_prec"]=(precision(~is_loss2, ~pred_loss2, weights=tf.cast(is_loss, tf.float32)), None)
        eval_metric_ops[prefix+"f1_loss3_prec"]=(precision(~is_loss3, ~pred_loss3, weights=tf.cast(is_loss2, tf.float32)), None)
  
        eval_metric_ops[prefix+"f1_draw_rec"]=(recall(is_draw, pred_draw), None)
        eval_metric_ops[prefix+"f1_zero_rec"]=(recall(is_zero, pred_zero), None)
        eval_metric_ops[prefix+"f1_win_rec"]=(recall(is_win, pred_win), None)
        eval_metric_ops[prefix+"f1_win2_rec"]=(recall(~is_win2, ~pred_win2, weights=tf.cast(is_win, tf.float32)), None)
        eval_metric_ops[prefix+"f1_win3_rec"]=(recall(~is_win3, ~pred_win3, weights=tf.cast(is_win2, tf.float32)), None)
        eval_metric_ops[prefix+"f1_loss_rec"]=(recall(is_loss, pred_loss), None)
        eval_metric_ops[prefix+"f1_loss2_rec"]=(recall(~is_loss2, ~pred_loss2, weights=tf.cast(is_loss, tf.float32)), None)
        eval_metric_ops[prefix+"f1_loss3_rec"]=(recall(~is_loss3, ~pred_loss3, weights=tf.cast(is_loss2, tf.float32)), None)
  
        eval_metric_ops[prefix+"f1_draw_pred"]=(precision(pred_draw, tf.ones_like(pred_draw)), None)
        eval_metric_ops[prefix+"f1_zero_pred"]=(precision(pred_zero, tf.ones_like(pred_draw)), None)
        eval_metric_ops[prefix+"f1_win_pred"]=(precision(pred_win, tf.ones_like(pred_draw)), None)
        eval_metric_ops[prefix+"f1_win2_pred"]=(precision(~pred_win2, tf.ones_like(pred_draw), weights=tf.cast(is_win, tf.float32)), None)
        eval_metric_ops[prefix+"f1_win3_pred"]=(precision(~pred_win3, tf.ones_like(pred_draw), weights=tf.cast(is_win2, tf.float32)), None)
        eval_metric_ops[prefix+"f1_loss_pred"]=(precision(pred_loss, tf.ones_like(pred_draw)), None)
        eval_metric_ops[prefix+"f1_loss2_pred"]=(precision(~pred_loss2, tf.ones_like(pred_draw), weights=tf.cast(is_loss, tf.float32)), None)
        eval_metric_ops[prefix+"f1_loss3_pred"]=(precision(~pred_loss3, tf.ones_like(pred_draw), weights=tf.cast(is_loss2, tf.float32)), None)
        prefix = prefix[:-1]
  #      eval_metric_ops.update(collect_summary(prefix, "pred_draw", mode, tensor=pred_draw))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_draw", mode, tensor=p_pred_draw))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_zero", mode, tensor=p_pred_zero))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_win", mode, tensor=p_pred_win))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_win2", mode, tensor=p_pred_win2))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_win3", mode, tensor=p_pred_win3))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_loss", mode, tensor=p_pred_loss))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_loss2", mode, tensor=p_pred_loss2))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_loss3", mode, tensor=p_pred_loss3))
  #      eval_metric_ops.update(collect_summary(prefix, "p_pred_win_loss_diff", mode, tensor=p_pred_win-p_pred_loss))
  #
        eval_metric_ops.update(collect_summary(prefix, "f1_draw_act", mode, tensor=f1_tensor(is_draw, pred_draw)))
        eval_metric_ops.update(collect_summary(prefix, "f1_zero_act", mode, tensor=f1_tensor(is_zero, pred_zero)))
        eval_metric_ops.update(collect_summary(prefix, "f1_win_act", mode, tensor=f1_tensor(is_win, pred_win)))
        eval_metric_ops.update(collect_summary(prefix, "f1_win2_act", mode, tensor=f1_tensor(~is_win2, ~pred_win2, weights=is_win)))
        eval_metric_ops.update(collect_summary(prefix, "f1_win3_act", mode, tensor=f1_tensor(~is_win3, ~pred_win3, weights=is_win2)))
        eval_metric_ops.update(collect_summary(prefix, "f1_loss_act", mode, tensor=f1_tensor(is_loss, pred_loss)))
        eval_metric_ops.update(collect_summary(prefix, "f1_loss2_act", mode, tensor=f1_tensor(~is_loss2, ~pred_loss2, weights=is_loss)))
        eval_metric_ops.update(collect_summary(prefix, "f1_loss3_act", mode, tensor=f1_tensor(~is_loss3, ~pred_loss3, weights=is_loss2)))
  
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_win", mode, tensor=tf.cast(pred_win, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_loss", mode, tensor=tf.cast(pred_loss, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_win2", mode, tensor=tf.cast(tf.logical_and(pred_win2, tf.logical_not(pred_win3)), tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_loss2", mode, tensor=tf.cast(tf.logical_and(pred_loss2, tf.logical_not(pred_loss3)), tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_win3", mode, tensor=tf.cast(pred_win3, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_loss3", mode, tensor=tf.cast(pred_loss3, tf.float32)))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_zero", mode, tensor=tf.cast(pred_zero, tf.float32)))
      
    return eval_metric_ops

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
      #pred_home_win = tf.cast((tf.greater(pGS, pGC) & t_is_home_bool) | (tf.less(pGS, pGC) & tf.logical_not(t_is_home_bool)), dtype)
      #pred_away_win = tf.cast((tf.less(pGS, pGC) & t_is_home_bool) | (tf.greater(pGS, pGC) & tf.logical_not(t_is_home_bool)), dtype)

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
      #eval_metric_ops.update(collect_summary(prefix, "metric_pred_home_win", mode, tensor=pred_home_win))
      #eval_metric_ops.update(collect_summary(prefix, "metric_pred_away_win", mode, tensor=pred_away_win))
      eval_metric_ops.update(collect_summary(prefix, "metric_pred_draw", mode, tensor=pred_draw))
      eval_metric_ops.update(collect_summary(prefix, "pt_softpoints", mode, tensor=pt_softpoints))
      eval_metric_ops.update(collect_summary(prefix, "metric_ev_goals_diff_L1", mode, tensor=l_diff_ev_goals_L1))
      eval_metric_ops.update(collect_summary(prefix, "metric_cor_diff", mode, tensor=corrcoef(ev_goals_1-ev_goals_2, labels_float-labels_float2)))
    
    eval_metric_ops.update(create_f1_metrics(prefix+"/", predictions, labels, labels2, t_is_home_bool, mode))
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
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      # shuffle the labels such that the results from another match of either Team1 or Team2 is fed as label 
      match_history_t1_seqlen = tf.cast(10*features["newgame"][:,0], tf.int32)
      match_history_t2_seqlen = tf.cast(10*features["newgame"][:,2], tf.int32)
      
      def create_shuffled_labels(mh, mhseqlen):
        rand1 = tf.random_uniform(dtype=tf.int32, minval=0, maxval=10000, shape=tf.shape(features["newgame"])[0:1])
        hist_idx = tf.mod(rand1, tf.maximum(mhseqlen-3, 1))+2
        cat_idx = tf.stack([tf.range(0, tf.shape(features["newgame"])[0]), tf.shape(mh)[1]-hist_idx], axis=1) # minus "-": count from end of sequence
        hist_idx = tf.gather_nd(mh, cat_idx)
        hist_idx = tf.squeeze(hist_idx)
        hist_idx = tf.cast(hist_idx, tf.int32)
        selected_batch_shuffled = tf.where(mhseqlen<=2, tf.squeeze(selected_batch), hist_idx)
        labels_shuffled = tf.gather(alllabels_placeholder, selected_batch_shuffled)
        labels_shuffled = tf.cast(labels_shuffled, tf.float32)
        return labels_shuffled 

      labels_shuffled1 = create_shuffled_labels(features["match_history_t1"], match_history_t1_seqlen)
      labels_shuffled2 = create_shuffled_labels(features["match_history_t2"], match_history_t2_seqlen)
      randomvalue = tf.random.uniform((1,))[0]
      labels = tf.cond(randomvalue > 0.15, 
                       lambda: labels, 
                       lambda: tf.cond(randomvalue < 0.075, 
                                       lambda: labels_shuffled1, 
                                       lambda: labels_shuffled2)
                       )

    
    alldata0 = tf.concat([alldata_placeholder[0:1]*0.0, alldata_placeholder], axis=0)
    alllabels0 = tf.concat([alllabels_placeholder[0:1]*0.0, alllabels_placeholder], axis=0)
    def build_history_input(name):
      hist_idx = tf.cast(features[name], tf.int32)
#      print_op = tf.print(name, hist_idx[0:5], output_stream=sys.stdout)
#      with tf.control_dependencies([print_op]):
      hist_idx = hist_idx+1
      hist_data = tf.gather(params=alldata0, indices=hist_idx)
      hist_labels = tf.gather(params=alllabels0, indices=hist_idx)
      features[name] = tf.concat([hist_data, hist_labels], axis=2)
      features[name] = tf.cast(features[name], tf.float32)

    build_history_input("match_history_t1")
    build_history_input("match_history_t2")
    build_history_input("match_history_t12")
    
    with tf.variable_scope("Model"):

      #features = {k:decode_home_away_matches(f) for k,f in features.items() }
      #labels   = decode_home_away_matches(labels)

      t_is_home_bool = tf.equal(features["newgame"][:,1] , 1)
      graph_outputs = buildGraph(features, labels, mode, params, t_is_home_bool)
      #outputs, sp_logits, hidden_layer, eval_metric_ops, cond_probs, decode_t1, decode_t2, decode_t12  = graph_outputs
      outputs, sp_logits, hidden_layer, eval_metric_ops, cond_probs, cbsp_logits = graph_outputs
#      t_is_train_bool = tf.equal(features["Train"] , True)

      def apply_prefix(predictions, prefix):
        return {prefix+k:v for k,v in predictions.items() }

      
      with tf.variable_scope("sp"):
          predictions, sp_loss = create_hierarchical_losses_and_predictions(sp_logits, labels, features, t_is_home_bool, mode, tc, eval_metric_ops)
          #predictions = create_hierarchical_predictions(outputs, sp_logits, t_is_home_bool, tc, mode, prefix="sp", apply_point_scheme=False)
          #predictions = create_quantile_scheme_prediction(outputs, sp_logits, t_is_home_bool, tc, mode, prefix="sp")
      
        #predictions = create_predictions(outputs, sp_logits, t_is_home_bool, tc)
      predictions = apply_prefix(predictions, "sp/")

      h1_logits, h2_logits, p_pred_h1, label_features_h1, p_pred_h2, label_features_h2, t_mask, test_p_pred_12_h2, p_pred_12_h2 = cond_probs
      
      predictions["cp/test_p_pred_12_h2"]=test_p_pred_12_h2
      h2_logits = tf.log(p_pred_12_h2) # use predicted probabilities as logits
      
      predictions["cp/p_pred_12_h2"]=p_pred_12_h2
      
      if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        predictions["cp1/logits"] = h1_logits
        predictions["cp/logits"] = h2_logits
        predictions["cp1/outputs"] = p_pred_h1
        predictions["cp/outputs"] = p_pred_h2
        predictions["cp1/labels"] = label_features_h1
        predictions["cp/labels"] = label_features_h2
            
      cbsp_predictions = create_predictions(outputs, cbsp_logits, t_is_home_bool, tc, False)
      # select largest p_pred_12 after smoothing
      cbsp_predictions ["p_pred_12_sm"] = apply_kernel_smoothing(cbsp_predictions ["p_pred_12"])
      a = tf.argmax(cbsp_predictions ["p_pred_12_sm"], axis=1)
      pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
      pred = tf.cast(pred, tf.int32)
      cbsp_predictions ["pred"]=pred 

      cbsp_predictions = calc_probabilities(cbsp_predictions["p_pred_12"], cbsp_predictions)
      cbsp_predictions = apply_poisson_summary(cbsp_predictions["p_pred_12"], t_is_home_bool, tc, predictions = cbsp_predictions)
      predictions.update(apply_prefix(cbsp_predictions , "cbsp/"))

      with tf.variable_scope("cp"):
          #cp_predictions = create_hierarchical_predictions(outputs, h2_logits, t_is_home_bool, tc, mode, prefix="cp")
          cp_predictions = create_quantile_scheme_prediction(outputs, h2_logits, t_is_home_bool, tc, mode, prefix="cp", p_pred_12 = predictions["cp/p_pred_12_h2"])
      predictions.update(apply_prefix(cp_predictions, "cp/"))
      cp1_predictions = create_predictions(outputs, h1_logits, t_is_home_bool, tc, False)
      predictions.update(apply_prefix(cp1_predictions, "cp1/"))

      avg_p_pred_12 = predictions["sp/p_pred_12"]+predictions["cp/p_pred_12_h2"] # averaging of sp and cp strategies 
      avg_logits = tf.log(avg_p_pred_12) 
      with tf.variable_scope("av"):
#        avg_predictions = create_predictions(outputs, avg_logits, t_is_home_bool, tc, False)
#        avg_pred = create_fixed_scheme_prediction_new(avg_predictions["p_pred_12"], t_is_home_bool, mode)
#        avg_predictions.update({"pred":avg_pred})
#        predictions.update(apply_prefix(avg_predictions, "av/"))
        # avg_predictions = create_hierarchical_predictions(outputs, avg_logits, t_is_home_bool, tc, mode, prefix="av")
        avg_predictions = create_quantile_scheme_prediction(outputs, avg_logits, t_is_home_bool, tc, mode, prefix="av", p_pred_12 = avg_p_pred_12)
        predictions.update(apply_prefix(avg_predictions, "av/"))
        
        # for avmx - average the ev_points, not the logits or p_pred_12 probabilities
        avmx_predictions = create_predictions(outputs, avg_logits, t_is_home_bool, tc, True)
        avmx_predictions = calc_probabilities(avmx_predictions["p_pred_12"], avmx_predictions)
        avmx_predictions["ev_points"] = (predictions["sp/ev_points"]+predictions["cp/ev_points"])/2
        # select the maximum ev_points 
        a = tf.argmax(avmx_predictions["ev_points"], axis=1)
        pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
        pred = tf.cast(pred, tf.int32)
        avmx_predictions["pred"]=pred 
        predictions.update(apply_prefix(avmx_predictions, "avmx/"))
        
      with tf.variable_scope("cpmx"):
        cpmx_predictions = create_predictions(outputs, h2_logits, t_is_home_bool, tc, True)
        cpmx_predictions = calc_probabilities(cpmx_predictions["p_pred_12"], cpmx_predictions)
        predictions.update(apply_prefix(cpmx_predictions, "cpmx/"))

      T1_GFT = tf.exp(outputs[:,0])
      T2_GFT = tf.exp(outputs[:,1])
      T1_GHT = tf.exp(outputs[:,2])
      T2_GHT = tf.exp(outputs[:,3])
      T1_GH2 = tf.exp(outputs[:,16])
      T2_GH2 = tf.exp(outputs[:,17])
      T1_xG = tf.exp(outputs[:,-2])
      T2_xG = tf.exp(outputs[:,-1])
      T1_GFT_est = (T1_GFT+T1_GHT+T1_GH2+T1_xG)/3
      T2_GFT_est = (T2_GFT+T2_GHT+T2_GH2+T2_xG)/3
      epsilon = 1e-7
      predictions_poisson_FT = create_predictions_from_ev_goals(T1_GFT_est, T2_GFT_est, t_is_home_bool, tc)
      
      p1pt_predictions = {k:v for k,v in predictions_poisson_FT.items() } # copy
      p1pt_predictions.update(point_maximization_layer(outputs, tf.stack([tf.log(T1_GFT_est+epsilon), tf.log(T2_GFT_est+epsilon)], axis=1), "pgpt", t_is_home_bool, tc, mode))
      p1pt_predictions = apply_prefix(p1pt_predictions, "pgpt/" )
      predictions.update(apply_prefix(predictions_poisson_FT, "pg/"))
      predictions.update(p1pt_predictions)

      predictions["outputs_poisson"] = outputs
      #predictions["index"] = index
      
      ########################################
      with tf.variable_scope("xpt"):
          xpt_predictions = point_maximization_layer(outputs, tf.concat([predictions["cp/p_pred_12_h2"]], axis=1), "xpt", t_is_home_bool, tc, mode, use_max_points=True)

          xpt_predictions  = calc_probabilities(xpt_predictions ["p_pred_12"], xpt_predictions)
          xpt_predictions = apply_poisson_summary(xpt_predictions["p_pred_12"], t_is_home_bool, tc, predictions = xpt_predictions)

          #xpt_predictions ["p_pred_12_sm"] = apply_kernel_smoothing(xpt_predictions ["p_pred_12"])
#          a = tf.argmax(xpt_predictions ["ev_points"], axis=1)
#          pred = tf.reshape(tf.stack([a // 7, tf.mod(a, 7)], axis=1), [-1,2])
#          pred = tf.cast(pred, tf.int32)
#          xpt_predictions ["pred"]=pred 

          predictions.update(apply_prefix(xpt_predictions, "xpt/"))
      
      ########################################

      for k,v in segmentation_strategies.items():
        with tf.variable_scope(k):
          if v=="cp2":
            #segm_pred = create_fixed_scheme_prediction_new(predictions[k+"/p_pred_12"], t_is_home_bool, mode)
            segm_pred = create_hierarchical_predictions(outputs, h2_logits, t_is_home_bool, tc, mode, prefix=v, p_pred_12 = predictions[k+"/p_pred_12"], apply_point_scheme=True)
            segm_pred = segm_pred["pred"] 
          else:
            segm_pred = create_quantile_scheme_prediction(outputs, predictions[k+"/p_pred_12"], t_is_home_bool, tc, mode, prefix=v)
            segm_pred = segm_pred["pred"] 
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

      #eval_metric_ops.update(collect_summary("global_step", "t_is_home_bool", mode, tensor=tf.cast(t_is_home_bool, tf.float32)))
      #tf.summary.scalar("global_step/t_is_home_bool", tf.reduce_mean(tf.cast(t_is_home_bool, tf.float32)))
      #tf.summary.scalar("global_step/t_is_home_bool0", tf.reduce_mean(tf.cast(t_is_home_bool[0::2], tf.float32)))
      #tf.summary.scalar("global_step/t_is_home_bool1", tf.reduce_mean(tf.cast(t_is_home_bool[1::2], tf.float32)))
      
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

    eval_loss_ops, loss = create_losses_RNN(outputs, sp_loss, cond_probs, labels, features, predictions, t_is_home_bool, mode, tc, t_is_home_win_bool , t_is_home_loss_bool, eval_metric_ops)
    eval_metric_ops.update(eval_loss_ops)
    
#    eval_ae_loss_ops, ae_loss = create_autoencoder_losses(loss, decode_t1, decode_t2, decode_t12, features, labels, mode)  
#    eval_metric_ops.update(eval_ae_loss_ops)
#    loss += ae_loss
    
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
    #exclude_list = ["CNN"]
    def skip_gradients(gradients, variables, exclude_list): 
      gradvars = [(g,v) for g,v in zip(gradients, variables) if g is not None and not any(s in v.name for s in exclude_list)]      
      gradients, variables = zip(*gradvars)
#      print(variables)
#      print(gradients)
      return gradients, variables
      
    # last 100 steps: set gradient to 0 for exclude_list
    def null_gradients(gradients, variables, exclude_list): 
      return [g if g is None or not any(s in v.name for s in exclude_list) else \
              tf.cast(tf.mod(global_step-1, save_steps)<=(save_steps-30), tf.float32)*g for g,v in zip(gradients, variables)]

    def large_gradients(gradients, variables, exclude_list): 
      # double the gradient for all except those in exclude list
      return [g if g is None else \
              tf.cast(tf.mod(global_step-1, save_steps)<=(save_steps-30), tf.float32)*g \
                        if any(s in v.name for s in exclude_list) else \
              g + 0.5*g*tf.cast(tf.mod(global_step-1, save_steps) > (save_steps-30), tf.float32) for g,v in zip(gradients, variables)]

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

#    # ##################################
#    # freeze Layer0 in the last 100 steps and increase regularization on remaining layers instead
#    reg_gradients = large_gradients(reg_gradients, reg_variables, exclude_list = ["Layer0", "Softpoints/WDL"])
#    gradients = null_gradients(gradients, variables, exclude_list = ["Layer0", "Softpoints/WDL"])

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
                                               "xpt" : eval_metric_ops["summary/xpt/z_points"][0][1],
                                               "pg2" : eval_metric_ops["summary/pg2/z_points"][0][1],
                                               "sp" : eval_metric_ops["summary/sp/z_points"][0][1], 
                                               "cbsp" : eval_metric_ops["summary/cbsp/z_points"][0][1]}, 
      every_n_iter=250)
    
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
                                    save_summary_steps=250,
                                    keep_checkpoint_max=max_to_keep,
                                    log_step_count_steps=100),
                                #warm_start_from = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="D:/Models/conv1_auto_pistor4", vars_to_warm_start=[".*CNN.*", ".*PointMax.*", ".*Softpoints.*", ".*Poisson.*", ".*cp.*"])
                                #warm_start_from = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="D:/Models/xg_sp_bwin_pistor2", vars_to_warm_start=[".*CNN.*"])
                              
                              )

