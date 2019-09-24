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
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
#import shutil
import sys
import tempfile
import pickle
import time
from pathlib import Path
import csv
import pandas as pd
pd.set_option('expand_frame_repr', False)

import numpy as np

#np.set_printoptions(threshold=50)
from datetime import datetime
import os
from threading import Event
import signal

from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Estimators import Model_1920 as themodel
from Estimators import Utilities as utils

#from tensorflow.python.training.session_run_hook import SessionRunHook
#from tensorflow.contrib.layers import l2_regularizer

from collections import Counter
#from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import random
import itertools
from tensorflow.python import debug as tf_debug

#@ops.RegisterGradient("BernoulliSample_ST")
#def bernoulliSample_ST(op, grad):
#    return [tf.clip_by_norm(grad, 20.0), tf.zeros(tf.shape(op.inputs[1]))]

Feature_COLUMNS = ["HomeTeam","AwayTeam"]
Label_COLUMNS = ["FTHG","FTAG"]
CSV_COLUMNS = Feature_COLUMNS + Label_COLUMNS
Derived_COLUMNS = ["t1goals", "t2goals", "t1goals_where", "t2goals_where"]
COLS = ["HGFT","AGFT","HGHT","AGHT","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR"]
Meta_COLUMNS = ["t1games", "t2games", "t1games_where", "t2games_where"]
COLS_Extended = COLS + ['HWin', 'AWin', 'HLoss', 'ALoss', 'HDraw', 'ADraw']

point_scheme_tcs = [[4,6,7], [3,2,5], [2,2,4], [2.62, 3.77, 4.93], [285/9/27, 247/9/26, 196/9/27]]
point_scheme_pistor = [[3,3,3], [2,2,2], [1,1,1], [1.65, 2.46, 1.73], [285/9/27, 247/9/26, 196/9/27]]
point_scheme_sky = [[5,5,5], [2,2,2], [2,2,2], [2.60, 3.38, 2.66], [285/9/27, 247/9/26, 196/9/27]]
point_scheme_goal_diff = [[-1,-1,-1], [-1,-1,-1], [-1,-1,-1], [7, 9, 7.5], [285/9/27, 247/9/26, 196/9/27]]

point_scheme = point_scheme_pistor

SEQ_LENGTH = 4
TIMESERIES_COL = 'rawdata'


def get_train_test_data(model_dir, train_seasons, test_seasons, data_dir):
  full_data =   pd.read_csv(data_dir+"/full_data.csv")
  is_train = full_data.Season.astype(str).apply(lambda x: x.zfill(4)).isin(train_seasons).repeat(2)
  is_test = full_data.Season.astype(str).apply(lambda x: x.zfill(4)).isin(test_seasons).repeat(2)
  is_test.value_counts()
  
  all_data =  pd.read_csv(data_dir+"/all_features.csv")
  all_labels =  pd.read_csv(data_dir+"/all_labels.csv") 
  team_mapping = pd.read_csv(data_dir+"/team_mapping.csv") 
  #all_features = pd.read_csv(data_dir+"/lfda_data.csv") 
  feature_names = pd.read_csv(data_dir+"/feature_candidates_long.csv")
  print(feature_names)
  feature_names = feature_names.x.tolist() 
  if 'BW1' in feature_names:
    # place 'BW1', 'BW2', 'BW0' first is list, so that they have a fixed colunm position
    feature_names = pd.Series(['BW1', 'BW2', 'BW0'] + feature_names).drop_duplicates().tolist()
  all_features = all_data[feature_names]
  all_data["Train"]=is_train.values&(~all_data["Predict"])
  all_data["Test"]=is_test.values&(~all_data["Predict"])

#  all_labels["Train"]=is_train.values
#  all_labels["Test"]=is_test.values
#  all_labels["Predict"] = all_data["Predict"]
#
#  all_features["Train"]=is_train.values
#  all_features["Test"]=is_test.values
#  all_features["Predict"] = all_data["Predict"]
  teamnames = team_mapping.Teamname.tolist()

  return all_data, all_labels, all_features, teamnames, team_mapping

def build_features(all_data, all_labels, all_features, teamnames, team_mapping):
  
  all_data["gameindex"]=all_data.index

  mh1 = [all_data.groupby("Team1_index").gameindex.shift(i) for i in range(SEQ_LENGTH, 0, -1)]
  mh1 = np.stack(mh1, axis=1)
  all_data["mh1len"] = np.sum(~pd.isna(mh1), axis=1)
  mh1[np.isnan(mh1)]=-1
  mh1 = mh1.astype(np.int16)
  
  mh2 = [all_data.groupby("Team2_index").gameindex.shift(i) for i in range(SEQ_LENGTH, 0, -1)]
  mh2 = np.stack(mh2, axis=1)
  all_data["mh2len"] = np.sum(~pd.isna(mh2), axis=1)
  mh2[np.isnan(mh2)]=-1
  mh2 = mh2.astype(np.int16)
  
  mh12 = [all_data.groupby(["Team1_index","Team2_index"]).gameindex.shift(i) for i in range(SEQ_LENGTH, 0, -1)]
  mh12 = np.stack(mh12, axis=1)
  all_data["mh12len"] = np.sum(~pd.isna(mh12), axis=1)
  mh12[np.isnan(mh12)]=-1
  mh12 = mh12.astype(np.int16)
  
  label_column_names = all_labels.columns
#  feature_column_names = all_features.columns
  
  prefix0 = all_data[["mh1len", "Where", "mh2len", "mh12len"]].astype(np.float32)
  prefix = prefix0.values
  prefix[:,[0,2,3]]*=0.1
  match_input_layer = np.concatenate([prefix, all_features.values], axis=1)
  print("prefix.columns.tolist()+all_features.columns.tolist()")
  print(prefix0.columns.tolist()+all_features.columns.tolist())
#  tn = len(teamnames)
#  #lc = len(label_column_names)
#  fc = len(feature_column_names)
#
#  match_input_layer = np.zeros(shape=[len(features), 4+2*tn+fc], dtype=np.float32)
#  match_input_layer[:, 0] = features["mh1len"] * 0.1
#  match_input_layer[:, 1] = features["Where"] 
#  match_input_layer[:, 2] = features["mh2len"] * 0.1
#  match_input_layer[:, 3] = features["mh12len"] * 0.1
#  j = 4+2*tn
#  match_input_layer[:, 4:j] = teamsoh
##  match_input_layer[:, j:j+lc] = features [label_column_names]
##  j = j+lc
#  match_input_layer[:, j:j+fc] = features[feature_column_names]
#  
#  #labels = features [label_column_names].values.astype(np.float32)
  return {
      "match_input_layer": match_input_layer,
      "gameindex": all_data.gameindex.values.astype(np.int16), 
      "match_history_t1": mh1,
      "match_history_t2": mh2,
      "match_history_t12": mh12,
      }, all_labels.values, team_mapping, label_column_names

def plot_confusion_matrix(plotaxis, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plotaxis.imshow(cm, interpolation='nearest', cmap=cmap)
    plotaxis.set_title(title)
    tick_marks = np.arange(len(classes))
    plotaxis.set_xticks(tick_marks)
    plotaxis.set_xticklabels(classes, rotation=45)
    plotaxis.set_yticks(tick_marks)
    plotaxis.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plotaxis.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plotaxis.set_ylabel('True label')
    plotaxis.set_xlabel('Predicted label')

def plot_softprob(pred, gs, gc, title="", prefix=""):
  if prefix=="ens/":
    return
  
  print("-----------------------------------------------------")
  print("{} / {}:{}".format(title, gs, gc))
  default_color = np.array([[ 0.12,  0.47,  0.71,  0.5  ]])
  if prefix=="p1/" or prefix=="p3/":
    default_color = "darkolivegreen"
  if prefix=="p2/" or prefix=="p4/":
    default_color = "darkmagenta"

  sp = pred[prefix+"p_pred_12"]
  #print(sp)
  spt = pred[prefix+"ev_points"]
  gs = min(gs,6)
  gc = min(gc,6)
  margin_pred_prob1 = pred[prefix+"p_marg_1"]
  margin_poisson_prob1 = pred[prefix+"p_poisson_1"]
  margin_pred_prob2 = pred[prefix+"p_marg_2"]
  margin_poisson_prob2 = pred[prefix+"p_poisson_2"]
  margin_pred_expected1 = pred[prefix+"ev_goals_1"] 
  margin_pred_expected2 = pred[prefix+"ev_goals_2"] 
  g=[0.0,1.0,2.0,3.0,4.0,5.0,6.0]
  g1=[0.0]*7+[1.0]*7+[2.0]*7+[3.0]*7+[4.0]*7+[5.0]*7+[6.0]*7
  g2=g*7
  fig, ax = plt.subplots(1,3,figsize=(15,5))
  ax[0].scatter(g1, g2, s=sp*10000, alpha=0.4, color=default_color)
  ax[0].scatter(gs, gc, s=sp[gs*7+gc]*10000, alpha=0.7, color=default_color)
  for i, txt in enumerate(sp):
    ax[0].annotate("{:4.2f}".format(txt*100), (g1[i],g2[i]))
  ax[1].scatter(g1, g2, s=spt*500, alpha=0.4,color='red')
  ax[1].scatter(gs, gc, s=spt[gs*7+gc]*500, alpha=0.7,color='red')
  for i, txt in enumerate(spt):
    ax[1].annotate("{:4.2f}".format(txt), (g1[i],g2[i]))
  ax[0].set_title(prefix)
  ax[1].set_title(prefix)
  max_sp = max(sp)
  max_sp_index = np.argmax(sp) 
  ax[0].scatter((max_sp_index//7).astype(float), np.mod(max_sp_index, 7).astype(float), s=max_sp*10000.0, facecolors='none', edgecolors='black', linewidth=2)
  max_spt = max(spt)
  max_spt_index = np.argmax(spt) 
  ax[1].scatter((max_spt_index//7).astype(float), np.mod(max_spt_index, 7).astype(float), s=max_spt*500.0, facecolors='none', edgecolors='black', linewidth=2)
  
  p_loss=0.0
  p_win=0.0
  p_draw=0.0
  for i in range(7):
    for j in range(7):
      if i>j:
        p_win += sp[i*7+j]
      if i<j:
        p_loss += sp[i*7+j]
      if i==j:
        p_draw += sp[i*7+j]
  ax[2].axis('equal')
  explode = [0, 0, 0]
  explode[1-np.sign(gs-gc)] = 0.1
  wedges, _, _ = ax[2].pie([p_win, p_draw, p_loss], labels=["Win", "Draw", "Loss"], colors=["blue", "green", "red"], startangle=90, autopct='%1.1f%%', 
                   radius=1.0, explode=explode, wedgeprops = {"alpha":0.5})
  wedges[1-np.sign(gs-gc)].set_alpha(0.8)
  plt.show()

  w=0.35
  fig, ax = plt.subplots(1,3,figsize=(15,1))
  ax[0].bar(g, margin_pred_prob1,alpha=0.6, width=w, color=default_color)
  ax[0].bar([x+w for x in g], margin_poisson_prob1,alpha=0.3,color="red",width=0.35)
  ax[0].bar(gs, margin_pred_prob1[gs],alpha=0.5, width=w, color=default_color)
  ax[0].bar(gs+w, margin_poisson_prob1[gs],alpha=0.7,color="red",width=0.35)
  ax[0].axvline(x=margin_pred_expected1, color='red')
  ax[1].bar(g, margin_pred_prob2,alpha=0.6, width=w, color=default_color)
  ax[1].bar([x+w for x in g], margin_poisson_prob2,alpha=0.3,color="red",width=0.35)
  ax[1].bar(gc, margin_pred_prob2[gc],alpha=0.5, width=w, color=default_color)
  ax[1].bar(gc+w, margin_poisson_prob2[gc],alpha=0.7,color="red",width=0.35)
  ax[1].axvline(x=margin_pred_expected2, color='red')
  ax[0].set_title(margin_pred_expected1) 
  ax[1].set_title(margin_pred_expected2) 
  bars = ax[2].bar([0,1,2], height=[p_win, p_draw, p_loss], tick_label=["Win", "Draw", "Loss"], color=["blue", "green", "red"], alpha=0.5)
  bars[1-np.sign(gs-gc)].set_alpha(0.8)
  bars[1-np.sign(gs-gc)].set_linewidth(2)
  bars[1-np.sign(gs-gc)].set_edgecolor("black")
  for i in [-1,0,1]:
    if i != np.sign(gs-gc):
      bars[1-i].set_hatch("x")
  plt.show()

def global_prepare():
  global CALC_GRAPH, calc_points_tensor
  global pl_pGS, pl_pGC, pl_GS, pl_GC, pl_is_home
  CALC_GRAPH = tf.Graph()
  with CALC_GRAPH.as_default():
    pl_pGS = tf.placeholder(tf.int64, name="pl_pGS")
    pl_pGC = tf.placeholder(tf.int64, name="pl_pGC")
    pl_GS = tf.placeholder(tf.int64, name="pl_GS")
    pl_GC = tf.placeholder(tf.int64, name="pl_GC")
    pl_is_home = tf.placeholder(tf.bool, name="pl_is_home")
    calc_points_tensor = themodel.calc_points(pl_pGS,pl_pGC, pl_GS, pl_GC, pl_is_home)[0]


def print_match_dates(X, team_onehot_encoder):
  features = X["newgame"]
  tn = len(team_onehot_encoder.classes_)
  match_dates = features[:,4+2*tn+0]*1000.0
  match_dates = [datetime.strftime(datetime.fromordinal(int(m)+734138), "%Y/%m/%d") for m in match_dates]
  match_dates = list(sorted(set(match_dates)))  
  print(len(features))
  print(match_dates)
  return match_dates

def static_probabilities(model_data):
  model, features_arrays, labels_array, features_placeholder, train_idx, test_idx, pred_idx = model_data
  with tf.Session() as sess:
    sess=sess
    train_X = {k: v[train_idx] for k, v in features_arrays.items()}
    train_y = labels_array[train_idx]
    test_X = {k: v[test_idx] for k, v in features_arrays.items()}
    test_y = labels_array[test_idx]
    plot_softprob(themodel.makeStaticPrediction((train_X), (train_y)),6,6,"Static Prediction Train Data")
    plot_softprob(themodel.makeStaticPrediction((test_X), (test_y)),6,6,"Static Prediction Test Data")

def plot_predictions_3(df, prefix, dataset):
  df = df.loc[(df.Prefix==prefix)&(df.dataset==dataset)].copy()

  fig = plt.figure(figsize=(18,4))
  ax1 = plt.subplot2grid((1,4), (0,0), colspan=2, rowspan=1)
  ax2 = plt.subplot2grid((1,4), (0,2), colspan=1, rowspan=1)
  ax2.axis('off')
  ax3 = plt.subplot2grid((1,4), (0,3), colspan=1, rowspan=1)
  ax3.axis('off')
  
  goal_cnt = Counter([str(gs)+":"+str(gc) if w=="Home" else str(gc)+":"+str(gs) for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])])
  gdiff_cnt = Counter([gs-gc if w=="Home" else gc-gs for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])])
  tend_cnt = Counter([np.sign(gs-gc) if w=="Home" else np.sign(gc-gs) for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])])
  
  # goodness of fit:
  df["gof"] = ["3_full" if p1==gs and p2==gc else 
        "2_diff" if p1-p2==gs-gc else
        "1_tendency" if np.sign(p1-p2)==np.sign(gs-gc) else
        "0_none"
        for p1,p2, gs, gc in zip(df["pGS"], df["pGC"], df["GS"], df["GC"])]
  df["pFTHG"] =  [p1 if w=="Home" else p2 for p1,p2,w in zip(df["pGS"],df["pGC"],df["Where"])]
  df["pFTAG"] =  [p2 if w=="Home" else p1 for p1,p2,w in zip(df["pGS"],df["pGC"],df["Where"])]
  df["pGoals"] = [str(g1)+":"+str(g2) for g1, g2 in zip(df["pFTHG"],df["pFTAG"]) ]
  df["pGDiff"] = df["pFTHG"]-df["pFTAG"]
  df["pTendency"] = np.sign(df["pFTHG"]-df["pFTAG"])
  df["total_points"]=0
  
  if point_scheme[0][0]==-1:
    # GoalDiff calculation
    df["total_points"] = (np.sign(df["pGS"]-df["pGC"])==np.sign(df["GS"]-df["GC"]))*(10.0 - np.minimum(5.0, np.abs(df["pGS"]-df["GS"])+np.abs(df["pGC"]-df["GC"])))
  else:
    df["total_points"]+=[point_scheme[0][1] if t==0 and gof=="3_full" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[1][1] if t==0 and gof=="2_diff" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[2][1] if t==0 and gof=="1_tendency" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[2][0] if t==1 and gof=="1_tendency" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[1][0] if t==1 and gof=="2_diff" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[0][0] if t==1 and gof=="3_full" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[2][2] if t==-1 and gof=="1_tendency" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[1][2] if t==-1 and gof=="2_diff" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
    df["total_points"]+=[point_scheme[0][2] if t==-1 and gof=="3_full" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]

  df = pd.concat([df, pd.get_dummies(df["gof"])], axis=1) # one-hot encoding of gof
  if "3_full" not in df: 
    df["3_full"]=0
  if "2_diff" not in df: 
    df["2_diff"]=0
  if "1_tendency" not in df: 
    df["1_tendency"]=0
    
  df["hit_goal"]=100*df["3_full"]
  df["hit_diff"]=100*(df["3_full"]+df["2_diff"])
  df["hit_tend"]=100*(df["3_full"]+df["2_diff"]+df["1_tendency"])
  
  df["sort_idx"]=[(x[0]-x[1])*(1+0.1*x[0])+0.01*x[0] for x in zip(df["pFTHG"],df["pFTAG"])]
  df = df.sort_values(by=('sort_idx'), ascending=True)

  def get_freq(cnt, lookup):
    return [cnt[x]*100.0/len(lookup) if x in cnt else 0 for x in lookup]
  
  df["goal_freq"] = get_freq(goal_cnt, df["pGoals"])
  df["gdiff_freq"] = get_freq(gdiff_cnt, df["pGDiff"])
  df["tend_freq"] = get_freq(tend_cnt, df["pTendency"])
  
  t5 = df.pivot_table(index = 'Team1', 
            aggfunc={'0_none':np.sum, '1_tendency':np.sum, '2_diff':np.sum, '3_full':np.sum,
                    'total_points':[np.sum, np.mean],
                    "hit_tend":np.mean, "hit_diff":np.mean, "hit_goal":np.mean},
                     margins=False, fill_value=0)
  t5.columns = ["None", "Tendency", "Diff", "Full", "Diff%", "Goal%", "Tendency%", "AvgPoints", "TotalPoints"]
  t5 = t5.sort_values(by=('AvgPoints'), ascending=False)
  print(t5)
  print(prefix)
  t2 = df.pivot_table(index = [df["sort_idx"], "pGoals"], 
           aggfunc={'0_none':np.sum, '1_tendency':np.sum, '2_diff':np.sum, '3_full':np.sum,
                    'total_points':[len, np.sum, np.mean],
                    "hit_goal":np.mean, "goal_freq":np.mean},
           margins=False, fill_value=0)
  
  if False:
    model_dir = "D:/Models/model_gen2_sky_1819"
    with open(model_dir+'/'+prefix[:-1]+'t2.dmp', 'wb') as fp:
      pickle.dump(t2, fp)
    with open(model_dir+'/'+prefix[:-1]+'df.dmp', 'wb') as fp:
      pickle.dump(df, fp)
    t2 = pickle.load(open(model_dir+'/'+prefix[:-1]+'t2.dmp', 'rb')) 
    df = pickle.load(open(model_dir+'/'+prefix[:-1]+'df.dmp', 'rb')) 
  
  
  def append_metrics(pred, act, scores):
    df2 = pd.DataFrame({"act":act, "pred":[p for p in pred]}) # convert pred to list in order to prevent unwanted index reordering
    df2["TP"] = df2.pred & df2.act
    df2["TN"] = ~df2.pred & ~df2.act
    df2["FP"] = df2.pred & ~df2.act
    df2["FN"] = ~df2.pred & df2.act
    df2 = df2.astype(int)
    df3 = df2.sum()
    df3["MCC"]=(df3.TP*df3.TN-df3.FP*df3.FN)/np.sqrt((df3.TP+df3.FP)*(df3.TP+df3.FN)*(df3.TN+df3.FP)*(df3.TN+df3.FN))
    df3["F1"]=2*df3.TP/(2*df3.TP+df3.FP+df3.FN)
    df3["ACC"]=(df3.TP+df3.TN)/(df3.TP+df3.TN+df3.FP+df3.FN)
    df3["Recall"]=df3.TP/(df3.TP+df3.FN)
    df3["Precision"]=df3.TP/(df3.TP+df3.FP)
    df3["name"]=r
    df3 = df3.T
    scores = scores.append(df3, ignore_index=True)
    return scores

  scores = pd.DataFrame()
  for r in t2.reset_index().pGoals:
    pred = r == df["pGoals"]
    act = [x if w=="Home" else x[::-1] for x,w in zip(df["act"], df["Where"])]
    act = pd.Series([r==x for x in act])
    scores = append_metrics(pred, act, scores)
  #print(scores)
  
  t2.reset_index(inplace=True)
  t2.columns = ['_'.join(col).strip() for col in t2.columns.values]
  t2 = t2.drop(['sort_idx_'], axis=1)
  t2 = t2.rename(columns={"pGoals_":"Goals",
                          "0_none_sum":"None",
                          "1_tendency_sum":"Tendency",
                          "1_tendency_sum":"Tendency",
                          "2_diff_sum":"Diff",
                          "3_full_sum":"Full",
                          "total_points_len":"Total",
                          "hit_goal_mean":"ActualRate",
                          "goal_freq_mean":"TargetRate",
                          "total_points_mean":"AvgPoints",
                          "total_points_sum":"TotalPoints",
                          })
  t2 = t2.assign(EffRate=t2.ActualRate-t2.TargetRate, Contribution=t2.TotalPoints/len(df))
  t2 = t2[['Goals', 'None', 'Tendency', 'Diff', 'Full', 'Total',
           'ActualRate','TargetRate','EffRate',
           'AvgPoints', 'TotalPoints', 'Contribution' ]]
  
  print()
  print(t2)
  
  avg_points = np.sum(t2["Contribution"])

  pie_chart_values = t2[["None", "Tendency", "Diff", "Full"]].sum()

  t3 = df.pivot_table(index = ["pGDiff"], 
           aggfunc={'0_none':np.sum, '1_tendency':np.sum, '2_diff':np.sum, '3_full':np.sum,
                    'total_points':[len, np.sum, np.mean],
                    "hit_diff":np.mean, "gdiff_freq":np.mean},
           margins=False, fill_value=0)
  
  t3.reset_index(inplace=True)
  t3.columns = ['_'.join(col).strip() for col in t3.columns.values]
  t3 = t3.rename(columns={"pGDiff_":"GDiff",
                          "0_none_sum":"None",
                          "1_tendency_sum":"Tendency",
                          "1_tendency_sum":"Tendency",
                          "2_diff_sum":"Diff",
                          "3_full_sum":"Full",
                          "total_points_len":"Total",
                          "hit_diff_mean":"ActualRate",
                          "gdiff_freq_mean":"TargetRate",
                          "total_points_mean":"AvgPoints",
                          "total_points_sum":"TotalPoints",
                          })
  t3 = t3.assign(EffRate=t3.ActualRate-t3.TargetRate, Contribution=t3.TotalPoints/len(df))
  t3 = t3[['GDiff', 'None', 'Tendency', 'Diff', 'Full', 'Total',
           'ActualRate','TargetRate','EffRate',
           'AvgPoints', 'TotalPoints', 'Contribution' ]]
  
  print()
  print(t3)

  for r in t3.reset_index().GDiff:
    pred = df["pGDiff"]==r
    act = [gs-gc if w=="Home" else gc-gs for gs,gc,w in zip(df["GS"], df["GC"], df["Where"])]
    act = pd.Series([r==x for x in act])
    scores = append_metrics(pred, act, scores)
  #print(scores)


  tend_cnt = Counter(df["pTendency"])
  tendency_values = [tend_cnt.get(-1), tend_cnt.get(0), tend_cnt.get(1)] 
  tendency_values  = [0 if v is None else v for v in tendency_values  ]
  
  print()

  t4 = df.pivot_table(index = ["pTendency"], 
           aggfunc={'0_none':np.sum, '1_tendency':np.sum, '2_diff':np.sum, '3_full':np.sum,
                    'total_points':[len, np.sum, np.mean],
                    "hit_tend":np.mean, "tend_freq":np.mean},
           margins=False, fill_value=0)
  
  t4.reset_index(inplace=True)
  t4.columns = ['_'.join(col).strip() for col in t4.columns.values]
  t4 = t4.rename(columns={"pTendency_":"Prediction",
                          "0_none_sum":"None",
                          "1_tendency_sum":"Tendency",
                          "1_tendency_sum":"Tendency",
                          "2_diff_sum":"Diff",
                          "3_full_sum":"Full",
                          "total_points_len":"Total",
                          "hit_tend_mean":"ActualRate",
                          "tend_freq_mean":"TargetRate",
                          "total_points_mean":"AvgPoints",
                          "total_points_sum":"TotalPoints",
                          })
  t4 = t4.assign(EffRate=t4.ActualRate-t4.TargetRate, Contribution=t4.TotalPoints/len(df))
  t4 = t4[['Prediction', 'None', 'Tendency', 'Diff', 'Full', 'Total',
           'ActualRate','TargetRate','EffRate',
           'AvgPoints', 'TotalPoints', 'Contribution' ]]
  t4["Prediction"]=["Draw" if p==0 else "Homewin" if p==1 else "Awaywin" for p in t4["Prediction"]]
  print(t4)

  for r in t4.reset_index().Prediction:
    pred = ["Draw" if p==0 else "Homewin" if p>0 else "Awaywin" for p in df["pGDiff"]]
    pred = pd.Series([p==r for p in pred])
    act = [gs-gc if w=="Home" else gc-gs for gs,gc,w in zip(df["GS"], df["GC"], df["Where"])]
    act = ["Draw" if p==0 else "Homewin" if p>0 else "Awaywin" for p in act]
    act = pd.Series([r==x for x in act])
    scores = append_metrics(pred, act, scores)
  
  scores = pd.concat([scores.name, 
             scores[['act', 'pred','TN', 'FN', 'FP', 'TP']].astype(int),
             scores[['Precision', 'Recall', 'F1', 'ACC', 'MCC']]], axis=1)
  print()
  print(scores)
  
  t1 = df.pivot_table(index = [df["sort_idx"], 'pGoals'], columns=['gof'], values=["Team1"], aggfunc=len, margins=False, fill_value=0)
  t1.columns = ["None", "Tendency", "Diff", "Full"][:len(t1.columns)]
  t1.index = t1.index.droplevel(level=0)
  t1.plot(kind='bar', stacked=True, ax=ax1)
  ax1.set_title("{}".format(prefix)).set_size(15)
  
  _,_,autotexts = ax2.pie(pie_chart_values, 
          labels=["None", "Tendency", "Diff", "Full"], autopct='%1.1f%%', startangle=90)
  for t in autotexts:
    t.set_color("white")
  ax2.set_title("{}: {:.04f} ({})".format(prefix, avg_points, dataset)).set_size(20)

  percentages = [pie_chart_values[0], # None
                 pie_chart_values[1]+pie_chart_values[2]+pie_chart_values[3], #Tendency
                 pie_chart_values[2]+pie_chart_values[3], #GDiff
                 pie_chart_values[3], #Full
                 ]    
  for t,p in zip(autotexts, percentages):
    t.set_text("{:.01f}%".format(100.0 * p / len(df)))
  
  y_pred = [np.sign(p1-p2) if w=="Home" else np.sign(p2-p1) for p1,p2,w in zip(df["pGS"],df["pGC"],df["Where"])]
  y_pred = ["Draw" if i==0 else "HomeWin" if i==1 else "AwayWin" for i in y_pred]
  y_test = [np.sign(p1-p2) if w=="Home" else np.sign(p2-p1) for p1,p2,w in zip(df["GS"],df["GC"],df["Where"])]
  y_test = ["Draw" if i==0 else "HomeWin" if i==1 else "AwayWin" for i in y_test]

  cnf_matrix = confusion_matrix(y_test, y_pred)
  
  pAtA, pDtA, pHtA, pAtD, pDtD, pHtD, pAtH, pDtH, pHtH = cnf_matrix.reshape([9])
  ax3.axis('equal')
  wedges, _, autotexts = ax3.pie([pHtA, pHtH, pHtD, pDtH, pDtD, pDtA, pAtD, pAtA, pAtH], 
                         labels=["", "Home", "", "", "Draw", "", "", "Away", ""], 
                         #colors=["blue", "blue", "blue", "green", "green", "green", "red", "red", "red"], 
                         colors=["white"]*9,
                         startangle=90, autopct='%1.1f%%', 
                         radius=1.0, pctdistance=0.75,
                         wedgeprops = {"alpha":1.0, "linewidth":3})

  true_colors = ["red", "blue", "green", "blue", "green", "red", "green", "red", "blue"]
  pred_colors = ["blue", "blue", "blue", "green", "green", "green", "red", "red", "red"]

  for t,c in zip(autotexts, true_colors):
    t.set_color(c)

  for w,c in zip(wedges, pred_colors):
    w.set_edgecolor(c)

#  wedges[0].set_edgecolor("red")
#  wedges[1].set_edgecolor("blue")
#  wedges[2].set_edgecolor("green")
#  wedges[3].set_edgecolor("blue")
#  wedges[4].set_edgecolor("green")
#  wedges[5].set_edgecolor("red")
#  wedges[6].set_edgecolor("green")
#  wedges[7].set_edgecolor("red")
#  wedges[8].set_edgecolor("blue")

  #wedges[1].set_alpha(0.6)
  #wedges[4].set_alpha(0.6)
  #wedges[7].set_alpha(0.6)

#  for w in [0,2,3,5,6,8]:
#    wedges[w].set_hatch('/')  
#    wedges[w].set_alpha(0.6)  
  
  wedges[1].set_color(colors.to_rgba("blue", 0.1))
  wedges[4].set_color(colors.to_rgba("green", 0.1))
  wedges[7].set_color(colors.to_rgba("red", 0.1))
  autotexts[1].set_color("white")
  autotexts[4].set_color("white")
  autotexts[7].set_color("white")

  ax3.set_title("Home: {:.1f}, Draw: {:.1f}, Away: {:.1f}".format(
      100.0 * tendency_values[2]/len(df),
      100.0 * tendency_values[1]/len(df),
      100.0 * tendency_values[0]/len(df)
    ))

  plt.show()
  plt.close()

  print()
  print("Points: {0:.4f}, Tendency: {1:.2f}, Diff: {2:.2f}, Full: {3:.2f},    Home: {4:.1f}, Draw: {5:.1f}, Away: {6:.1f}".format(
      avg_points,
      100.0 * (1-pie_chart_values[0]/len(df)),
      100.0 * (pie_chart_values[2]+pie_chart_values[3])/len(df),
      100.0 * pie_chart_values[3]/len(df),
      100.0 * tendency_values[2]/len(df),
      100.0 * tendency_values[1]/len(df),
      100.0 * tendency_values[0]/len(df)
    ))
  
  c_home = df["Where"]=="Home"
  c_win  = df['pGS'] > df['pGC']
  c_loss = df['pGS'] < df['pGC']
  c_draw = df['pGS'] == df['pGC']
  c_tendency = np.sign(df['pGS']- df['pGC']) == np.sign(df["GS"] - df["GC"]) 

  default_color = "blue"
  default_cmap=plt.cm.Blues
  if prefix=="cp" or prefix=="cp2":
    default_color = "darkolivegreen"
    default_cmap=plt.cm.Greens
  if prefix=="pg2" or prefix=="pgpt":
    default_color = "darkmagenta"
    default_cmap=plt.cm.Purples # RdPu # PuRd
 
  def createTitle(series1, series2):
    return "pearson: {:.4f}, spearman: {:.4f}".format(
        series1.corr(series2, method="pearson"), 
        series1.corr(series2, method="spearman") 
        )

  if prefix!="ens":
      df["offset"] = np.random.rand(len(df))
      df["offset"] = df["offset"]*0.8 - 0.4 
      fig, ax = plt.subplots(1,2,figsize=(12,4))
      ax[0].set_title(createTitle(df["est1"], df["GS"]))
      ax[0].scatter(df[~c_home]["est1"], df[~c_home]["GS"]+df[~c_home]["offset"], alpha=0.1, color="red")
      ax[0].scatter(df[c_home]["est1"], df[c_home]["GS"]+df[c_home]["offset"], alpha=0.1, color=default_color)
      ax[1].set_title(createTitle(df["est2"], df["GC"]))
      ax[1].scatter(df[~c_home]["est2"], df[~c_home]["GC"]+df[~c_home]["offset"], alpha=0.1, color="red")
      ax[1].scatter(df[c_home]["est2"], df[c_home]["GC"]+df[c_home]["offset"], alpha=0.1, color=default_color)
      plt.show()
      fig, ax = plt.subplots(1,2,figsize=(12,4))
      ax[0].set_title(createTitle(df["est1"]-df["est2"], df["GS"]-df["GC"]))
      ax[0].scatter(df[~c_home]["est1"]-df[~c_home]["est2"], df[~c_home]["GS"]-df[~c_home]["GC"]+df[~c_home]["offset"],alpha=0.1, color="red")
      ax[0].scatter(df[c_home]["est1"]-df[c_home]["est2"], df[c_home]["GS"]-df[c_home]["GC"]+df[c_home]["offset"],alpha=0.1, color=default_color)
    
    #  fig, ax = plt.subplots(1,2,figsize=(12,4))
    #  ax[0].scatter(est1.loc[c_home], df[c_home]["GS"], alpha=0.1, color="blue")
    #  ax[0].scatter(est1.loc[~c_home], df[~c_home]["GS"], alpha=0.1, color="red")
    #  ax[1].scatter(est2.loc[c_home], df[c_home]["GC"], alpha=0.1, color="blue")
    #  ax[1].scatter(est2.loc[~c_home], df[~c_home]["GC"], alpha=0.1, color="red")
    #  plt.show()
    #  fig, ax = plt.subplots(1,2,figsize=(12,4))
    #  ax[0].scatter(est_diff.loc[c_home], df[c_home]["GS"]-df[c_home]["GC"],alpha=0.1, color="blue")
    #  ax[0].scatter(est_diff.loc[~c_home], df[~c_home]["GS"]-df[~c_home]["GC"],alpha=0.1, color="red")
      est1 = df.est1
      est2 = df.est2
      def plotEstimates(cond1, cond2, color, alpha=0.1):
        ax[1].scatter(est1[cond1 & c_tendency], est2[cond1 & c_tendency],alpha=alpha,color=color, marker='o')
        ax[1].scatter(est1[cond1 & ~c_tendency], est2[cond1 & ~c_tendency],alpha=alpha,color=color, marker='x')
        ax[1].scatter(est2[cond2 & c_tendency], est1[cond2 & c_tendency],alpha=alpha,color=color, marker='o')
        ax[1].scatter(est2[cond2 & ~c_tendency], est1[cond2 & ~c_tendency],alpha=alpha,color=color, marker='x')
      
      plotEstimates(c_win & c_home, c_loss & ~c_home, "blue")
      plotEstimates(c_loss & c_home, c_win & ~c_home, "red")
      plotEstimates(c_draw & c_home, c_draw & ~c_home, "green", 0.3)
      ax[1].set_title("Points: {:.4f} ({:.2f}%), H: {:.1f}, D: {:.1f}, A: {:.1f}".format(
          np.sum(t2["Contribution"]),
          100.0 * (1-pie_chart_values[0]/len(df)),
          100.0 * tendency_values[2]/len(df),
          100.0 * tendency_values[1]/len(df),
          100.0 * tendency_values[0]/len(df)
        ))
      plt.show()
      plt.close()

  np.set_printoptions(precision=2)
  
  # Plot non-normalized confusion matrix
  fig, ax = plt.subplots(1,2,figsize=(10,4))
  plot_confusion_matrix(ax[0], cnf_matrix, classes=["AwayWin", "Draw", "HomeWin"],
                        title='Tendency', cmap=default_cmap)
  
  # Plot normalized confusion matrix
  plot_confusion_matrix(ax[1], cnf_matrix, classes=["AwayWin", "Draw", "HomeWin"],
                        normalize=True,
                        title='Tendency', cmap=default_cmap)
  plt.show()
  plt.close()

  y_pred = [(p1-p2) if w=="Home" else (p2-p1) for p1,p2,w in zip(df["pGS"],df["pGC"],df["Where"])]
  y_test = [(p1-p2) if w=="Home" else (p2-p1) for p1,p2,w in zip(df["GS"],df["GC"],df["Where"])]
  y_test = [min(3,y) for y in y_test]    
  y_test = [max(-3,y) for y in y_test]    
  y_pred = [min(3,y) for y in y_pred]    
  y_pred = [max(-3,y) for y in y_pred]    
  cnf_matrix = confusion_matrix(y_test, y_pred)
  np.set_printoptions(precision=2)
  
  # Plot non-normalized confusion matrix
  fig, ax = plt.subplots(1,2,figsize=(10,4))
  plot_confusion_matrix(ax[0], cnf_matrix, classes=np.unique(y_test).tolist(),
                        title='GoalDiff', cmap=default_cmap)
  
  # Plot normalized confusion matrix
  plot_confusion_matrix(ax[1], cnf_matrix, classes=np.unique(y_test).tolist(),
                        normalize=True,
                        title='GoalDiff', cmap=default_cmap)
  plt.show()
  plt.close()
  
  if prefix=="ens/":
    t6 = df.pivot_table(index = ["Strategy"], 
             aggfunc={'0_none':np.sum, '1_tendency':np.sum, '2_diff':np.sum, '3_full':np.sum,
                      'total_points':[len, np.sum, np.mean],
                      "hit_goal":np.mean, "goal_freq":np.mean},
             margins=False, fill_value=0)
    
    t6.reset_index(inplace=True)
    t6.columns = ['_'.join(col).strip() for col in t6.columns.values]
    t6 = t6.rename(columns={"Strategy_":"Strategy",
                            "0_none_sum":"None",
                            "1_tendency_sum":"Tendency",
                            "1_tendency_sum":"Tendency",
                            "2_diff_sum":"Diff",
                            "3_full_sum":"Full",
                            "total_points_len":"Total",
                            "hit_goal_mean":"ActualRate",
                            "goal_freq_mean":"TargetRate",
                            "total_points_mean":"AvgPoints",
                            "total_points_sum":"TotalPoints",
                            })
    t6 = t6.assign(EffRate=t6.ActualRate-t6.TargetRate, Contribution=t6.TotalPoints/len(df))
    t6 = t6[['Strategy', 'None', 'Tendency', 'Diff', 'Full', 'Total',
             'ActualRate','TargetRate','EffRate',
             'AvgPoints', 'TotalPoints', 'Contribution' ]]
    
    print()
    print(t6)
    
    plt.axis('equal')
    plt.pie(t6["Total"], labels=t6['Strategy'], autopct='%1.1f%%')
    plt.show()
    plt.close()


def prepare_label_fit(predictions, features, labels, team_onehot_encoder, label_column_names, skip_plotting=False, output_name="outputs_poisson"):                             
  features = features["match_input_layer"]
  features = features[:len(predictions)] # cut off features if not enough predictions are present
  labels = labels[:len(predictions)] # cut off labels if not enough predictions are present
  tn = len(team_onehot_encoder.classes_)
  df = pd.DataFrame()
  df['Team1']=team_onehot_encoder.inverse_transform(features[:, 4:4+tn])
  df['Team2']=team_onehot_encoder.inverse_transform(features[:, 4+tn:4+2*tn])
  df['Where']=['Home' if h==1 else 'Away' for h in features[:, 1]]

#  df = pd.DataFrame().from_csv("file:///C:/tmp/Football/models_multi_2017/train_outputs_poisson.csv")
#  tn=36
#  label_column_names = df.columns[4::2]
  #print(label_column_names)
  fig = None
  if not skip_plotting:
    fig, ax = plt.subplots(1+(len(label_column_names)//4), 4, figsize=(20,70))
  for i,col in enumerate(label_column_names):
    #print(i)
    df["p_"+col]=[np.exp(p[output_name][i]) for p in predictions]
    df[col]=labels[:,i]
    dfcorr = df[["p_"+col, col]]
    cor_p = dfcorr[col].corr(dfcorr["p_"+col], method='pearson') 
    cor_s = dfcorr[col].corr(dfcorr["p_"+col], method='spearman') 
#    print(col)
#    print(cor_p)
#    print(cor_s)
    if not skip_plotting:
      ax0 = ax[i//4,np.mod(i,4)]
      df.boxplot(column = "p_"+col, 
                 by=col, ax=ax0, fontsize=10, grid=False)
      ax0.set_title('cor_p={:.4f}, cor_s={:.4f}'.format(cor_p,cor_s))
  
  #plt.show()  
  #fig.savefig("C:/tmp/Football/models_multi_2017/train_outputs_poisson.pdf")
  return df, fig


def print_prediction_summary(predictions, ens_predictions=None):
  for pt in ['Pt']:
    print("Begin {} ---------------------------".format(pt))
    p1 = predictions[['Where', pt, 'Prefix']].copy()
    p1["Season"]=p1.index//612
    p2 = p1[['Season','Prefix',pt,'Where' ]]
    print("Season Home/Away")
    print(p2.pivot_table(index=['Season', "Where"], values=pt, aggfunc=np.mean, columns="Prefix", margins=True))
  
    print("Season Complete")
    print(p2.pivot_table(index=['Season'], values=pt, aggfunc=np.mean, columns="Prefix", margins=True))
  
    if ens_predictions is None: 
      return
    
    p4 = ens_predictions[[pt, 'Prefix', "Strategy"]].copy()
    p4["Season"]=p4.index//306
    p4 = p4[['Season','Prefix',pt,'Strategy' ]]
    print("Ensemble Strategy Mix")
    print(pd.crosstab(p4['Season'], columns=p4["Strategy"], normalize="index", margins=True))
    print("Ensemble Strategy Contribution")
    print(p4.pivot_table(index=['Season'], values=pt, aggfunc=[np.mean], columns=["Prefix", "Strategy"], margins=True))
    #print(p4.pivot_table(index=['Season'], values=pt, aggfunc=np.mean, columns="Prefix", margins=False))
    print("End {} ---------------------------".format(pt))

def plot_point_summary(results):
  data = results.loc[results.dataset.isin(["train","test"])]
  data = pd.pivot_table(data, index = "Prefix", values="Pt", columns="dataset")
  fig, ax = plt.subplots(figsize=(10,6))
  data.plot(kind='bar', stacked=False, ax=ax, 
            title='Points by strategy', 
            legend=True, table=False, use_index=True,
            fontsize=12, grid=True,
            ylim=(data.test.max()*0.7, 0.06+np.max(data[["train","test"]].max())))
  ax.axhline(np.max(data.test), color="red")
  ax.annotate("{:.04f}".format(data.test.max()), [np.argmax(list(data.test)), 0.03+data.test.max()], fontsize=15)
  ax.set_xlabel('Strategy')
  ax.set_ylabel("Points")
  fig.tight_layout()
  plt.show()
 
def plot_checkpoints(df, predictions):
  for prefix in themodel.prefix_list: 
    prefix_df = df.loc[df.Prefix==prefix[:-1]]
    s = random.sample(range(len(prefix_df)), 1)[0]
    print(prefix, s)
    #print({k:v.shape for k,v in predictions.items()})
    sample_preds = {k:v[s] for k,v in predictions.items()}
    plot_softprob(sample_preds, prefix_df["GS"][s], prefix_df["GC"][s], prefix_df["Time"][s]+" : "+prefix_df["Team1"][s]+" - "+prefix_df["Team2"][s]+" ("+prefix_df["Where"][s]+")", prefix=prefix)
    s = random.sample(range(len(prefix_df)), 1)[0]
    sample_preds = {k:v[s] for k,v in predictions.items()}
    plot_softprob(sample_preds, prefix_df["GS"][s], prefix_df["GC"][s], prefix_df["Time"][s]+" : "+prefix_df["Team1"][s]+" - "+prefix_df["Team2"][s]+" ("+prefix_df["Where"][s]+")", prefix=prefix)
    for dataset in ["test", "train"]:
      plot_predictions_3(df, prefix[:-1], dataset)

  print_prediction_summary(df.loc[df.dataset=="test"], None) #test_ens_predictions)
  print_prediction_summary(df.loc[df.dataset=="train"], None) #, train_ens_predictions)
  plot_point_summary(df)
    

def get_input_data(model_dir, train_data, test_data, data_dir):
  all_data, all_labels, all_features, teamnames, team_mapping = get_train_test_data(model_dir, train_data, test_data, data_dir)
  features_arrays, labels_array, team_onehot_encoder, label_column_names = build_features(all_data.copy(), all_labels, all_features, teamnames, team_mapping)
  data = (all_data, teamnames, features_arrays, labels_array, team_onehot_encoder, label_column_names)
  return data


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self, features, labels):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None
        self.feed_dict = {"alllabels:0":labels,
                          "alldata:0": features['match_input_layer'] }

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)
    
    def before_run(self, run_context):
#        a = run_context.original_args
#        if a.feed_dict is None:
#          a.feed_dict = self.feed_dict
#        else:
#          a.feed_dict.update(self.feed_dict)
        return tf.train.SessionRunArgs(run_context.original_args.fetches,
                                       self.feed_dict,
                                       run_context.original_args.options)

def get_input_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, 
                 data_index=[]):
  #assert features.shape[0] == labels.shape[0]

  iterator_initializer_hook = IteratorInitializerHook(features, labels)

  def train_inputs():
    features_placeholder={k:tf.placeholder(v.dtype,shape=[None]+[x for x in v.shape[1:]]) for k,v in features.items()}
    print("features_placeholder: ", features_placeholder)
    label_placeholder=tf.placeholder(labels.dtype,shape=[None]+[x for x in labels.shape[1:]])
    feed_dict = {features_placeholder[k]:v[data_index] for k,v in features.items()}
    feed_dict[label_placeholder]=labels[data_index]
    #dataset = tf.data.Dataset.from_tensors((features_placeholder, label_placeholder))
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, label_placeholder))
    if mode==tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=6000).batch(256).repeat()
    else:
      dataset = dataset.batch(len(labels[data_index])).repeat(1)
    #print(data_index)
    print("dataset: ", dataset)
    iterator = dataset.make_initializable_iterator()
    next_example, next_label = iterator.get_next()
    # Set runhook to initialize iterator
    print("feed_dict", feed_dict.keys())
    print("next_example", next_example)
    print("next_label", next_label)
    iterator_initializer_hook.iterator_initializer_func = \
        lambda sess: sess.run(
            iterator.initializer,
            feed_dict=feed_dict)
    # Return batched (features, labels)
    return next_example, next_label

  # Return function and hook
  return train_inputs, iterator_initializer_hook

def train_model(model_data, train_steps):
  model, features_arrays, labels_array, features_placeholder, train_idx, test_idx, pred_idx = model_data
  train_input_fn, train_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.TRAIN, data_index=train_idx)

  DEBUG =False
  if DEBUG:
    debug_hook = tf_debug.LocalCLIDebugHook(ui_type='readline', dump_root='C:/tmp/Football/debug_dump')
    debug_hook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    hooks = [debug_hook]
  else:
    hooks = []

  model.train(
        input_fn=train_input_fn,
        steps=train_steps,
        hooks=hooks+[train_iterator_hook])

def train_eval_model(model_data, train_steps):
  model, features_arrays, labels_array, features_placeholder, train_idx, test_idx, pred_idx = model_data

  latest = tf.train.latest_checkpoint(model.model_dir)
  if latest:
    latest = int(os.path.basename(str(latest)).split('-')[1])
  else:
    latest = 0
  
  train_input_fn, train_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.TRAIN, data_index=train_idx)
  testeval_input_fn, testeval_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.EVAL, data_index=test_idx )
  traineval_input_fn, traineval_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.EVAL, data_index=train_idx)
  #testpred_input_fn, testpred_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.PREDICT, data_index=test_idx+pred_idx)
  #trainpred_input_fn, trainpred_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.PREDICT, data_index=train_idx)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=latest+train_steps, hooks=[train_iterator_hook])
  eval_spec = tf.estimator.EvalSpec(input_fn=testeval_input_fn, steps=None, hooks=[testeval_iterator_hook], throttle_secs=30, start_delay_secs=10)

  tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


  
def find_checkpoints_in_scope(model_dir, checkpoints, use_swa):
  export_dir = model_dir 
#  if use_swa:
#    export_dir = export_dir + "/swa"
  print(export_dir)
  checkpoint_paths = tf.train.get_checkpoint_state(export_dir).all_model_checkpoint_paths
  global_steps = [int(os.path.basename(str(cp)).split('-')[1]) for cp in checkpoint_paths]
  cp_df_all = pd.DataFrame({"global_step":global_steps, "checkpoint":checkpoint_paths})
  cp_df_final = pd.DataFrame()
  for cp in checkpoints.split(","):
    fromto = cp.split(":")
    fromto = [ft.strip() for ft in fromto]
    fromto = [int(ft) if ft!="" else None for ft in fromto]
    if len(fromto)==1:
      # no separator
      r = fromto[0]
      if r is None:
        cp_df = cp_df_all
      elif r < 0:
        # slice
        cp_df = cp_df_all.iloc[slice(r, None)]
      else:
        cp_df = cp_df_all.loc[cp_df_all.global_step==r]
    elif len(fromto)==2:
      # range
      ffrom = fromto[0]
      tto = fromto[1]
      if ffrom is None:
        ffrom=0
      else:
        ffrom=int(ffrom)
      if tto is None:
        tto = 100000000
      else:
        tto=int(tto)+1
      cp_df = cp_df_all.loc[cp_df_all.global_step.between(ffrom, tto, inclusive=True)]
    else:
      raise("wrong number of colon characters in "+fromto)  
    cp_df_final = cp_df_final.append(cp_df)
  if len(cp_df_final)==0:
    print("No checkpoints selected in {} using filter \"{}\"".format(export_dir, checkpoints))
  return cp_df_final.sort_values("global_step")

def evaluate_checkpoints(model_data, checkpoints, use_swa, eval_stop=False):
  model, features_arrays, labels_array, features_placeholder, train_idx, test_idx, pred_idx = model_data
  #tf.reset_default_graph()
  est_spec = model.model_fn(features=features_placeholder, labels=labels_array, mode="eval", config = model.config)
  cps = find_checkpoints_in_scope(model.model_dir, checkpoints, use_swa)
  if len(cps)==0:
    return
  model_dir = os.path.dirname(cps.iloc[0].checkpoint)
  train_writer = tf.summary.FileWriter( model_dir+'/eval_train')
  test_writer = tf.summary.FileWriter( model_dir+'/eval_test')

  eval_metric_ops = est_spec.eval_metric_ops
  for key, value in eval_metric_ops.items():
    tf.summary.scalar(key, value[1])
  
  summary_op=tf.summary.merge_all()
  init_l = tf.local_variables_initializer() # take care of summary metrics initialization
  loss = est_spec.loss
  
  data_index = test_idx
  feed_dict = {features_placeholder[k] : v[data_index] for k,v in features_arrays.items() if k!='match_input_layer'}
  feed_dict[ "alldata:0"]=features_arrays['match_input_layer']
  feed_dict[ "alllabels:0"]=labels_array

  data_index = train_idx
  feed_dict_train = {features_placeholder[k] : v[data_index] for k,v in features_arrays.items() if k!='match_input_layer'}
  feed_dict_train[ "alldata:0"]=features_arrays['match_input_layer']
  feed_dict_train[ "alllabels:0"]=labels_array

  saver = tf.train.Saver()
  cps_done = []
  for sig in ('TERM', 'INT'):
      signal.signal(getattr(signal, 'SIG'+sig), quit);
  exit_ = Event()
  with tf.Session() as sess:
    while not exit_.is_set():
      if len(cps)>0:
        for cp, global_step in zip(cps.checkpoint, cps.global_step):
          print(cp)
          saver.restore(sess, cp)
          _, outputs = sess.run([init_l, loss], feed_dict=feed_dict)
          summary = sess.run(summary_op, feed_dict=feed_dict)
          test_writer.add_summary(summary, global_step)
          print("test", global_step, outputs)
          _, outputs = sess.run([init_l, loss], feed_dict=feed_dict_train)
          summary = sess.run(summary_op, feed_dict=feed_dict_train)
          train_writer.add_summary(summary, global_step)
          print("train", global_step, outputs)
        cps_done.extend(cps.global_step)
        #print("cps_done", cps_done)
      else:
        if eval_stop:
          return
        else:
          exit_.wait(10)
      cps2 = find_checkpoints_in_scope(model.model_dir, checkpoints, use_swa)
      #print("cps2", cps2)
      cps = cps2.loc[~cps2.global_step.isin(cps_done)]
      #print("cps", cps)
  train_writer.close()
  test_writer.close()        


def predict_checkpoints(model_data, cps, all_data, skip_plotting):
  model, features_arrays, labels_array, features_placeholder, train_idx, test_idx, pred_idx = model_data
  #tf.reset_default_graph()
  est_spec = model.model_fn(features=features_placeholder, labels=labels_array, mode="infer", config = model.config)
  if len(cps)==0:
    return
  model_dir = model.model_dir #os.path.dirname(cps.iloc[0].checkpoint)

  data_index = train_idx+test_idx+pred_idx
  print("len(data_index)", len(data_index))
  data_set = ["train"]*len(train_idx)+["test"]*len(test_idx)+["pred"]*len(pred_idx)
  team1 = all_data.Team1[data_index].tolist()
  team2 = all_data.Team2[data_index].tolist()
  datetimes = ((all_data.Date[data_index]*1000+734138).astype(int)\
    .apply(lambda x: datetime.fromordinal(x)) +\
    (all_data.Time[data_index]+15.5).astype(int).apply(lambda x: np.timedelta64(x, 'h'))  +\
    ((all_data.Time[data_index]*60+30).astype(int)%60).apply(lambda x: np.timedelta64(x, 'm')) \
     ).dt.strftime("%d.%m.%Y %H:%M").tolist()
  feed_dict = {features_placeholder[k] : v[data_index] for k,v in features_arrays.items() if k!='match_input_layer'}
  feed_dict[ "alldata:0"]=features_arrays['match_input_layer']
  feed_dict[ "alllabels:0"]=labels_array
  
  features_batch = features_arrays['match_input_layer'][data_index]
  labels_batch = labels_array[data_index]
  
  pred = est_spec.predictions
  #print(pred)  
  saver = tf.train.Saver()
  with tf.Session() as sess:
    for cp, global_step in zip(cps.checkpoint, cps.global_step):
      saver.restore(sess, cp)
      predictions = sess.run(pred, feed_dict=feed_dict)
      #print({k:v.shape for k,v in predictions.items()})      
#      print(predictions["sp/p_pred_12"][0])
      #pd.DataFrame(predictions["outputs_poisson"]).to_csv(model_dir+'/poisson_data_pred.csv')
      #pd.DataFrame(labels_batch).to_csv(model_dir+'/poisson_data_gt.csv')
      results = [enrich_predictions(predictions, features_batch, labels_batch, team1, team2, datetimes, prefix, data_set, global_step) for prefix in themodel.prefix_list ]
      results = pd.concat(results, sort=False)
      if not skip_plotting:
        plot_checkpoints(results, predictions)
      results["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      results = results[["Date", "Team1", "Team2", "Time", "act", "pred", "Where", "est1","est2","Pt", "Prefix", "Strategy", "win", "draw", "loss", "winPt", "drawPt", "lossPt", "dataset", "global_step", "score", "train", "test"]]
#      with open(model_dir+'/all_predictions_df.csv', 'a') as f:
#        results.to_csv(f, header=f.tell()==0, quoting=csv.QUOTE_NONNUMERIC, index=False, line_terminator='\n')
      new_results = results.loc[results.dataset=="pred"]  
      print(new_results)
      with open(model_dir+'/new_predictions_df.csv', 'a') as f:
        new_results.to_csv(f, header=f.tell()==0, quoting=csv.QUOTE_NONNUMERIC, index=False, line_terminator='\n')

def enrich_predictions(predictions, features, labels, team1, team2, datetimes, prefix, dataset, global_step):

#  print("features.shape", features.shape)
#  print("len(predictions)", len(predictions))
#  print("len(predictions[sp/p_pred_12])", len(predictions["sp/p_pred_12"]))
#  print("len(predictions[sp/p_pred_12])", len(predictions["sp/p_pred_12"]))
  if predictions is None:
    return []
  if len(predictions["sp/p_pred_12"])==0:
    return []
  
  #features = features[:len(predictions)] # cut off features if not enough predictions are present
  #labels = labels[:len(predictions)] # cut off labels if not enough predictions are present
  
  df = pd.DataFrame()  
  df["GS"] = labels[:,0].astype(np.int)
  df["GC"] = labels[:,1].astype(np.int)
#  print(len(df))
#  print(len(dataset))
#  print(len(predictions[prefix+"pred"][:,0]))
#  print(predictions[prefix+"pred"][:,0])
  if prefix=="cp1/":
    df["GS"] = labels[:,2].astype(np.int)
    df["GC"] = labels[:,3].astype(np.int)
    
  df['pGS'] = predictions[prefix+"pred"][:,0]
  df['pGC'] = predictions[prefix+"pred"][:,1]
#  df['pGS'] = [p[prefix+"pred"][0] for p in predictions]
#  df['pGC'] = [p[prefix+"pred"][1] for p in predictions]

  if prefix!="ens/":
#    est1 = pd.Series([p[prefix+"ev_goals_1"] for p in predictions], name="est1")
#    est2 = pd.Series([p[prefix+"ev_goals_2"] for p in predictions], name="est2")
    est1 = pd.Series(predictions[prefix+"ev_goals_1"], name="est1")
    est2 = pd.Series(predictions[prefix+"ev_goals_2"], name="est2")
    df['est1'] = est1
    df['est2'] = est2
    df['Strategy'] = ''
  else:
    strategy_list = themodel.ens_prefix_list # ["p1", "p2","p3","p4","p5","p7","sp","sm","p1pt", "p2pt", "p4pt", "sppt", "smpt"]
    strategy_list = [s+" home" for s in strategy_list] + [s+" away" for s in strategy_list]
    strategy_index = predictions[prefix+"selected_strategy"]
    df['Strategy'] = [strategy_list[ind] for ind in strategy_index]
    print(Counter(df['Strategy']))
    
  #print(team_onehot_encoder.classes_)
  df['Team1']=team1
  df['Team2']=team2
  df['Time']=datetimes
  df['Where']=['Home' if h==1 else 'Away' for h in features[:, 1]]
  df["act"]  = [str(gs)+':'+str(gc) for gs,gc in zip(df["GS"],df["GC"]) ]
  df["pred"] = [str(gs)+':'+str(gc) for gs,gc in zip(df["pGS"],df["pGC"]) ]
  print(df.shape)
  if prefix!="ens/":
    df["win"]=0.0
    df["loss"]=0.0
    df["draw"]=0.0
    df["winPt"]=0.0
    df["lossPt"]=0.0
    df["drawPt"]=0.0
    for i_gs in range(7):
      for i_gc in range(7):
        i = i_gs*7+i_gc
        if i_gs>i_gc:
          df["win"]+=predictions[prefix+"p_pred_12"][:,i] 
          df["winPt"]=pd.concat([df["winPt"], pd.Series(predictions[prefix+"ev_points"][:,i])], axis=1).max(axis=1)
        if i_gs<i_gc:
          df["loss"]+=predictions[prefix+"p_pred_12"][:,i]
          df["lossPt"]=pd.concat([df["lossPt"], pd.Series(predictions[prefix+"ev_points"][:,i])], axis=1).max(axis=1)
        if i_gs==i_gc:
          df["draw"]+=predictions[prefix+"p_pred_12"][:,i] 
          df["drawPt"]=pd.concat([df["drawPt"], pd.Series(predictions[prefix+"ev_points"][:,i])], axis=1).max(axis=1)
    df["win"]*=100.0
    df["loss"]*=100.0
    df["draw"]*=100.0
  df["dataset"]=dataset
  df["global_step"]=global_step
  print(df.shape)

  #tensor = tf.constant(df[[ "pGS", "pGC", "GS", "GC"]].as_matrix(), dtype = tf.int64)
  #is_home = tf.equal(features[:,2] , 1)
  with tf.Session(graph=CALC_GRAPH) as sess:
    feed_dict={pl_pGS: df[["pGS"]].values,
               pl_pGC: df[["pGC"]].values,
               pl_GS: df[["GS"]].values,
               pl_GC: df[["GC"]].values,
               pl_is_home: features[:,1:2],
               }
    df['Pt'] = sess.run(tf.cast(calc_points_tensor, tf.int8), feed_dict=feed_dict)
    
    df['Prefix'] = prefix[:-1]
  
  scores = df.groupby(["dataset", "Prefix", "global_step"]).Pt.mean()
  scores = scores.rename("score")
  scores_wide = pd.pivot_table(scores.to_frame(), index=["Prefix", "global_step"],
                     columns=['dataset'], aggfunc=np.mean)
  scores_wide.columns = scores_wide.columns.droplevel(level=0)
  print(scores_wide)
  df = df.join(scores, on = ["dataset", "Prefix", "global_step"])
  df = df.join(scores_wide, on = ["Prefix", "global_step"], rsuffix="score")
    
  def preparePrintData(prefix, df):
    if prefix!="ens/":
      return df[["dataset", "Team1", "Team2", "act", "pred", "Where", "est1","est2","Pt", "Prefix", "win", "draw", "loss", "winPt", "drawPt", "lossPt", "global_step", "score", "pred", "test", "train"]]
    else:
      return df[["dataset", "Team1", "Team2", "act", "pred", "Pt", "Prefix", "Strategy"]]
  
  #print(preparePrintData(prefix, df).tail(20))
  #print(df.Where.describe())
  print(df.shape)
  return df

  
def dispatch_main(target_distr, model_dir, train_steps, train_data, test_data, 
                   checkpoints, save_steps, data_dir, max_to_keep, 
                   evaluate_after_steps, skip_plotting, target_system, modes, use_swa, histograms):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
  
  global point_scheme 
  if target_system=="TCS":
    point_scheme = point_scheme_tcs
    themodel.point_scheme = point_scheme_tcs
  elif target_system=="Pistor":
    point_scheme = point_scheme_pistor
    themodel.point_scheme = point_scheme_pistor
  elif target_system=="Sky":
    point_scheme = point_scheme_sky
    themodel.point_scheme = point_scheme_sky
  elif target_system=="GoalDiff":
    point_scheme = point_scheme_goal_diff
    themodel.point_scheme = point_scheme_goal_diff
  else:
    raise Exception("Unknown point scheme")
    
  global_prepare()

  all_data, teamnames, features_arrays, labels_array, team_onehot_encoder, label_column_names = get_input_data(model_dir, train_data, test_data, data_dir)

#  train_idx = range(2*306*len(train_data))
#  test_idx = range(2*306*len(train_data), 2*306*len(train_data)+2*306*len(test_data))
#  print(train_idx)
#  print(test_idx)
  print(target_system)
  print(all_data.shape)
  print(labels_array.shape)
  print(labels_array.dtype)
  print([(k, v.shape, v.dtype) for k,v in features_arrays.items()])

#  print(feature_columns)
#  print(teamnames)
  train_idx = all_data.index[all_data['Train']].tolist()
  test_idx  = all_data.index[all_data['Test']].tolist()
  pred_idx  = all_data.index[all_data['Predict']].tolist()
  # skip first rounds if test data is placed in front
  if test_idx:
    if np.min(test_idx)==0 and np.min(train_idx)>45:
      test_idx = [t for t in test_idx if t>45]
  
#  train_idx = list(sum([(2*i, 2*i+1) for i in train_idx ], ()))
#  test_idx = list(sum([(2*i, 2*i+1) for i in test_idx ], ()))
#  pred_idx = list(sum([(2*i, 2*i+1) for i in pred_idx ], ()))

#  train_idx = [val for sublist in train_idx for val in sublist]
#  test_idx = [val for sublist in test_idx for val in sublist]
#  pred_idx = [val for sublist in pred_idx for val in sublist]
  if train_idx:
    print("Train index {}-{}".format(np.min(train_idx), np.max(train_idx)))
  if test_idx:
    print("Test index {}-{}".format(np.min(test_idx), np.max(test_idx)))
  print("Prediction index {}-{}".format(np.min(pred_idx), np.max(pred_idx)))
  

  tf.reset_default_graph()
  #print({k: v.shape for k, v in features_arrays.items()})
  my_feature_columns = [tf.feature_column.numeric_column(key=k, shape=v.shape[1:]) for k, v in features_arrays.items() if k != 'match_input_layer']
  #print(my_feature_columns)  
  
  features_placeholder={k:tf.placeholder(v.dtype,shape=[None]+[x for x in v.shape[1:]]) for k,v in features_arrays.items() if k!='match_input_layer'}
  features_placeholder["alllabels"]=tf.placeholder(labels_array.dtype, labels_array.shape)
  features_placeholder["alldata"]=tf.placeholder(features_arrays['match_input_layer'].dtype, features_arrays['match_input_layer'].shape)
  print(label_column_names)
  model = themodel.create_estimator(model_dir, label_column_names, my_feature_columns, features_arrays['match_input_layer'], labels_array, save_steps, evaluate_after_steps, max_to_keep, len(teamnames), use_swa, histograms, target_distr)
  
  model_data = (model, features_arrays, labels_array, features_placeholder, train_idx, test_idx, pred_idx)
  if modes == "static":
    static_probabilities(model_data)    
  elif "upgrade" in modes: # change to True if model structure has been changed
    utils.upgrade_estimator_model(model_dir, model, features=features_placeholder, labels=labels_array)
  elif modes == "train": 
    train_model(model_data, train_steps)
  elif modes == "train_eval": 
    train_eval_model(model_data, train_steps)
  elif modes == "eval": 
    evaluate_checkpoints(model_data, checkpoints, use_swa)
  elif modes == "eval_stop":
    evaluate_checkpoints(model_data, checkpoints, use_swa, eval_stop=True)
  elif modes == "predict": 
    cps = find_checkpoints_in_scope(model_dir, checkpoints, use_swa)
    predict_checkpoints(model_data, cps, all_data, skip_plotting)    
  
  return    
    

FLAGS = None

def main(_):
  
    #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/Mapping_Layer/WM", target_file_name="mapping.csv", all_tensor_names=False, all_tensors=False)
    #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel", target_file_name="rnn_candidate_kernel.csv", all_tensor_names=False, all_tensors=False)
    #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel", target_file_name="rnn_gates_kernel.csv", all_tensor_names=False, all_tensors=False)
    target_distr=[(5, 20, 35), 10, (15, 8, 2), (20, 20, 80)] # [(3:0, 3:1, 2:1), 1:1, (1:2, 1:3, 0:3), (0:0, 0:1/1:0, 0:2/2:0)]
    
    if FLAGS.target_system=="Pistor" or FLAGS.target_system=="GoalDiff":
        # Pistor
        target_distr={  "cp":[(5, 15, 30), 25, (15, 8, 2), (20, 20, 80)],
                        "sp":[(2, 20, 43), 15, (14, 5, 1), (20, 20, 80)],
        #                "pgpt":[(5, 20, 35), 25, (8, 5, 2), (20, 20, 80)],
                        "pg2":[(2, 10, 58), 5, (16, 3, 1), (20, 20, 80)],
                        "av":[(5, 15, 30), 20, (20, 8, 2), (20, 20, 80)],
                        }
    elif FLAGS.target_system=="Sky":
        # Sky
        target_distr={"cp":[(5, 20, 45), 1, (22, 5, 2), (10, 10, 85)],
                      "sp":[(6, 10, 58), 1, (19, 5, 2), (10, 10, 90)],
        #                "pgpt":[(5, 20, 35), 25, (8, 5, 2), (20, 20, 80)],
                      "pg2":[(2, 8, 70), 1, (13, 4, 2), (30, 30, 70)],
                      "av":[(7, 15, 48), 0, (23, 5, 2), (30, 30, 70)],
                    }
    else:
        raise("Wrong system")
    
    dispatch_main(target_distr, FLAGS.model_dir, FLAGS.train_steps,
                     FLAGS.train_data, FLAGS.test_data, FLAGS.checkpoints,
                     FLAGS.save_steps, FLAGS.data_dir, FLAGS.max_to_keep, 
                     FLAGS.evaluate_after_steps, FLAGS.skip_plotting, FLAGS.target_system, FLAGS.modes, FLAGS.swa, FLAGS.histograms)
  
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--skip_plotting", type=bool,
      #default=True, 
      default=False, 
      help="Print plots of predicted data"
  )
  parser.add_argument(
      "--histograms", type=bool,
      #default=True, 
      default=False, 
      help="create histogram data in summary files"
  )
  parser.add_argument(
      "--train_steps", type=int,
      default=200000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--save_steps", type=int,
      #default=200,
      default=500,
      help="Number of training steps between checkpoint files."
  )
  parser.add_argument(
      "--evaluate_after_steps", type=int,
      default=200,
      #default=300,
      help="Number of training steps after which to run evaluation. Should be a multiple of save_steps"
  )
  parser.add_argument(
      "--max_to_keep", type=int,
      default=500,
      help="Number of checkpoint files to keep."
  )
  parser.add_argument(
      "--train_data", type=str,
      default=["1314","1415", "1516", "1617", "1718", "1819", "1920"], #
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data", type=str,
      default=["0910", "1011", "1112", "1213"], #
      help="Path to the test data."
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      #default="c:/git/football/TF/data",
      default="d:/gitrepository/Football/football/TF/data",
      help="input data"
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="d:/Models/model_1920_pistor_long4",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--target_system",
      type=str,
      default="Pistor",
      #default="Sky",
      #default="TCS",
      #default="GoalDiff",
      help="Point system to optimize for"
  )
  parser.add_argument(
      "--swa", type=bool,
      #default=True,
      default=False,
      help="Run in Stochastic Weight Averaging mode."
  )
  parser.add_argument(
      "--modes",
      type=str,
      #default="static",
      default="train",
      #default="eval",
      #default="predict",
      #default="upgrade",
      #default="train_eval",
      #default="upgrade,train,eval,predict",
      help="What to do"
  )
  parser.add_argument(
      "--checkpoints", type=str,
      #default="19000:",
      #default="60000:92000", 
      default="-1",  # slice(-2, None)
      #default="100:",
      #default="",
      help="Range of checkpoints for evaluation / prediction. Format: "
  )
  FLAGS, unparsed = parser.parse_known_args()
  print([sys.argv[0]] + unparsed)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# Path('C:/tmp/Football/models/reset.txt').touch()

