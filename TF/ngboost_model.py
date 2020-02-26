# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:23:02 2020

@author: marti
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, cohen_kappa_score, balanced_accuracy_score, classification_report, log_loss
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ngboost.scores import CRPS, MLE
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import k_categorical
from ngboost.learners import default_tree_learner
from scipy.stats import poisson

import seaborn as sns

import argparse
#import shutil
import sys
import tempfile
import pickle
import csv
import pandas as pd
pd.set_option('expand_frame_repr', False)

import numpy as np

#np.set_printoptions(threshold=50)
from datetime import datetime
import os
from threading import Event

import matplotlib.pyplot as plt
import matplotlib.colors as colors

#from tensorflow.python.training.session_run_hook import SessionRunHook
#from tensorflow.contrib.layers import l2_regularizer

from collections import Counter
#from pathlib import Path
from sklearn.metrics import confusion_matrix
import random
import itertools

import shap
shap.initjs()  

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

SEQ_LENGTH = 10
TIMESERIES_COL = 'rawdata'


def get_train_test_data(model_dir, train_seasons, test_seasons, data_dir, useBWIN):
  print("train_seasons")
  print(train_seasons)
  print(type(train_seasons))
  print("test_seasons")
  print(test_seasons)
  print(type(test_seasons))
  print("useBWIN")
  print(useBWIN)
  print(type(useBWIN))

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
  
  feature_names = ['Time', 't1games', 't1dayssince', 't2dayssince', 't1dayssince_ema', 't2dayssince_ema', 'roundsleft', 't1promoted', 't2promoted', 't1points', 't2points', 't1rank', 't2rank', 't1rank6_attention', 't2rank6_attention', 't1rank16_attention', 't2rank16_attention', 't1cards_ema', 't2cards_ema', 'BW1', 'BW0', 'BW2', 'T1_CUM_T1_GFT', 'T2_CUM_T2_GFT', 'T1_CUM_T1_W_GFT', 'T2_CUM_T2_W_GFT', 'T1_CUM_T2_GFT', 'T2_CUM_T1_GFT', 'T1_CUM_T2_W_GFT', 'T2_CUM_T1_W_GFT', 'T12_CUM_T1_GFT', 'T12_CUM_T1_W_GFT', 'T21_CUM_T2_GFT', 'T21_CUM_T2_W_GFT', 'T12_CUM_T12_GFT', 'T12_CUM_T12_W_GFT', 'T1221_CUM_GFT', 'T1221_CUM_W_GFT', 'T1_CUM_T1_GHT', 'T2_CUM_T2_GHT', 'T1_CUM_T1_W_GHT', 'T2_CUM_T2_W_GHT', 'T1_CUM_T2_GHT', 'T2_CUM_T1_GHT', 'T1_CUM_T2_W_GHT', 'T2_CUM_T1_W_GHT', 'T12_CUM_T1_GHT', 'T12_CUM_T1_W_GHT', 'T21_CUM_T2_GHT', 'T21_CUM_T2_W_GHT', 'T12_CUM_T12_GHT', 'T12_CUM_T12_W_GHT', 'T1221_CUM_GHT', 'T1221_CUM_W_GHT', 'T1_CUM_T1_S', 'T2_CUM_T2_S', 'T1_CUM_T1_W_S', 'T2_CUM_T2_W_S', 'T1_CUM_T2_S', 'T2_CUM_T1_S', 'T1_CUM_T2_W_S', 'T2_CUM_T1_W_S', 'T12_CUM_T1_S', 'T12_CUM_T1_W_S', 'T21_CUM_T2_S', 'T21_CUM_T2_W_S', 'T12_CUM_T12_S', 'T12_CUM_T12_W_S', 'T1221_CUM_S', 'T1221_CUM_W_S', 'T1_CUM_T1_ST', 'T2_CUM_T2_ST', 'T1_CUM_T1_W_ST', 'T2_CUM_T2_W_ST', 'T1_CUM_T2_ST', 'T2_CUM_T1_ST', 'T1_CUM_T2_W_ST', 'T2_CUM_T1_W_ST', 'T12_CUM_T1_ST', 'T12_CUM_T1_W_ST', 'T21_CUM_T2_ST', 'T21_CUM_T2_W_ST', 'T12_CUM_T12_ST', 'T12_CUM_T12_W_ST', 'T1221_CUM_ST', 'T1221_CUM_W_ST', 'T1_CUM_T1_F', 'T2_CUM_T2_F', 'T1_CUM_T1_W_F', 'T2_CUM_T2_W_F', 'T1_CUM_T2_F', 'T2_CUM_T1_F', 'T1_CUM_T2_W_F', 'T2_CUM_T1_W_F', 'T12_CUM_T1_F', 'T12_CUM_T1_W_F', 'T21_CUM_T2_F', 'T21_CUM_T2_W_F', 'T12_CUM_T12_F', 'T12_CUM_T12_W_F', 'T1221_CUM_F', 'T1221_CUM_W_F', 'T1_CUM_T1_C', 'T2_CUM_T2_C', 'T1_CUM_T1_W_C', 'T2_CUM_T2_W_C', 'T1_CUM_T2_C', 'T2_CUM_T1_C', 'T1_CUM_T2_W_C', 'T2_CUM_T1_W_C', 'T12_CUM_T1_C', 'T12_CUM_T1_W_C', 'T21_CUM_T2_C', 'T21_CUM_T2_W_C', 'T12_CUM_T12_C', 'T12_CUM_T12_W_C', 'T1221_CUM_C', 'T1221_CUM_W_C', 'T1_CUM_T1_Y', 'T2_CUM_T2_Y', 'T1_CUM_T1_W_Y', 'T2_CUM_T2_W_Y', 'T1_CUM_T2_Y', 'T2_CUM_T1_Y', 'T1_CUM_T2_W_Y', 'T2_CUM_T1_W_Y', 'T12_CUM_T1_Y', 'T12_CUM_T1_W_Y', 'T21_CUM_T2_Y', 'T21_CUM_T2_W_Y', 'T12_CUM_T12_Y', 'T12_CUM_T12_W_Y', 'T1221_CUM_Y', 'T1221_CUM_W_Y', 'T1_CUM_T1_R', 'T2_CUM_T2_R', 'T1_CUM_T1_W_R', 'T2_CUM_T2_W_R', 'T1_CUM_T2_R', 'T2_CUM_T1_R', 'T1_CUM_T2_W_R', 'T2_CUM_T1_W_R', 'T12_CUM_T1_R', 'T12_CUM_T1_W_R', 'T21_CUM_T2_R', 'T21_CUM_T2_W_R', 'T12_CUM_T12_R', 'T12_CUM_T12_W_R', 'T1221_CUM_R', 'T1221_CUM_W_R', 'T1_CUM_T1_xG', 'T2_CUM_T2_xG', 'T1_CUM_T1_W_xG', 'T2_CUM_T2_W_xG', 'T1_CUM_T2_xG', 'T2_CUM_T1_xG', 'T1_CUM_T2_W_xG', 'T2_CUM_T1_W_xG', 'T12_CUM_T1_xG', 'T12_CUM_T1_W_xG', 'T21_CUM_T2_xG', 'T21_CUM_T2_W_xG', 'T12_CUM_T12_xG', 'T12_CUM_T12_W_xG', 'T1221_CUM_xG', 'T1221_CUM_W_xG', 'T1_CUM_T1_GH2', 'T2_CUM_T2_GH2', 'T1_CUM_T1_W_GH2', 'T2_CUM_T2_W_GH2', 'T1_CUM_T2_GH2', 'T2_CUM_T1_GH2', 'T1_CUM_T2_W_GH2', 'T2_CUM_T1_W_GH2', 'T12_CUM_T1_GH2', 'T12_CUM_T1_W_GH2', 'T21_CUM_T2_GH2', 'T21_CUM_T2_W_GH2', 'T12_CUM_T12_GH2', 'T12_CUM_T12_W_GH2', 'T1221_CUM_GH2', 'T1221_CUM_W_GH2', 'T1_CUM_T1_Win', 'T2_CUM_T2_Win', 'T1_CUM_T1_W_Win', 'T2_CUM_T2_W_Win', 'T1_CUM_T1_HTWin', 'T2_CUM_T2_HTWin', 'T1_CUM_T1_W_HTWin', 'T2_CUM_T2_W_HTWin', 'T1_CUM_T1_Loss', 'T2_CUM_T2_Loss', 'T1_CUM_T1_W_Loss', 'T2_CUM_T2_W_Loss', 'T1_CUM_T1_HTLoss', 'T2_CUM_T2_HTLoss', 'T1_CUM_T1_W_HTLoss', 'T2_CUM_T2_W_HTLoss', 'T1_CUM_T1_Draw', 'T2_CUM_T2_Draw', 'T1_CUM_T1_W_Draw', 'T2_CUM_T2_W_Draw', 'T1_CUM_T1_HTDraw', 'T2_CUM_T2_HTDraw', 'T1_CUM_T1_W_HTDraw', 'T2_CUM_T2_W_HTDraw']  
  feature_names += ["T1_spi", "T2_spi", "T1_imp", "T2_imp", "T1_GFTe", "T2_GFTe", "pp1", "pp0", "pp2"]  
  feature_names += ["T1_CUM_T1_GFTa", "T2_CUM_T2_GFTa", "T1_CUM_T1_W_GFTa", "T2_CUM_T2_W_GFTa", "T1_CUM_T2_GFTa", "T2_CUM_T1_GFTa", "T1_CUM_T2_W_GFTa", "T2_CUM_T1_W_GFTa", "T12_CUM_T1_GFTa", "T12_CUM_T1_W_GFTa", "T21_CUM_T2_GFTa", "T21_CUM_T2_W_GFTa", "T12_CUM_T12_GFTa", "T12_CUM_T12_W_GFTa", "T1221_CUM_GFTa", "T1221_CUM_W_GFTa", "T1_CUM_T1_xsg", "T2_CUM_T2_xsg", "T1_CUM_T1_W_xsg", "T2_CUM_T2_W_xsg", "T1_CUM_T2_xsg", "T2_CUM_T1_xsg", "T1_CUM_T2_W_xsg", "T2_CUM_T1_W_xsg", "T12_CUM_T1_xsg", "T12_CUM_T1_W_xsg", "T21_CUM_T2_xsg", "T21_CUM_T2_W_xsg", "T12_CUM_T12_xsg", "T12_CUM_T12_W_xsg", "T1221_CUM_xsg", "T1221_CUM_W_xsg", "T1_CUM_T1_xnsg", "T2_CUM_T2_xnsg", "T1_CUM_T1_W_xnsg", "T2_CUM_T2_W_xnsg", "T1_CUM_T2_xnsg", "T2_CUM_T1_xnsg", "T1_CUM_T2_W_xnsg", "T2_CUM_T1_W_xnsg", "T12_CUM_T1_xnsg", "T12_CUM_T1_W_xnsg", "T21_CUM_T2_xnsg", "T21_CUM_T2_W_xnsg", "T12_CUM_T12_xnsg", "T12_CUM_T12_W_xnsg", "T1221_CUM_xnsg", "T1221_CUM_W_xnsg", "T1_CUM_T1_spi", "T2_CUM_T2_spi", "T1_CUM_T1_W_spi", "T2_CUM_T2_W_spi", "T1_CUM_T2_spi", "T2_CUM_T1_spi", "T1_CUM_T2_W_spi", "T2_CUM_T1_W_spi", "T12_CUM_T1_spi", "T12_CUM_T1_W_spi", "T21_CUM_T2_spi", "T21_CUM_T2_W_spi", "T12_CUM_T12_spi", "T12_CUM_T12_W_spi", "T1221_CUM_spi", "T1221_CUM_W_spi", "T1_CUM_T1_imp", "T2_CUM_T2_imp", "T1_CUM_T1_W_imp", "T2_CUM_T2_W_imp", "T1_CUM_T2_imp", "T2_CUM_T1_imp", "T1_CUM_T2_W_imp", "T2_CUM_T1_W_imp", "T12_CUM_T1_imp", "T12_CUM_T1_W_imp", "T21_CUM_T2_imp", "T21_CUM_T2_W_imp", "T12_CUM_T12_imp", "T12_CUM_T12_W_imp", "T1221_CUM_imp", "T1221_CUM_W_imp", "T1_CUM_T1_GFTe", "T2_CUM_T2_GFTe", "T1_CUM_T1_W_GFTe", "T2_CUM_T2_W_GFTe", "T1_CUM_T2_GFTe", "T2_CUM_T1_GFTe", "T1_CUM_T2_W_GFTe", "T2_CUM_T1_W_GFTe", "T12_CUM_T1_GFTe", "T12_CUM_T1_W_GFTe", "T21_CUM_T2_GFTe", "T21_CUM_T2_W_GFTe", "T12_CUM_T12_GFTe", "T12_CUM_T12_W_GFTe", "T1221_CUM_GFTe", "T1221_CUM_W_GFTe"]
  
  all_features = all_data[feature_names].copy()
  all_data["Train"]=is_train.values&(~all_data["Predict"])
  all_data["Test"]=is_test.values&(~all_data["Predict"])

  if not useBWIN and 'BW1' in feature_names:  
    all_data["BW1"]=0.0
    all_data["BW0"]=0.0
    all_data["BW2"]=0.0
    all_features.loc[:,"BW1"]=0.0
    all_features.loc[:,"BW0"]=0.0
    all_features.loc[:,"BW2"]=0.0
    print("BWIN features set to zero")
    
#  all_labels["Train"]=is_train.values
#  all_labels["Test"]=is_test.values
#  all_labels["Predict"] = all_data["Predict"]
#
#  all_features["Train"]=is_train.values
#  all_features["Test"]=is_test.values
#  all_features["Predict"] = all_data["Predict"]
  teamnames = team_mapping.Teamname.tolist()
  print(all_data.iloc[1000])
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
  
  print("match_input_layer [:,23:26]")
  print(match_input_layer [:,23:26])
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
  print(label_column_names)
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
    ax[0].annotate("{:4.2f}".format(txt*100), (g1[i]-0.3,g2[i]-0.1))
  ax[1].scatter(g1, g2, s=spt*500, alpha=0.4,color='red')
  ax[1].scatter(gs, gc, s=spt[gs*7+gc]*500, alpha=0.7,color='red')
  for i, txt in enumerate(spt):
    ax[1].annotate("{:4.2f}".format(txt), (g1[i]-0.3,g2[i]-0.1))
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


def print_match_dates(X, team_onehot_encoder):
  features = X["newgame"]
  tn = len(team_onehot_encoder.classes_)
  match_dates = features[:,4+2*tn+0]*1000.0
  match_dates = [datetime.strftime(datetime.fromordinal(int(m)+734138), "%Y/%m/%d") for m in match_dates]
  match_dates = list(sorted(set(match_dates)))  
  print(len(features))
  print(match_dates)
  return match_dates

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
  for prefix in ["ngb"]: 
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
    

def get_input_data(model_dir, train_data, test_data, data_dir, useBWIN):
  all_data, all_labels, all_features, teamnames, team_mapping = get_train_test_data(model_dir, train_data, test_data, data_dir, useBWIN)
  print(all_data.iloc[1000])

  features_arrays, labels_array, team_onehot_encoder, label_column_names = build_features(all_data.copy(), all_labels, all_features, teamnames, team_mapping)
  print(features_arrays["match_input_layer"][:,23:26])
  data = (all_data, teamnames, features_arrays, labels_array, team_onehot_encoder, label_column_names)
  return data


def train_model(model_data, train_steps):
  features_arrays, labels_array, train_idx, test_idx, pred_idx = model_data
  
  #print(label_column_names)
  print(features_arrays["match_input_layer"].shape)
  print(labels_array.shape)
  
  
  X = features_arrays["match_input_layer"]
  Y = np.sign(labels_array[:,0]-labels_array[:,1]).astype(int)+1
  X_train = X[train_idx]
  X_test= X[test_idx]
  Y_train = Y[train_idx]
  Y_test= Y[test_idx]
  
  ngb = NGBClassifier(Dist=k_categorical(3),
                      n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=0.2) # tell ngboost that there are 3 possible outcomes
  print(X_train)
  print(Y_train)
  ngb.fit(X_train, Y_train) # Y should have only 3 values: {0,1,2}

  # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
  Y_pred = ngb.predict_proba(X_test)
  lb = None #["Loss", "Draw", "Win"]
  print(np.argmax(Y_pred, axis=1))
  print(accuracy_score(Y_test, np.argmax(Y_pred, axis=1)))
  print(confusion_matrix(Y_test, np.argmax(Y_pred, axis=1)))
  print(confusion_matrix(Y_test, np.argmax(Y_pred, axis=1), normalize='all'))
  print(classification_report(Y_test, np.argmax(Y_pred, axis=1)))
  
  Y_pred_train = ngb.predict_proba(X_train)
  print(accuracy_score(Y_train, np.argmax(Y_pred_train, axis=1)))
  print(confusion_matrix(Y_train, np.argmax(Y_pred_train, axis=1)))
  print(confusion_matrix(Y_train, np.argmax(Y_pred_train, axis=1), normalize='all'))
  print(classification_report(Y_train, np.argmax(Y_pred_train, axis=1)))
#        "match_input_layer": match_input_layer,
#      "gameindex": all_data.gameindex.values.astype(np.int16), 
#      "match_history_t1": mh1,
#      "match_history_t2": mh2,
#      "match_history_t12": mh12,
#      }, all_labels.values, team_mapping, label_column_names
#  with open('ngb.pickle', 'wb') as f:
#      pickle.dump(ngb, f)  
  
  feature_importance = pd.DataFrame({'feature':["F"+str(i) for i in range(X_train.shape[1])], 
                                   'importance':ngb.feature_importances_[0]})\
    .sort_values('importance',ascending=False).reset_index().drop(columns='index')
  fig, ax = plt.subplots()
  plt.title('Feature Importance Plot')
  sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance)

  train_input_fn, train_iterator_hook = get_input_fn(features_arrays, labels_array, mode=tf.estimator.ModeKeys.TRAIN, data_index=train_idx)

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
    pl_pGS = tf.placeholder(tf.int64, name="pl_pGS")
    pl_pGC = tf.placeholder(tf.int64, name="pl_pGC")
    pl_GS = tf.placeholder(tf.int64, name="pl_GS")
    pl_GC = tf.placeholder(tf.int64, name="pl_GC")
    pl_is_home = tf.placeholder(tf.bool, name="pl_is_home")
    calc_points_tensor = themodel.calc_points(pl_pGS,pl_pGC, pl_GS, pl_GC, pl_is_home)[0]
    calc_points_eval = (calc_points_tensor, pl_pGS, pl_pGC, pl_GS, pl_GC, pl_is_home)
    
    for cp, global_step in zip(cps.checkpoint, cps.global_step):
      saver.restore(sess, cp)
      predictions = sess.run(pred, feed_dict=feed_dict)
      #print({k:v.shape for k,v in predictions.items()})      
#      print(predictions["sp/p_pred_12"][0])
      #pd.DataFrame(predictions["outputs_poisson"]).to_csv(model_dir+'/poisson_data_pred.csv')
      #pd.DataFrame(labels_batch).to_csv(model_dir+'/poisson_data_gt.csv')
      results = [enrich_predictions(predictions, features_batch, labels_batch, team1, team2, datetimes, prefix, data_set, global_step, sess, calc_points_eval) for prefix in themodel.prefix_list ]
      results = pd.concat(results, sort=False)
      if not skip_plotting:
        plot_checkpoints(results, predictions)
      results["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      results = results[["Date", "Team1", "Team2", "Time", "act", "pred", "Where", "est1","est2","Pt", "Prefix", "Strategy", "win", "draw", "loss", "winPt", "drawPt", "lossPt", "dataset", "global_step", "score", "train", "test"]]
#      with open(model_dir+'/all_predictions_df.csv', 'a') as f:
#        results.to_csv(f, header=f.tell()==0, quoting=csv.QUOTE_NONNUMERIC, index=False, line_terminator='\n')
      new_results = results.loc[results.dataset=="pred"]  
      #print(new_results)
      with open(model_dir+'/new_predictions_df.csv', 'a') as f:
          print(model_dir+'/new_predictions_df.csv')
          new_results.to_csv(f, header=f.tell()==0, quoting=csv.QUOTE_NONNUMERIC, index=False, line_terminator='\n')

ind_win = [  i_gs*7+i_gc for i_gs in range(7) for i_gc in range(7) if i_gs>i_gc]
ind_loss = [  i_gs*7+i_gc for i_gs in range(7) for i_gc in range(7) if i_gs<i_gc]
ind_draw = [  i_gs*7+i_gc for i_gs in range(7) for i_gc in range(7) if i_gs==i_gc]

def enrich_predictions(predictions, features, labels, team1, team2, datetimes, prefix, dataset, global_step, sess=None, calc_points_eval = None):

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
    df['est1'] = predictions[prefix+"ev_goals_1"]
    df['est2'] = predictions[prefix+"ev_goals_2"]
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
  #print(df.shape)
  if prefix!="ens/":
      
    df["win"]=  np.sum(predictions[prefix+"p_pred_12"][:,ind_win],  axis=1)
    df["winPt"]=  np.max(predictions[prefix+"ev_points"][:,ind_win],  axis=1)
    df["draw"]=  np.sum(predictions[prefix+"p_pred_12"][:,ind_draw],  axis=1)
    df["drawPt"]=  np.max(predictions[prefix+"ev_points"][:,ind_draw],  axis=1)
    df["loss"]=  np.sum(predictions[prefix+"p_pred_12"][:,ind_loss],  axis=1)
    df["lossPt"]=  np.max(predictions[prefix+"ev_points"][:,ind_loss],  axis=1)
    df["win"]*=100.0
    df["loss"]*=100.0
    df["draw"]*=100.0
  df["dataset"]=dataset
  df["global_step"]=global_step
  #print(df.shape)

  #tensor = tf.constant(df[[ "pGS", "pGC", "GS", "GC"]].as_matrix(), dtype = tf.int64)
  #is_home = tf.equal(features[:,2] , 1)
  if sess is not None:
    calc_points_tensor, pl_pGS, pl_pGC, pl_GS, pl_GC, pl_is_home = calc_points_eval
    feed_dict={pl_pGS: df[["pGS"]].values,
               pl_pGC: df[["pGC"]].values,
               pl_GS: df[["GS"]].values,
               pl_GC: df[["GC"]].values,
               pl_is_home: features[:,1:2],
               }
    df['Pt'] = sess.run(tf.cast(calc_points_tensor, tf.int8), feed_dict=feed_dict)
    
    df['Prefix'] = prefix[:-1]
  else:
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
  #print(scores_wide)
  df = df.join(scores, on = ["dataset", "Prefix", "global_step"])
  df = df.join(scores_wide, on = ["Prefix", "global_step"], rsuffix="score")
    
  def preparePrintData(prefix, df):
    if prefix!="ens/":
      return df[["dataset", "Team1", "Team2", "act", "pred", "Where", "est1","est2","Pt", "Prefix", "win", "draw", "loss", "winPt", "drawPt", "lossPt", "global_step", "score", "pred", "test", "train"]]
    else:
      return df[["dataset", "Team1", "Team2", "act", "pred", "Pt", "Prefix", "Strategy"]]
  
  #print(preparePrintData(prefix, df).tail(20))
  #print(df.Where.describe())
  #print(df.shape)
  return df

  
def plot_probs(probs, softpoints, gs, gc, title=""):
  
  print("-----------------------------------------------------")
  print("{} / {}:{}".format(title, gs, gc))
  default_color = np.array([[ 0.12,  0.47,  0.71,  0.5  ]])

  sp = probs
  #print(sp)
  spt = softpoints
  gs = min(gs,6)
  gc = min(gc,6)
  pp = probs.reshape((7,7))
  margin_pred_prob1 = np.sum(pp, axis=1)
  margin_pred_prob2 = np.sum(pp, axis=0)
  margin_pred_expected1 = np.sum(np.arange(7)*margin_pred_prob1)
  margin_pred_expected2 = np.sum(np.arange(7)*margin_pred_prob2)
  margin_poisson_prob1 = [poisson.pmf(i, margin_pred_expected1) for i in range(7)]
  margin_poisson_prob2 = [poisson.pmf(i, margin_pred_expected2) for i in range(7)]
  # margin_pred_prob1 = pred[prefix+"p_marg_1"]
  # margin_poisson_prob1 = pred[prefix+"p_poisson_1"]
  # margin_pred_prob2 = pred[prefix+"p_marg_2"]
  # margin_poisson_prob2 = pred[prefix+"p_poisson_2"]
  # margin_pred_expected1 = pred[prefix+"ev_goals_1"] 
  # margin_pred_expected2 = pred[prefix+"ev_goals_2"] 
  g=[0.0,1.0,2.0,3.0,4.0,5.0,6.0]
  g1=[0.0]*7+[1.0]*7+[2.0]*7+[3.0]*7+[4.0]*7+[5.0]*7+[6.0]*7
  g2=g*7
  fig, ax = plt.subplots(1,3,figsize=(15,5))
  ax[0].scatter(g1, g2, s=sp*10000, alpha=0.4, color=default_color)
  ax[0].scatter(gs, gc, s=sp[gs*7+gc]*10000, alpha=0.7, color=default_color)
  for i, txt in enumerate(sp):
    ax[0].annotate("{:4.2f}".format(txt*100), (g1[i]-0.3,g2[i]-0.1))
  ax[1].scatter(g1, g2, s=spt*500, alpha=0.4,color='red')
  ax[1].scatter(gs, gc, s=spt[gs*7+gc]*500, alpha=0.7,color='red')
  for i, txt in enumerate(spt):
    ax[1].annotate("{:4.2f}".format(txt), (g1[i]-0.3,g2[i]-0.1))
  ax[0].set_title("p")
  ax[1].set_title("ev")
  ax[2].set_title(title)
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




def dispatch_main(target_distr, model_dir, train_steps, train_data, test_data, 
                   checkpoints, save_steps, data_dir, max_to_keep, 
                   reset_variables, skip_plotting, target_system, modes, use_swa, histograms, useBWIN):
  
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
  
  global point_scheme 
  if target_system=="TCS":
    point_scheme = point_scheme_tcs
  elif target_system=="Pistor":
    point_scheme = point_scheme_pistor
  elif target_system=="Sky":
    point_scheme = point_scheme_sky
  elif target_system=="GoalDiff":
    point_scheme = point_scheme_goal_diff
  else:
    raise Exception("Unknown point scheme")
    
  train_data = [s.strip() for s in train_data.strip('[]').split(',')]
  test_data = [s.strip() for s in test_data.strip('[]').split(',')]
  
  all_data, teamnames, features_arrays, labels_array, team_onehot_encoder, label_column_names = get_input_data(model_dir, train_data, test_data, data_dir, useBWIN)
  print(features_arrays["match_input_layer"][:,23:26])

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
  
  model_data = (features_arrays, labels_array, train_idx, test_idx, pred_idx)
#  if modes == "static":
#    static_probabilities(model_data)    
#  elif "upgrade" in modes: # change to True if model structure has been changed
#    utils.upgrade_estimator_model(model_dir, model, features=features_placeholder, labels=labels_array, reset_variables=reset_variables)
#  elif modes == "train": 
#    train_model(model_data, train_steps)
#  elif modes == "train_eval": 
#    train_eval_model(model_data, train_steps)
#  elif modes == "eval": 
#    evaluate_checkpoints(model_data, checkpoints, use_swa)
#  elif modes == "eval_stop":
#    evaluate_checkpoints(model_data, checkpoints, use_swa, eval_stop=True)
#  elif modes == "predict": 
#    cps = find_checkpoints_in_scope(model_dir, checkpoints, use_swa)
#    predict_checkpoints(model_data, cps, all_data, skip_plotting)    
  
  return    model_data, point_scheme, label_column_names
    

FLAGS = None

def main(_):
  
    #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/Mapping_Layer/WM", target_file_name="mapping.csv", all_tensor_names=False, all_tensors=False)
    #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel", target_file_name="rnn_candidate_kernel.csv", all_tensor_names=False, all_tensors=False)
    #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel", target_file_name="rnn_gates_kernel.csv", all_tensor_names=False, all_tensors=False)
    target_distr=[(5, 20, 35), 10, (15, 8, 2), (20, 20, 80)] # [(3:0, 3:1, 2:1), 1:1, (1:2, 1:3, 0:3), (0:0, 0:1/1:0, 0:2/2:0)]
    
    if FLAGS.target_system=="Pistor" :
        # Pistor
        target_distr={  "cp":[(2, 10, 48), 15, (16, 8, 1), (2, 2, 50)],
                        "sp":[(2, 20, 43), 15, (14, 5, 1), (20, 20, 80)],
        #                "pgpt":[(5, 20, 35), 25, (8, 5, 2), (20, 20, 80)],
                        "pg2":[(2, 12, 48), 5, (22, 5, 1), (20, 20, 80)],
                        "av":[(2, 10, 43), 15, (21, 8, 1), (2, 2, 50)],
                        }
    elif FLAGS.target_system=="Sky" or FLAGS.target_system=="GoalDiff":
        # Sky
        target_distr={"cp":[(5, 20, 45), 1, (22, 5, 2), (1, 1, 85)],
                      "sp":[(6, 10, 58), 1, (19, 5, 2), (10, 10, 90)],
        #                "pgpt":[(5, 20, 35), 25, (8, 5, 2), (20, 20, 80)],
                      "pg2":[(2, 13, 55), 1, (18, 9, 2), (30, 30, 70)],
                      "av":[(7, 15, 48), 0, (23, 5, 2), (3, 3, 70)],
                    }
    else:
        raise("Wrong system")
    
    return dispatch_main(target_distr, FLAGS.model_dir, FLAGS.train_steps,
                     FLAGS.train_data, FLAGS.test_data, FLAGS.checkpoints,
                     FLAGS.save_steps, FLAGS.data_dir, FLAGS.max_to_keep, 
                     FLAGS.reset_variables, FLAGS.skip_plotting, FLAGS.target_system, 
                     FLAGS.modes, FLAGS.swa, FLAGS.histograms, 
                     FLAGS.useBWIN)
  
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--skip_plotting", type=bool,
      default=True, 
      #default=False, 
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
      "--reset_variables", type=str, #nargs='+',
      default="cbsp",
      #default=300,
      help="List of variable names to be re-initialized during upgrade"
  )
  parser.add_argument(
      "--max_to_keep", type=int,
      default=500,
      help="Number of checkpoint files to keep."
  )
  parser.add_argument(
      "--train_data", type=str,
      #default="0910,1112,1314,1516,1718,1920", #
      #default="1314,1415,1516,1617,1718,1819,1920", #
      #default="0910,1011,1112,1213,1314,1415,1516,1617,1718", #
      default="1617,1718", #
      #default="1819,1920",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data", type=str,
      #default="1011,1213,1415,1617,1819", #
      #default="0910,1011,1112,1213", #
      #default="1617,1718", #
      default="1819,1920",
      help="Path to the test data."
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default="c:/git/football/TF/data",
      #default="d:/gitrepository/Football/football/TF/data",
      help="input data"
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="d:/Models/model_1920_pistor_16",
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
      "--useBWIN", type=bool,
      default=True,
      #default=False,
      help="Run in Stochastic Weight Averaging mode."
  )
  parser.add_argument(
      "--modes",
      type=str,
      #default="static",
      default="train",
      #default="eval_stop",
      #default="eval",
      #default="predict",
      #default="upgrade",
      #default="train_eval",
      #default="upgrade,train,eval,predict",
      help="What to do"
  )
  parser.add_argument(
      "--checkpoints", type=str,
      #default="560000:",
      #default="60000:92000", 
      default="-1",  # slice(-2, None)
      #default="100:",
      #default="",
      help="Range of checkpoints for evaluation / prediction. Format: "
  )
  FLAGS, unparsed = parser.parse_known_args()
  print([sys.argv[0]] + unparsed)
  print(FLAGS)
  model_data, point_scheme, label_column_names = main([sys.argv[0]] + unparsed)


# Path('C:/tmp/Football/models/reset.txt').touch()




  features_arrays, labels_array, train_idx, test_idx, pred_idx = model_data
  
  #print(label_column_names)
  print(features_arrays["match_input_layer"].shape)
  print(labels_array.shape)
  
  if point_scheme == point_scheme_pistor:
    point_matrix =  np.array([[3 if (i//7 == j//7) and (np.mod(i, 7)==np.mod(j, 7)) else 
                               2 if (i//7 - np.mod(i, 7) == j//7 - np.mod(j, 7)) else
                               1 if (np.sign(i//7 - np.mod(i, 7)) == np.sign(j//7 - np.mod(j, 7))) else
                               0  for i in range(49)]  for j in range(49)] )
  elif point_scheme == point_scheme_sky:
    point_matrix =  np.array([[5 if (i//7 == j//7) and (np.mod(i, 7)==np.mod(j, 7)) else 
                               2 if (np.sign(i//7 - np.mod(i, 7)) == np.sign(j//7 - np.mod(j, 7))) else
                               0  for i in range(49)]  for j in range(49)] )
  if point_scheme == point_scheme_goal_diff:
    point_matrix =  np.array([[np.maximum(5, 10 - np.abs(i//7 - j//7) - np.abs(np.mod(i, 7) - np.mod(j, 7))) if (np.sign(i//7 - np.mod(i, 7)) == np.sign(j//7 - np.mod(j, 7))) else
                               0  for i in range(49)]  for j in range(49)] )
  
  X = features_arrays["match_input_layer"]
  label_filter = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 46, 47, 48, 49, 50, 51, 52, 53]
  X = np.concatenate([X,
  np.mean(labels_array[features_arrays["match_history_t1"][:,-5:]][:,:,label_filter], axis=1),
  np.mean(labels_array[features_arrays["match_history_t2"][:,-5:]][:,:,label_filter], axis=1),
  np.mean(labels_array[features_arrays["match_history_t12"][:,-2:]][:,:,label_filter], axis=1)
  #labels_array[features_arrays["match_history_t2"][:,-5:]][:,:,label_filter].reshape((-1,5*len(label_filter))),
  #labels_array[features_arrays["match_history_t12"][:,-2:]][:,:,label_filter].reshape((-1,2*len(label_filter)))
  ], axis=1)

  #label_column_sublist = label_column_names[18:21]
  label_column_sublist = label_column_names[label_filter]
  #label_column_sublist = label_column_names
  label_column_names_extended = \
      ["T1_Hi5_"+x for x in label_column_sublist.tolist()] + \
      ["T2_Hi5_"+x for x in label_column_sublist.tolist()] + \
      ["T12_Hi2_"+x for x in label_column_sublist.tolist()] 
      
  feature_names =  ['t1histlen', 'where', 't2histlen','t12histlen']
  feature_names += ['Time', 't1games', 't1dayssince', 't2dayssince', 't1dayssince_ema', 't2dayssince_ema', 'roundsleft', 't1promoted', 't2promoted', 't1points', 't2points', 't1rank', 't2rank', 't1rank6_attention', 't2rank6_attention', 't1rank16_attention', 't2rank16_attention', 't1cards_ema', 't2cards_ema', 'BW1', 'BW0', 'BW2', 'T1_CUM_T1_GFT', 'T2_CUM_T2_GFT', 'T1_CUM_T1_W_GFT', 'T2_CUM_T2_W_GFT', 'T1_CUM_T2_GFT', 'T2_CUM_T1_GFT', 'T1_CUM_T2_W_GFT', 'T2_CUM_T1_W_GFT', 'T12_CUM_T1_GFT', 'T12_CUM_T1_W_GFT', 'T21_CUM_T2_GFT', 'T21_CUM_T2_W_GFT', 'T12_CUM_T12_GFT', 'T12_CUM_T12_W_GFT', 'T1221_CUM_GFT', 'T1221_CUM_W_GFT', 'T1_CUM_T1_GHT', 'T2_CUM_T2_GHT', 'T1_CUM_T1_W_GHT', 'T2_CUM_T2_W_GHT', 'T1_CUM_T2_GHT', 'T2_CUM_T1_GHT', 'T1_CUM_T2_W_GHT', 'T2_CUM_T1_W_GHT', 'T12_CUM_T1_GHT', 'T12_CUM_T1_W_GHT', 'T21_CUM_T2_GHT', 'T21_CUM_T2_W_GHT', 'T12_CUM_T12_GHT', 'T12_CUM_T12_W_GHT', 'T1221_CUM_GHT', 'T1221_CUM_W_GHT', 'T1_CUM_T1_S', 'T2_CUM_T2_S', 'T1_CUM_T1_W_S', 'T2_CUM_T2_W_S', 'T1_CUM_T2_S', 'T2_CUM_T1_S', 'T1_CUM_T2_W_S', 'T2_CUM_T1_W_S', 'T12_CUM_T1_S', 'T12_CUM_T1_W_S', 'T21_CUM_T2_S', 'T21_CUM_T2_W_S', 'T12_CUM_T12_S', 'T12_CUM_T12_W_S', 'T1221_CUM_S', 'T1221_CUM_W_S', 'T1_CUM_T1_ST', 'T2_CUM_T2_ST', 'T1_CUM_T1_W_ST', 'T2_CUM_T2_W_ST', 'T1_CUM_T2_ST', 'T2_CUM_T1_ST', 'T1_CUM_T2_W_ST', 'T2_CUM_T1_W_ST', 'T12_CUM_T1_ST', 'T12_CUM_T1_W_ST', 'T21_CUM_T2_ST', 'T21_CUM_T2_W_ST', 'T12_CUM_T12_ST', 'T12_CUM_T12_W_ST', 'T1221_CUM_ST', 'T1221_CUM_W_ST', 'T1_CUM_T1_F', 'T2_CUM_T2_F', 'T1_CUM_T1_W_F', 'T2_CUM_T2_W_F', 'T1_CUM_T2_F', 'T2_CUM_T1_F', 'T1_CUM_T2_W_F', 'T2_CUM_T1_W_F', 'T12_CUM_T1_F', 'T12_CUM_T1_W_F', 'T21_CUM_T2_F', 'T21_CUM_T2_W_F', 'T12_CUM_T12_F', 'T12_CUM_T12_W_F', 'T1221_CUM_F', 'T1221_CUM_W_F', 'T1_CUM_T1_C', 'T2_CUM_T2_C', 'T1_CUM_T1_W_C', 'T2_CUM_T2_W_C', 'T1_CUM_T2_C', 'T2_CUM_T1_C', 'T1_CUM_T2_W_C', 'T2_CUM_T1_W_C', 'T12_CUM_T1_C', 'T12_CUM_T1_W_C', 'T21_CUM_T2_C', 'T21_CUM_T2_W_C', 'T12_CUM_T12_C', 'T12_CUM_T12_W_C', 'T1221_CUM_C', 'T1221_CUM_W_C', 'T1_CUM_T1_Y', 'T2_CUM_T2_Y', 'T1_CUM_T1_W_Y', 'T2_CUM_T2_W_Y', 'T1_CUM_T2_Y', 'T2_CUM_T1_Y', 'T1_CUM_T2_W_Y', 'T2_CUM_T1_W_Y', 'T12_CUM_T1_Y', 'T12_CUM_T1_W_Y', 'T21_CUM_T2_Y', 'T21_CUM_T2_W_Y', 'T12_CUM_T12_Y', 'T12_CUM_T12_W_Y', 'T1221_CUM_Y', 'T1221_CUM_W_Y', 'T1_CUM_T1_R', 'T2_CUM_T2_R', 'T1_CUM_T1_W_R', 'T2_CUM_T2_W_R', 'T1_CUM_T2_R', 'T2_CUM_T1_R', 'T1_CUM_T2_W_R', 'T2_CUM_T1_W_R', 'T12_CUM_T1_R', 'T12_CUM_T1_W_R', 'T21_CUM_T2_R', 'T21_CUM_T2_W_R', 'T12_CUM_T12_R', 'T12_CUM_T12_W_R', 'T1221_CUM_R', 'T1221_CUM_W_R', 'T1_CUM_T1_xG', 'T2_CUM_T2_xG', 'T1_CUM_T1_W_xG', 'T2_CUM_T2_W_xG', 'T1_CUM_T2_xG', 'T2_CUM_T1_xG', 'T1_CUM_T2_W_xG', 'T2_CUM_T1_W_xG', 'T12_CUM_T1_xG', 'T12_CUM_T1_W_xG', 'T21_CUM_T2_xG', 'T21_CUM_T2_W_xG', 'T12_CUM_T12_xG', 'T12_CUM_T12_W_xG', 'T1221_CUM_xG', 'T1221_CUM_W_xG', 'T1_CUM_T1_GH2', 'T2_CUM_T2_GH2', 'T1_CUM_T1_W_GH2', 'T2_CUM_T2_W_GH2', 'T1_CUM_T2_GH2', 'T2_CUM_T1_GH2', 'T1_CUM_T2_W_GH2', 'T2_CUM_T1_W_GH2', 'T12_CUM_T1_GH2', 'T12_CUM_T1_W_GH2', 'T21_CUM_T2_GH2', 'T21_CUM_T2_W_GH2', 'T12_CUM_T12_GH2', 'T12_CUM_T12_W_GH2', 'T1221_CUM_GH2', 'T1221_CUM_W_GH2', 'T1_CUM_T1_Win', 'T2_CUM_T2_Win', 'T1_CUM_T1_W_Win', 'T2_CUM_T2_W_Win', 'T1_CUM_T1_HTWin', 'T2_CUM_T2_HTWin', 'T1_CUM_T1_W_HTWin', 'T2_CUM_T2_W_HTWin', 'T1_CUM_T1_Loss', 'T2_CUM_T2_Loss', 'T1_CUM_T1_W_Loss', 'T2_CUM_T2_W_Loss', 'T1_CUM_T1_HTLoss', 'T2_CUM_T2_HTLoss', 'T1_CUM_T1_W_HTLoss', 'T2_CUM_T2_W_HTLoss', 'T1_CUM_T1_Draw', 'T2_CUM_T2_Draw', 'T1_CUM_T1_W_Draw', 'T2_CUM_T2_W_Draw', 'T1_CUM_T1_HTDraw', 'T2_CUM_T2_HTDraw', 'T1_CUM_T1_W_HTDraw', 'T2_CUM_T2_W_HTDraw']  
  feature_names += ["T1_spi", "T2_spi", "T1_imp", "T2_imp", "T1_GFTe", "T2_GFTe", "pp1", "pp0", "pp2"]  
  feature_names += ["T1_CUM_T1_GFTa", "T2_CUM_T2_GFTa", "T1_CUM_T1_W_GFTa", "T2_CUM_T2_W_GFTa", "T1_CUM_T2_GFTa", "T2_CUM_T1_GFTa", "T1_CUM_T2_W_GFTa", "T2_CUM_T1_W_GFTa", "T12_CUM_T1_GFTa", "T12_CUM_T1_W_GFTa", "T21_CUM_T2_GFTa", "T21_CUM_T2_W_GFTa", "T12_CUM_T12_GFTa", "T12_CUM_T12_W_GFTa", "T1221_CUM_GFTa", "T1221_CUM_W_GFTa", "T1_CUM_T1_xsg", "T2_CUM_T2_xsg", "T1_CUM_T1_W_xsg", "T2_CUM_T2_W_xsg", "T1_CUM_T2_xsg", "T2_CUM_T1_xsg", "T1_CUM_T2_W_xsg", "T2_CUM_T1_W_xsg", "T12_CUM_T1_xsg", "T12_CUM_T1_W_xsg", "T21_CUM_T2_xsg", "T21_CUM_T2_W_xsg", "T12_CUM_T12_xsg", "T12_CUM_T12_W_xsg", "T1221_CUM_xsg", "T1221_CUM_W_xsg", "T1_CUM_T1_xnsg", "T2_CUM_T2_xnsg", "T1_CUM_T1_W_xnsg", "T2_CUM_T2_W_xnsg", "T1_CUM_T2_xnsg", "T2_CUM_T1_xnsg", "T1_CUM_T2_W_xnsg", "T2_CUM_T1_W_xnsg", "T12_CUM_T1_xnsg", "T12_CUM_T1_W_xnsg", "T21_CUM_T2_xnsg", "T21_CUM_T2_W_xnsg", "T12_CUM_T12_xnsg", "T12_CUM_T12_W_xnsg", "T1221_CUM_xnsg", "T1221_CUM_W_xnsg", "T1_CUM_T1_spi", "T2_CUM_T2_spi", "T1_CUM_T1_W_spi", "T2_CUM_T2_W_spi", "T1_CUM_T2_spi", "T2_CUM_T1_spi", "T1_CUM_T2_W_spi", "T2_CUM_T1_W_spi", "T12_CUM_T1_spi", "T12_CUM_T1_W_spi", "T21_CUM_T2_spi", "T21_CUM_T2_W_spi", "T12_CUM_T12_spi", "T12_CUM_T12_W_spi", "T1221_CUM_spi", "T1221_CUM_W_spi", "T1_CUM_T1_imp", "T2_CUM_T2_imp", "T1_CUM_T1_W_imp", "T2_CUM_T2_W_imp", "T1_CUM_T2_imp", "T2_CUM_T1_imp", "T1_CUM_T2_W_imp", "T2_CUM_T1_W_imp", "T12_CUM_T1_imp", "T12_CUM_T1_W_imp", "T21_CUM_T2_imp", "T21_CUM_T2_W_imp", "T12_CUM_T12_imp", "T12_CUM_T12_W_imp", "T1221_CUM_imp", "T1221_CUM_W_imp", "T1_CUM_T1_GFTe", "T2_CUM_T2_GFTe", "T1_CUM_T1_W_GFTe", "T2_CUM_T2_W_GFTe", "T1_CUM_T2_GFTe", "T2_CUM_T1_GFTe", "T1_CUM_T2_W_GFTe", "T2_CUM_T1_W_GFTe", "T12_CUM_T1_GFTe", "T12_CUM_T1_W_GFTe", "T21_CUM_T2_GFTe", "T21_CUM_T2_W_GFTe", "T12_CUM_T12_GFTe", "T12_CUM_T12_W_GFTe", "T1221_CUM_GFTe", "T1221_CUM_W_GFTe"]
  feature_names += label_column_names_extended 

  print(X.shape)          


#  train_idx = train_idx[::2]
#  test_idx = test_idx[::2]
#  pred_idx = pred_idx[::2]


  X_train = X[train_idx]
  X_test= X[test_idx]
  X_pred= X[pred_idx]
  bwin_index = [1, 23, 24, 25]
  X_train_bwin = np.take(X[train_idx], bwin_index, axis=1)
  X_test_bwin= np.take(X[test_idx], bwin_index, axis=1)
  X_pred_bwin= np.take(X[pred_idx], bwin_index, axis=1)

  #spi_index = [210, 211, 212, 213, 214, 215, 216, 217, 218]
  spi_index = [1, 214, 215, 216, 217, 218]
  X_train_spi= np.take(X[train_idx], spi_index, axis=1)
  X_test_spi= np.take(X[test_idx], spi_index, axis=1)
  X_pred_spi= np.take(X[pred_idx], spi_index, axis=1)


  
  Y1 = np.sign(labels_array[:,0]-labels_array[:,1]).astype(int)+1
  Y1_train = Y1[train_idx]
  Y1_test= Y1[test_idx]

  Y2 = (np.minimum(labels_array[:,0], 6)-np.minimum(labels_array[:,1], 6)).astype(int)+6
  Y2_train = Y2[train_idx]
  Y2_test= Y2[test_idx]

  Y3 = (np.minimum(labels_array[:,0], 6)*7+np.minimum(labels_array[:,1], 6)).astype(int)
  Y3_train = Y3[train_idx]
  Y3_test= Y3[test_idx]

  Y4 = np.minimum(labels_array[:,0], 6).astype(int)
  Y4_train = Y4[train_idx]
  Y4_test= Y4[test_idx]

  Y5 = np.minimum(labels_array[:,1], 6).astype(int)
  Y5_train = Y5[train_idx]
  Y5_test= Y5[test_idx]

  #Y6 = (np.minimum(labels_array[:,0], 6)+np.minimum(labels_array[:,1], 6)).astype(int)
  Y6 = np.minimum(np.sum(labels_array[:, 0:2], axis=1), 12).astype(int)
  Y6_train = Y6[train_idx]
  Y6_test= Y6[test_idx]

  Y8 = (np.minimum(labels_array[:,0], 6)-np.minimum(labels_array[:,1], 6)).astype(int)
  Y8 = np.minimum(Y8, 2)
  Y8 = np.maximum(Y8, -2)
  Y8 = Y8 + 2
  Y8_train = Y8[train_idx]
  Y8_test= Y8[test_idx]

  Y0 = set(range(49))-set(Y3_train) # dummy rows for missing categories
  X0 = X[np.random.choice(train_idx, len(Y0))]  
  X_train = np.concatenate((X_train, X0))
  X_train_bwin = np.concatenate((X_train_bwin, np.take(X0, bwin_index, axis=1)))
  X_train_spi = np.concatenate((X_train_spi, np.take(X0, spi_index, axis=1)))
  #X_train_eg = np.concatenate((X_train_eg, np.take(X0, eg_index, axis=1)))
  Y3_train = np.concatenate((Y3_train, np.array(list(Y0))))
  Y1_train = np.concatenate((Y1_train, np.sign(np.array(list(Y0))//7-np.mod(np.array(list(Y0)),7)).astype(int)+1))
  Y2_train = np.concatenate((Y2_train, (np.array(list(Y0))//7-np.mod(np.array(list(Y0)),7)).astype(int)+6))
  Y4_train = np.concatenate((Y4_train, (np.array(list(Y0))//7).astype(int)))
  Y5_train = np.concatenate((Y5_train, (np.mod(np.array(list(Y0)),7)).astype(int)))
  #Y6_train = np.concatenate((Y6_train, (np.array(list(Y0))//7+np.mod(np.array(list(Y0)),7)).astype(int)))
  Y6_train = np.concatenate((Y6_train, (np.array(list(Y0))//7 + np.mod(np.array(list(Y0)),7)).astype(int)))
  Y8_train = np.concatenate((Y8_train, np.maximum(-2, np.minimum(2, (np.array(list(Y0))//7-np.mod(np.array(list(Y0)),7)).astype(int)))+2))

  Y_spi_raw_train = X_train[:, [218, 217, 216]] # SPI
  Y_spi_raw_test = X_test[:, [218, 217, 216]] # SPI
  Y_spi_raw_new = X_pred[:, [218, 217, 216]] # SPI
  Y_bwin_raw_train = X_train[:, [25, 24, 23]] # BWIN
  Y_bwin_raw_test = X_test[:, [25, 24, 23]] # BWIN
  Y_bwin_raw_new = X_pred[:, [25, 24, 23]] # BWIN

  eg_index = [214, 215]  
  X_train_eg= np.take(X_train, eg_index, axis=1)
  X_test_eg= np.take(X_test, eg_index, axis=1)
  X_pred_eg= np.take(X_pred, eg_index, axis=1)

  def make_poisson(X):
      m = np.stack([ poisson.pmf(i//7, X[:,0])*poisson.pmf(np.mod(i, 7), X[:,1]) for i in range(49)], axis=1)
      m = m / np.sum(m, axis=1, keepdims=True)
      return m

  Y_eg_raw_train = make_poisson(X_train_eg)
  Y_eg_raw_test = make_poisson(X_test_eg)
  Y_eg_raw_new = make_poisson(X_pred_eg)
  
  def print_performance(labels, probs, name=""):
      return(pd.DataFrame.from_dict({"name":[name],
              "accuracy":[accuracy_score(labels, np.argmax(probs, axis=1))],
              "log_loss":[log_loss(labels, probs, labels=range(probs.shape[1]))],
              "balanced_accuracy_score":[balanced_accuracy_score(labels, np.argmax(probs, axis=1))],
              "cohen_kappa_score":[cohen_kappa_score(labels, np.argmax(probs, axis=1))]}))

  def model_progress(model, X_train, X_test, Y1_train, Y1_test):
      spd1_train = model.staged_pred_dist(X_train)[2:]
      spd1_test = model.staged_pred_dist(X_test)[2:]
      spd1_train
      fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(16,10))
      axs[0][0].set_title('Accuracy')
      axs[0][0].plot([accuracy_score(Y1_train, np.argmax(spd1_train[i].class_probs(), axis=1)) for i in range(len(spd1_train))])
      axs[0][0].plot([accuracy_score(Y1_test, np.argmax(spd1_test[i].class_probs(), axis=1)) for i in range(len(spd1_test))])
      axs[0][1].set_title('Balanced Accuracy Max. Prob')
      axs[0][1].plot([balanced_accuracy_score(Y1_train, np.argmax(spd1_train[i].class_probs(), axis=1))for i in range(len(spd1_train))])
      axs[0][1].plot([balanced_accuracy_score(Y1_test, np.argmax(spd1_test[i].class_probs(), axis=1))for i in range(len(spd1_test))])
      axs[1][0].set_title('Log Loss')
      axs[1][0].plot([log_loss(Y1_train, spd1_train[i].class_probs())for i in range(len(spd1_train))])
      axs[1][0].plot([log_loss(Y1_test, spd1_test[i].class_probs(), labels = range(len(set(Y1_train))))for i in range(len(spd1_test))])
      axs[1][1].set_title('Cohen Kappa Score')
      axs[1][1].plot([cohen_kappa_score(Y1_train, np.argmax(spd1_train[i].class_probs(), axis=1))for i in range(len(spd1_train))])
      axs[1][1].plot([cohen_kappa_score(Y1_test, np.argmax(spd1_test[i].class_probs(), axis=1), labels = range(len(set(Y1_train))))for i in range(len(spd1_test))])
      plt.show()
      return spd1_train, spd1_test
  lm_spi = LogisticRegression()
  lm_spi = LinearDiscriminantAnalysis()
  lm_spi.fit(X_train_spi, Y1_train)
  lm_spi.classes_
  #lm_spi.n_iter_
  lm_spi.coef_
  
  Y_spi_pred = lm_spi.predict_proba(X_test_spi)
  Y_spi_pred_train = lm_spi.predict_proba(X_train_spi)
  Y_spi_pred_new = lm_spi.predict_proba(X_pred_spi)

      
  lm_combined = LogisticRegression()
  lm_combined = LinearDiscriminantAnalysis()
  lm_combined .fit(np.concatenate([X_train_spi, X_train_bwin], axis=1), Y1_train)
  lm_combined .classes_
  #lm_spi.n_iter_
  lm_combined .coef_
  
  Y_combined_pred = lm_combined .predict_proba(np.concatenate([X_test_spi, X_test_bwin], axis=1))
  Y_combined_pred_train = lm_combined.predict_proba(np.concatenate([X_train_spi, X_train_bwin], axis=1))
  Y_combined_pred_new = lm_combined.predict_proba(np.concatenate([X_pred_spi, X_pred_bwin], axis=1))


#  ngb_spi = NGBClassifier(Dist=k_categorical(3),
#                      n_estimators=550, verbose_eval=10,
#                 learning_rate=0.0015,
#                 minibatch_frac=0.25)
#  
#  ngb_spi.fit(X_train_spi, Y1_train, X_val = X_test_spi, Y_val = Y1_test,
#                     #early_stopping_rounds = 150, 
##                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
##                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
#                     ) # Y should have only 3 values: {0,1,2}
#
#  spd_spi_train, spd_spi_test = model_progress(ngb_spi, X_train_spi, X_test_spi, Y1_train, Y1_test)  
#
#  Y_spi_pred = np.mean([x.class_probs() for x in spd_spi_test[50:500]], axis=0)
#  Y_spi_pred_train = np.mean([x.class_probs() for x in spd_spi_train[50:500]], axis=0)
#  Y_spi_pred_new = np.mean([x.class_probs() for x in ngb_spi.staged_pred_dist(X_pred_spi)[50:500]], axis=0)

  print(pd.concat([
          print_performance(Y1_test, Y_spi_pred, name="ngb_spi test"),
          print_performance(Y1_train, Y_spi_pred_train, name="ngb_spi train"),
          print_performance(Y1_test, Y_spi_raw_test, name="raw_spi test"),
          print_performance(Y1_train, Y_spi_raw_train, name="raw_spi train")]))

  print(confusion_matrix(Y1_test, np.argmax(Y_spi_pred, axis=1)))
  print(confusion_matrix(Y1_train, np.argmax(Y_spi_pred_train, axis=1)))
  
  import statsmodels.api as sm
  ngb_eg_1 = sm.GLM(Y4_train, X_train_eg, family=sm.families.Poisson()).fit()
  ngb_eg_1.summary()
  ngb_eg_2 = sm.GLM(Y5_train, X_train_eg, family=sm.families.Poisson()).fit()
  ngb_eg_2.summary()
  
#  ngb_eg_1= NGBRegressor(n_estimators=550, verbose_eval=10,
#                 learning_rate=0.005,
#                 minibatch_frac=0.25)
#  ngb_eg_1.fit(X_train_eg, Y4_train, X_val = X_test_eg, Y_val = Y4_test,
#                     #early_stopping_rounds = 150, 
##                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
##                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
#                     ) # Y should have only 3 values: {0,1,2}
#
#  ngb_eg_2= NGBRegressor(n_estimators=550, verbose_eval=10,
#                 learning_rate=0.005,
#                 minibatch_frac=0.25)
#  ngb_eg_2.fit(X_train_eg, Y5_train, X_val = X_test_eg, Y_val = Y5_test,
#                     #early_stopping_rounds = 150, 
##                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
##                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
#                     ) # Y should have only 3 values: {0,1,2}

  
  Y_eg_train = np.stack([ngb_eg_1.predict(X_train_eg), ngb_eg_2.predict(X_train_eg)], axis=1)
  Y_eg_test = np.stack([ngb_eg_1.predict(X_test_eg), ngb_eg_2.predict(X_test_eg)], axis=1)
  Y_eg_new = np.stack([ngb_eg_1.predict(X_pred_eg), ngb_eg_2.predict(X_pred_eg)], axis=1)

  Y3_eg_raw_train = make_poisson(Y_eg_train)
  Y3_eg_raw_test = make_poisson(Y_eg_test)
  Y3_eg_raw_new = make_poisson(Y_eg_new)



  lm_bwin = LogisticRegression()
  lm_bwin = LinearDiscriminantAnalysis()
  lm_bwin.fit(X_train_bwin, Y1_train)
  lm_bwin.classes_
  #lm_bwin.n_iter_
  lm_bwin.coef_
  
  Y_bwin_pred = lm_bwin.predict_proba(X_test_bwin)
  Y_bwin_pred_train = lm_bwin.predict_proba(X_train_bwin)
  Y_bwin_pred_new = lm_bwin.predict_proba(X_pred_bwin)
  
#  ngb_bwin = NGBClassifier(Dist=k_categorical(3),
#                      n_estimators=550, verbose_eval=10,
#                 learning_rate=0.005,
#                 minibatch_frac=0.25)
#  
#  ngb_bwin.fit(X_train_bwin, Y1_train, X_val = X_test_bwin, Y_val = Y1_test,
#                     #early_stopping_rounds = 150, 
##                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
##                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
#                     ) # Y should have only 3 values: {0,1,2}
#
#  spd_bwin_train, spd_bwin_test = model_progress(ngb_bwin, X_train_bwin, X_test_bwin, Y1_train, Y1_test)  
#
#  Y_bwin_pred = np.mean([x.class_probs() for x in spd_bwin_test[100:400]], axis=0)
#  Y_bwin_pred_train = np.mean([x.class_probs() for x in spd_bwin_train[100:400]], axis=0)
#  Y_bwin_pred_new = np.mean([x.class_probs() for x in ngb_bwin.staged_pred_dist(X_pred_bwin)[100:400]], axis=0)

  print(pd.concat([
          print_performance(Y1_test, Y_spi_pred, name="ngb_spi test"),
          print_performance(Y1_train, Y_spi_pred_train, name="ngb_spi train"),
          print_performance(Y1_test, Y_spi_raw_test, name="raw_spi test"),
          print_performance(Y1_train, Y_spi_raw_train, name="raw_spi train"),
          
          print_performance(Y1_test, Y_bwin_pred, name="ngb_bwin test"),
          print_performance(Y1_train, Y_bwin_pred_train, name="ngb_bwin train"),
          print_performance(Y1_test, Y_bwin_raw_test, name="raw_bwin test"),
          print_performance(Y1_train, Y_bwin_raw_train, name="raw_bwin train"),
          
          print_performance(Y1_test, Y_combined_pred, name="Y_combined_pred test"),
          print_performance(Y1_train, Y_combined_pred_train, name="Y_combined_pred_train"),
          
          print_performance(Y1_test, (Y_bwin_pred+Y_spi_pred)/2, name="ngb mixed test"),
          print_performance(Y1_train, (Y_bwin_pred_train+Y_spi_pred_train)/2, name="ngb mixed train"),
          print_performance(Y1_test, (Y_bwin_raw_test+Y_spi_raw_test)/2, name="raw mixed test"),
          print_performance(Y1_train, (Y_bwin_raw_train+Y_spi_raw_train)/2, name="raw mixed train")         
          ]))

  print(confusion_matrix(Y1_test, np.argmax(Y_bwin_pred, axis=1)))
  print(confusion_matrix(Y1_train, np.argmax(Y_bwin_pred_train, axis=1)))

  

#  print(pd.DataFrame({'feature':[feature_names[i] for i in bwin_index], 
#                              'draw':ngb_bwin.feature_importances_[0],
#                               'win':ngb_bwin.feature_importances_[1]}))
#
#  print(pd.DataFrame({'feature':[feature_names[i] for i in spi_index], 
#                              'draw':ngb_spi.feature_importances_[0],
#                               'win':ngb_spi.feature_importances_[1]}))

#  print(confusion_matrix(Y1_test, np.argmax(Y_bwin_pred, axis=1), normalize='all'))
#  print(classification_report(Y1_test, np.argmax(Y_bwin_pred, axis=1)))
#  print(np.mean(Y_bwin_pred, axis=0))    
#  print(Counter(np.argmax(Y_bwin_pred, axis=1)))
#  
#  #Y_bwin_pred_train = ngb_bwin.predict_proba(X_train_bwin)
#  print(confusion_matrix(Y1_train, np.argmax(Y_bwin_pred_train, axis=1)))
#  print(confusion_matrix(Y1_train, np.argmax(Y_bwin_pred_train, axis=1), normalize='all'))
#  print(classification_report(Y1_train, np.argmax(Y_bwin_pred_train, axis=1)))
#  print(np.mean(Y_bwin_pred_train, axis=0))    
#  print(Counter(np.argmax(Y_bwin_pred_train, axis=1)))
  

#  Y_bwin_pred = ngb_bwin.predict_proba(X_test_bwin)
#
#  Y_bwin_pred = X_test[:, [218, 217, 216]] # SPI
#  Y_bwin_pred_train = X_train[:, [218, 217, 216]] # SPI
#  Y_bwin_pred = X_test[:, [25, 24, 23]] # BWIN
#  Y_bwin_pred_train = X_train[:, [25, 24, 23]] # BWIN
#  Y_bwin_pred = (X_test[:, [218, 217, 216]]+X_test[:, [25, 24, 23]])/2 # SPI BWIN mixed
#  Y_bwin_pred_train = (X_train[:, [218, 217, 216]]+X_train[:, [25, 24, 23]])/2 # SPI BWIN mixed

  
#  plt.figure(figsize=[15,10])
#  plt.scatter(Y1_pred[0::2,0], Y1_pred[0::2,2], alpha=0.1, c=np.argmax(Y1_pred[0::2], axis=1), cmap='spring')
#  plt.scatter(Y_bwin_pred[0::2,0], Y_bwin_pred[0::2,2], alpha=0.1, c=np.argmax(Y_bwin_pred[0::2], axis=1), cmap='autumn')
#  plt.scatter(Y_spi_test[0::2,0], Y_spi_test[0::2,2], alpha=0.1, c=np.argmax(Y_spi_test[0::2], axis=1), cmap='summer')
#  plt.scatter(Y_bwin_test[0::2,0], Y_bwin_test[0::2,2], alpha=0.1, c=np.argmax(Y_bwin_test[0::2], axis=1), cmap='winter')
#  plt.plot()
#  

  plt.scatter(Y_bwin_pred[0::2,0], Y_bwin_pred[0::2,2], alpha=0.2, c=np.argmax(Y_bwin_pred[0::2], axis=1), cmap='prism')
  plt.scatter(Y_bwin_pred[1::2,0], Y_bwin_pred[1::2,2], alpha=0.2, c=np.argmax(Y_bwin_pred[1::2], axis=1), cmap='prism')

  plt.scatter(Y_spi_pred[0::2,0], Y_spi_pred[0::2,2], alpha=0.2, c=np.argmax(Y_spi_pred[0::2], axis=1), cmap='prism')
  plt.scatter(Y_spi_pred[1::2,0], Y_spi_pred[1::2,2], alpha=0.2, c=np.argmax(Y_spi_pred[1::2], axis=1), cmap='prism')
  
  #lin1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.2)
  lin1 = LogisticRegression(max_iter=200, C=0.00001)
  lin1.fit(X_train, Y1_train)
  lin1.coef_.shape
  Y1_lin_pred = lin1.predict_proba(X_test)
  Y1_lin_pred_train = lin1.predict_proba(X_train)
  
  lin1.max_iter

  X1_train = X_train.copy()
  for i in range(4, X1_train.shape[1]):
      np.random.shuffle(X1_train[0::2,i])
      np.random.shuffle(X1_train[1::2,i])
  
  X11_train = np.concatenate([X_train, X1_train], axis=0)
  Y11_train = np.concatenate([Y1_train, Y1_train], axis=0)
        
  X1_train[:,391]
  X_train[:,391]
  X1_train[:,1]
  X_train[:,1]
  
  ngb = NGBClassifier(Base=DecisionTreeRegressor(max_features=50,
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
          Dist=k_categorical(3),
                      n_estimators=2000, verbose_eval=10,
                 learning_rate=0.01,
                 minibatch_frac=0.75) # tell ngboost that there are 3 possible outcomes
  ngbmodel = ngb.fit(X11_train, Y11_train, X_val = X_test, Y_val = Y1_test,
                     #early_stopping_rounds = 150, 
#                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
#                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     ) # Y should have only 3 values: {0,1,2}
#  x0 = len(X_train)//2
#  ngbmodel = ngb.fit(X_train[2*np.concatenate([np.arange(x0)]*3).astype(int)], 
#                     Y1_train[2*np.concatenate([np.arange(x0)]+[np.random.randint(x0, size=x0)]+[np.random.randint(x0, size=x0)]).astype(int)], 
#                     X_val = X_test[::2], Y_val = Y1_test[::2])
  
  spd1_train, spd1_test = model_progress(ngb, X_train, X_test, Y1_train, Y1_test)  

  # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
  Y1_pred = ngb.predict_proba(X_test)
  Y1_pred = np.mean([x.class_probs() for x in spd1_test[50:]], axis=0)
  Y1_pred_train = np.mean([x.class_probs() for x in spd1_train[50:]], axis=0)
  Y1_pred_new = np.mean([x.class_probs() for x in ngb.staged_pred_dist(X_pred)[50:]], axis=0)

  print(pd.concat([
          print_performance(Y1_test, Y1_lin_pred, name="lin1 test"),
          print_performance(Y1_train, Y1_lin_pred_train, name="lin1 train"),
          print_performance(Y1_test, Y1_pred, name="ngb1 test"),
          print_performance(Y1_train, Y1_pred_train, name="ngb1 train"),
          print_performance(Y1_test, (Y_bwin_pred+Y_spi_pred+Y1_pred)/3, name="all mixed test"),
          print_performance(Y1_train, (Y_bwin_pred_train+Y_spi_pred_train+Y1_pred_train)/3, name="all mixed train")]))

  print(confusion_matrix(Y1_test, np.argmax(Y1_lin_pred, axis=1)))
  print(confusion_matrix(Y1_train, np.argmax(Y1_lin_pred_train, axis=1)))
  print(confusion_matrix(Y1_test, np.argmax(Y1_pred, axis=1)))
  print(confusion_matrix(Y1_train, np.argmax(Y1_pred_train, axis=1)))


  print(np.argmax(Y1_pred, axis=1))
  print(accuracy_score(Y1_test, np.argmax(Y1_pred, axis=1)))
  print(confusion_matrix(Y1_test, np.argmax(Y1_pred, axis=1)))
  print(confusion_matrix(Y1_test, np.argmax(Y1_pred, axis=1), normalize='all'))
  print(classification_report(Y1_test, np.argmax(Y1_pred, axis=1)))
  print(np.mean(Y1_pred, axis=0))    
  print(Counter(np.argmax(Y1_pred, axis=1)))
  
  Y1_pred_train = ngb.predict_proba(X_train)
  print(accuracy_score(Y1_train, np.argmax(Y1_pred_train, axis=1)))
  print(confusion_matrix(Y1_train, np.argmax(Y1_pred_train, axis=1)))
  print(confusion_matrix(Y1_train, np.argmax(Y1_pred_train, axis=1), normalize='all'))
  print(classification_report(Y1_train, np.argmax(Y1_pred_train, axis=1)))
  print(np.mean(Y1_pred_train, axis=0))    
  print(Counter(np.argmax(Y1_pred_train, axis=1)))
  
#        "match_input_layer": match_input_layer,
#      "gameindex": all_data.gameindex.values.astype(np.int16), 
#      "match_history_t1": mh1,
#      "match_history_t2": mh2,
#      "match_history_t12": mh12,
#      }, all_labels.values, team_mapping, label_column_names
#  with open('ngb.pickle', 'wb') as f:
#      pickle.dump(ngbmodel, f)  

  def plot_feature_importances(model, modelname):
      feature_importance_loc, feature_importance_scale = model.feature_importances_
    
      df_loc = pd.DataFrame({'feature':feature_names, 
                               'importance':feature_importance_loc})\
            .sort_values('importance',ascending=False)
      df_scale = pd.DataFrame({'feature':feature_names, 
                               'importance':feature_importance_scale})\
            .sort_values('importance',ascending=False)
      df_both = pd.DataFrame({'feature':feature_names, 
                              'location':feature_importance_loc+1e-4,
                               'scale':feature_importance_scale+1e-4})
            
      fig, ax = plt.subplots(figsize=(16,9))
      fig.suptitle("All Feature importance location vs scale\n"+modelname, fontsize=17)
      ax.set(xscale="log", yscale="log")  
      sns.scatterplot(x="location", y="scale", data=df_both)
      for line in range(0, len(df_both.feature)):
          ax.text(df_both.location[line], df_both.scale[line]*1.1, df_both.feature[line], horizontalalignment='center', size='x-small', color='black')
      plt.show()
     
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))
      fig.suptitle("Top 50 Feature importance plot for distribution parameters\n"+modelname, fontsize=17)
      sns.barplot(x='importance',y='feature',ax=ax1,data=df_loc[0:50], color="skyblue").set_title('loc param')
      ax1.tick_params(labelsize=8)
      ax2.tick_params(labelsize=8)
      sns.barplot(x='importance',y='feature',ax=ax2,data=df_scale[0:50], color="skyblue").set_title('scale param')
      plt.show()
      
  plot_feature_importances(ngb, "Model1")    

  #explainer = ngb.get_shap_tree_explainer(param_idx=0)
  
#  output=0
#  shap_data=X_pred[16:18]
#  shap_data=X_train
#  explainer = shap.TreeExplainer(ngb, model_output=output)
#  shap_values = explainer.shap_values(shap_data)
#  shap.summary_plot(shap_values, features=shap_data, feature_names=feature_names, title="ABC",
#                    max_display=30)
#
#  shap_interaction_values = explainer.shap_interaction_values(X_test, Y1_test)
#  #shap.decision_plot(explainer.expected_value, shap_values, features=shap_data, feature_names=feature_names, feature_display_range=None, link="logit")
#  #shap.force_plot(explainer.expected_value, shap_values, features=shap_data, feature_names=feature_names, link="logit", matplotlib=True, text_rotation=90)
#  #plt.show()
#    
#  explainer_draw = shap.TreeExplainer(ngb, model_output=0)
#  explainer_win = shap.TreeExplainer(ngb, model_output=1)
#
#  shap_data=X_pred
#  row_index = 11
#  shap.multioutput_decision_plot(
#      [explainer_draw.expected_value*0, explainer_draw.expected_value, explainer_win.expected_value], 
#      [explainer_draw.shap_values(shap_data)*0, explainer_draw.shap_values(shap_data), explainer_win.shap_values(shap_data)],
#                               row_index=row_index, 
#                               feature_names=feature_names, 
#                               #highlight=[np.argmax(heart_predictions[row_index])],
#                               legend_labels=['Loss', 'Draw', 'Win'],
#                               legend_location='lower right')
#
#  shap.decision_plot(explainer.expected_value, 
#                     shap_interaction_values[0:20], 
#                     feature_names=feature_names, link='logit',
#                     feature_order='hclust', feature_display_range=slice(None, -11, -1))  

  
  X2_train = X_train
  X2_train[:,4]=0 

  ngb2 = NGBClassifier(Base=DecisionTreeRegressor(max_features=30,
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
        Dist=k_categorical(13),
                      n_estimators=2000, verbose_eval=10,
                 learning_rate=0.005,
                 minibatch_frac=0.75) # tell ngboost that there are 3 possible outcomes
  ngbmodel2 = ngb2.fit(X11_train, np.concatenate([Y2_train, Y2_train], axis=0), 
                       X_val = X_test, Y_val = Y2_test,
                       early_stopping_rounds = 950, 
#                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
#                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     ) # Y should have only 3 values: {0,1,2}
  spd2_train, spd2_test = model_progress(ngb2, X2_train, X_test, Y2_train, Y2_test)  

  max_iter=ngb2.best_val_loss_itr
  max_iter=300
  # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
  Y2_pred = ngb2.predict_proba(X_test, max_iter=max_iter)
  Y2_pred = np.mean([x.class_probs() for x in spd2_test[100:300]], axis=0)
  Y2_pred_train = np.mean([x.class_probs() for x in spd2_train[100:300]], axis=0)
  Y2_pred_new = np.mean([x.class_probs() for x in ngb2.staged_pred_dist(X_pred)[100:]], axis=0)
  print(pd.concat([
          print_performance(Y2_test, Y2_pred, name="ngb2 test"),
          print_performance(Y2_train, Y2_pred_train, name="ngb2 train")]))
  print(accuracy_score(Y1_test-1, np.sign(np.argmax(Y2_pred, axis=1)-6)))
  print(confusion_matrix(Y1_test-1, np.sign(np.argmax(Y2_pred, axis=1)-6)))

  print(accuracy_score(Y1_train-1, np.sign(np.argmax(Y2_pred_train, axis=1)-6)))
  print(confusion_matrix(Y1_train-1, np.sign(np.argmax(Y2_pred_train, axis=1)-6)))

        
  lb = None #["Loss", "Draw", "Win"]
  print(np.argmax(Y2_pred, axis=1))
  print(accuracy_score(Y2_test, np.argmax(Y2_pred, axis=1)))
  print(confusion_matrix(Y2_test, np.argmax(Y2_pred, axis=1)))
  #print(confusion_matrix(Y_test, np.argmax(Y_pred, axis=1), normalize='all'))
  print(classification_report(Y2_test, np.argmax(Y2_pred, axis=1)))
  print(np.mean(Y2_pred, axis=0))    
  print(Counter(np.argmax(Y2_pred, axis=1)))

  Y2_pred_train = ngb2.predict_proba(X_train, max_iter=max_iter)
  print(accuracy_score(Y2_train, np.argmax(Y2_pred_train, axis=1)))
  print(confusion_matrix(Y2_train, np.argmax(Y2_pred_train, axis=1)))
  #print(confusion_matrix(Y_train, np.argmax(Y_pred_train, axis=1), normalize='all'))
  print(classification_report(Y2_train, np.argmax(Y2_pred_train, axis=1)))
  print(np.mean(Y2_pred_train, axis=0))    
  print(Counter(np.argmax(Y2_pred_train, axis=1)))

  feature_importance = pd.DataFrame({'feature':feature_names , 
                                   'importance':ngb2.feature_importances_[0]})\
    .sort_values('importance',ascending=False).reset_index().drop(columns='index')
  fig, ax = plt.subplots(figsize=(20, 20))
  plt.title('Feature Importance Plot')
  sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance.iloc[0:100])

  ngb6 = NGBClassifier(Base=DecisionTreeRegressor(max_features=30,
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=10,
                                         min_samples_split=20,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
        Dist=k_categorical(13),
                      n_estimators=2000, verbose_eval=10,
                 learning_rate=0.005,
                 minibatch_frac=0.5) # tell ngboost that there are 3 possible outcomes
#  X6_train = X_train
#  X6_train[:,4]=0 
  ngbmodel6 = ngb6.fit(X11_train, np.concatenate([Y6_train, Y6_train], axis=0), 
                       X_val = X_test, Y_val = Y6_test,
                       early_stopping_rounds = 950, 
#                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
#                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     ) # Y should have only 3 values: {0,1,6}
  spd6_train, spd6_test = model_progress(ngb6, X_train, X_test, Y6_train, Y6_test)  
  max_iter=ngb6.best_val_loss_itr
  # predicted probabilities of class 0, 1, and 6 (columns) for each observation (row)
  Y6_pred_new = np.mean([x.class_probs() for x in ngb6.staged_pred_dist(X_pred)[100:]], axis=0)
  #Y6_pred = ngb6.predict_proba(X_test, max_iter=max_iter)
  Y6_pred = np.mean([x.class_probs() for x in ngb6.staged_pred_dist(X_test)[100:]], axis=0)
  Y6_pred_train = np.mean([x.class_probs() for x in ngb6.staged_pred_dist(X_train)[100:]], axis=0)

  print(pd.concat([
          print_performance(Y6_test, Y6_pred, name="ngb6 test"),
          print_performance(Y6_train, Y6_pred_train, name="ngb6 train")]))

  lb = None #["Loss", "Draw", "Win"]
  print(np.argmax(Y6_pred, axis=1))
  print(accuracy_score(Y6_test, np.argmax(Y6_pred, axis=1)))
  print(confusion_matrix(Y6_test, np.argmax(Y6_pred, axis=1)))
  #print(confusion_matrix(Y_test, np.argmax(Y_pred, axis=1), normalize='all'))
  print(classification_report(Y6_test, np.argmax(Y6_pred, axis=1)))
  print(np.mean(Y6_pred, axis=0))    
  print(Counter(np.argmax(Y6_pred, axis=1)))

  #Y6_pred_train = ngb6.predict_proba(X_train, max_iter=max_iter)
  Y6_pred_train = np.mean([x.class_probs() for x in ngb6.staged_pred_dist(X_train)[100:]], axis=0)
  print(accuracy_score(Y6_train, np.argmax(Y6_pred_train, axis=1)))
  print(confusion_matrix(Y6_train, np.argmax(Y6_pred_train, axis=1)))
  #print(confusion_matrix(Y_train, np.argmax(Y_pred_train, axis=1), normalize='all'))
  print(classification_report(Y6_train, np.argmax(Y6_pred_train, axis=1)))
  print(np.mean(Y6_pred_train, axis=0))    
  print(Counter(np.argmax(Y6_pred_train, axis=1)))

  feature_importance = pd.DataFrame({'feature':feature_names , 
                                   'importance':ngb6.feature_importances_[0]})\
    .sort_values('importance',ascending=False).reset_index().drop(columns='index')
  fig, ax = plt.subplots(figsize=(20, 20))
  plt.title('Feature Importance Plot')
  sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance.iloc[0:100])


  ngb8 = NGBClassifier(Base=DecisionTreeRegressor(max_features=30,
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
        Dist=k_categorical(5),
                      n_estimators=2000, verbose_eval=10,
                 learning_rate=0.005,
                 minibatch_frac=0.5) # tell ngboost that there are 3 possible outcomes
  ngbmodel8 = ngb8.fit(X11_train, np.concatenate([Y8_train]*2, axis=0),
                       X_val = X_test, Y_val = Y8_test,
                       early_stopping_rounds = 950, 
#                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
#                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     ) # Y should have only 3 values: {0,1,2}
  spd8_train, spd8_test = model_progress(ngb8, X_train, X_test, Y8_train, Y8_test)  
  max_iter=ngb8.best_val_loss_itr
  # predicted probabilities of class 0, 1, and 8 (columns) for each observation (row)
  Y8_pred_new = np.mean([x.class_probs() for x in ngb8.staged_pred_dist(X_pred)[100:]], axis=0)
  Y8_pred = ngb8.predict_proba(X_test, max_iter=max_iter)
  Y8_pred = np.mean([x.class_probs() for x in ngb8.staged_pred_dist(X_test)[100:]], axis=0)
  Y8_pred_train = np.mean([x.class_probs() for x in ngb8.staged_pred_dist(X_train)[100:]], axis=0)
  print(pd.concat([
          print_performance(Y8_test, Y8_pred, name="ngb8 test"),
          print_performance(Y8_train, Y8_pred_train, name="ngb8 train")]))
  
  
  lb = None #["Loss", "Draw", "Win"]
  print(np.argmax(Y8_pred, axis=1))
  print(accuracy_score(Y8_test, np.argmax(Y8_pred, axis=1)))
  print(confusion_matrix(Y8_test, np.argmax(Y8_pred, axis=1)))
  #print(confusion_matrix(Y_test, np.argmax(Y_pred, axis=1), normalize='all'))
  print(classification_report(Y8_test, np.argmax(Y8_pred, axis=1)))
  print(np.mean(Y8_pred, axis=0))    
  print(Counter(np.argmax(Y8_pred, axis=1)))

  Y8_pred_train = ngb8.predict_proba(X_train, max_iter=max_iter)
  print(accuracy_score(Y8_train, np.argmax(Y8_pred_train, axis=1)))
  print(confusion_matrix(Y8_train, np.argmax(Y8_pred_train, axis=1)))
  #print(confusion_matrix(Y_train, np.argmax(Y_pred_train, axis=1), normalize='all'))
  print(classification_report(Y8_train, np.argmax(Y8_pred_train, axis=1)))
  print(np.mean(Y8_pred_train, axis=0))    
  print(Counter(np.argmax(Y8_pred_train, axis=1)))

  ngb4 = NGBClassifier(Base=DecisionTreeRegressor(max_features=30,# 0.03,
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
          Dist=k_categorical(7),
                      n_estimators=2000, verbose_eval=10,
                 learning_rate=0.005,
                 minibatch_frac=0.5) # tell ngboost that there are 3 possible outcomes
  ngbmodel = ngb4.fit(X11_train, np.concatenate([Y4_train, Y4_train], axis=0), X_val = X_test, Y_val = Y4_test,
                     early_stopping_rounds = 350, 
#                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
#                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     ) # Y should have only 3 values: {0,1,2}
  spd4_train, spd4_test = model_progress(ngb4, X_train, X_test, Y4_train, Y4_test)  
  max_iter = 300
  # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
  #Y4_pred = ngb4.predict_proba(X_test)
  Y4_pred_new = np.mean([x.class_probs() for x in ngb4.staged_pred_dist(X_pred)[100:]], axis=0)
  Y4_pred = np.mean([x.class_probs() for x in ngb4.staged_pred_dist(X_test)[100:]], axis=0)
  Y4_pred_train = np.mean([x.class_probs() for x in ngb4.staged_pred_dist(X_train)[100:]], axis=0)
  print(pd.concat([
          print_performance(Y4_test, Y4_pred, name="ngb4 test"),
          print_performance(Y4_train, Y4_pred_train, name="ngb4 train")]))

  #Y4_pred = spd4_test[max_iter].class_probs() 
  print(np.argmax(Y4_pred, axis=1))
  print(accuracy_score(Y4_test, np.argmax(Y4_pred, axis=1)))
  print(confusion_matrix(Y4_test, np.argmax(Y4_pred, axis=1)))
  print(confusion_matrix(Y4_test, np.argmax(Y4_pred, axis=1), normalize='all'))
  print(classification_report(Y4_test, np.argmax(Y4_pred, axis=1)))
  print(np.mean(Y4_pred, axis=0))    
  print(Counter(np.argmax(Y4_pred, axis=1)))
  
  #Y4_pred_train = ngb4.predict_proba(X_train)
#  Y4_pred_train = spd4_train[max_iter].class_probs() 
  print(accuracy_score(Y4_train, np.argmax(Y4_pred_train, axis=1)))
  print(confusion_matrix(Y4_train, np.argmax(Y4_pred_train, axis=1)))
  print(confusion_matrix(Y4_train, np.argmax(Y4_pred_train, axis=1), normalize='all'))
  print(classification_report(Y4_train, np.argmax(Y4_pred_train, axis=1)))
  print(np.mean(Y4_pred_train, axis=0))    
  print(Counter(np.argmax(Y4_pred_train, axis=1)))

  feature_importance = pd.DataFrame({'feature':feature_names , 
                                   'importance':ngb4.feature_importances_[0]})\
    .sort_values('importance',ascending=False).reset_index().drop(columns='index')
  fig, ax = plt.subplots(figsize=(20, 20))
  plt.title('Feature Importance Plot')
  sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance.iloc[0:100])


  ngb5 = NGBClassifier(Base=DecisionTreeRegressor(max_features=0.1,
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
          Dist=k_categorical(7),
                      n_estimators=2000, verbose_eval=10,
                 learning_rate=0.005,
                 minibatch_frac=0.5) # tell ngboost that there are 3 possible outcomes
  ngbmodel = ngb5.fit(X11_train, np.concatenate( [Y5_train]*2, axis=0), 
                      X_val = X_test, Y_val = Y5_test,
                     early_stopping_rounds = 500, 
#                     train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
#                     val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     ) # Y should have only 3 values: {0,1,2}
  spd5_train, spd5_test = model_progress(ngb5, X_train, X_test, Y5_train, Y5_test)  
  max_iter = 1000

  # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
  #Y5_pred = ngb5.predict_proba(X_test)
  Y5_pred_new = np.mean([x.class_probs() for x in ngb5.staged_pred_dist(X_pred)[100:]], axis=0)
  Y5_pred = np.mean([x.class_probs() for x in ngb5.staged_pred_dist(X_test)[100:]], axis=0)
  Y5_pred_train = np.mean([x.class_probs() for x in ngb5.staged_pred_dist(X_train)[100:]], axis=0)

  print(pd.concat([
          print_performance(Y5_test, Y5_pred, name="ngb5 test"),
          print_performance(Y5_train, Y5_pred_train, name="ngb5 train")]))

  #Y5_pred = spd5_test[max_iter].class_probs() 
  lb = None #["Loss", "Draw", "Win"]
  print(np.argmax(Y5_pred, axis=1))
  print(accuracy_score(Y5_test, np.argmax(Y5_pred, axis=1)))
  print(confusion_matrix(Y5_test, np.argmax(Y5_pred, axis=1)))
  print(confusion_matrix(Y5_test, np.argmax(Y5_pred, axis=1), normalize='all'))
  print(classification_report(Y5_test, np.argmax(Y5_pred, axis=1)))
  print(np.mean(Y5_pred, axis=0))    
  print(Counter(np.argmax(Y5_pred, axis=1)))
  
  #Y5_pred_train = ngb5.predict_proba(X_train)
  #Y5_pred_train = spd5_train[max_iter].class_probs() 
  print(accuracy_score(Y5_train, np.argmax(Y5_pred_train, axis=1)))
  print(confusion_matrix(Y5_train, np.argmax(Y5_pred_train, axis=1)))
  print(confusion_matrix(Y5_train, np.argmax(Y5_pred_train, axis=1), normalize='all'))
  print(classification_report(Y5_train, np.argmax(Y5_pred_train, axis=1)))
  print(np.mean(Y5_pred_train, axis=0))    
  print(Counter(np.argmax(Y5_pred_train, axis=1)))




  def  calc_softpoints(Y, prob):
      return np.matmul(prob, point_matrix).flatten()[np.arange(Y.shape[0])*49+Y]

  def argmax_softpoint(prob):
      return np.argmax(np.matmul(prob, point_matrix), axis=1)
  
  def calc_points(Y_true, Y_pred):
      oh = np.zeros((Y_true.size, 49))
      oh[np.arange(Y_true.size), Y_pred] = 1
      return calc_softpoints(Y_true, oh)

  def beautify(c):
      return Counter({str(k//7)+":"+str(np.mod(k,7)):v for k,v in c.items()})
  
  def invert(is_home, pred):
      return np.where(is_home, pred, 7*np.mod(pred,7)+pred//7)
  
  def point_dist(points):
      vc = pd.DataFrame({"Points":points}).Points.value_counts()
      df = pd.DataFrame({"Points":vc, "Percent":(vc / vc.sum()) * 100})
      return df

  def point_summary(points, pred):
      df = pd.DataFrame({"Points":points, "Pred":[str(k//7)+":"+str(np.mod(k,7)) for k in pred]})
      pv = df.pivot_table(index="Pred", columns="Points", aggfunc=[len], fill_value=0, margins=True)                
      pv2 = pv.div( pv.iloc[:,-1], axis=0 )
      pv3 = pv.div( pv.iloc[-1,:], axis=1 )
      return pd.concat([pv, 100*pv2.iloc[:,:-1], 100*pv3.iloc[:,-1]], axis=1)

  
  ngb3 = NGBClassifier(
          Base=DecisionTreeRegressor(max_features=0.1, #"auto",
                                                 ccp_alpha=0.0,
                                         criterion='friedman_mse', max_depth=1,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort='deprecated',
                                         random_state=None, splitter='best'),
          Dist=k_categorical(49), 
                       Score=MLE, 
                      n_estimators=4000, verbose_eval=10,
                 learning_rate=0.002,
                 minibatch_frac=0.5) # tell ngboost that there are 3 possible outcomes

  ngbmodel3 = ngb3.fit(X11_train, np.concatenate([Y3_train]*2, axis=0), 
                       X_val = X_test, Y_val = Y3_test,
                       early_stopping_rounds = 1000, 
#                     train_loss_monitor=lambda D,Y: 
#                         -np.mean(calc_points(Y, argmax_softpoint(D.class_probs()))),
#                     val_loss_monitor=lambda D,Y: 
#                         -np.mean(calc_points(Y, argmax_softpoint(D.class_probs()))),
#                     train_loss_monitor=lambda D,Y: 
#                         -np.mean(calc_softpoints(Y, D.class_probs())),
#                     val_loss_monitor=lambda D,Y: 
#                         -np.mean(calc_softpoints(Y, D.class_probs())),
                     #train_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1))),
                     #val_loss_monitor=lambda D,Y: -(np.mean(Y==np.argmax(D.class_probs(), axis=1)))
                     
                     ) # Y should have only 3 values: {0,1,2}
  ngb3.n_estimators
  max_iter=ngb3.best_val_loss_itr
  #max_iter=227
  
#  spd_train = ngb3.staged_pred_dist(X_train)
#  spd_test = ngb3.staged_pred_dist(X_test)
  spd_train, spd_test = model_progress(ngb3, X_train, X_test, Y3_train, Y3_test)  
  fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(15,10))
  axs[0][0].set_title('Softpoints')
  axs[0][0].plot([np.mean(calc_softpoints(Y3_train, spd_train[i].class_probs())) for i in range(len(spd_train))])
  axs[0][0].plot([np.mean(calc_softpoints(Y3_test, spd_test[i].class_probs())) for i in range(len(spd_test))])
  axs[0][1].set_title('Points')
  axs[0][1].plot([np.mean(calc_points(Y3_train, argmax_softpoint(spd_train[i].class_probs()))) for i in range(len(spd_train))])
  axs[0][1].plot([np.mean(calc_points(Y3_test, argmax_softpoint(spd_test[i].class_probs()))) for i in range(len(spd_test))])
  axs[1][0].set_title('Accuracy')
  axs[1][0].plot([accuracy_score(Y3_train, np.argmax(spd_train[i].class_probs(), axis=1))for i in range(len(spd_train))])
  axs[1][0].plot([accuracy_score(Y3_test, np.argmax(spd_test[i].class_probs(), axis=1))for i in range(len(spd_test))])
  axs[1][1].set_title('Log Loss')
  axs[1][1].plot([log_loss(Y3_train, spd_train[i].class_probs())for i in range(len(spd_train))])
  axs[1][1].plot([log_loss(y_true = Y3_test, y_pred = spd_test[i].class_probs(), labels = range(49))for i in range(len(spd_test))])
  axs[2][0].set_title('Balanced Accuracy Max. Prob')
  axs[2][0].plot([balanced_accuracy_score(Y3_train, np.argmax(spd_train[i].class_probs(), axis=1))for i in range(len(spd_train))])
  axs[2][0].plot([balanced_accuracy_score(Y3_test, np.argmax(spd_test[i].class_probs(), axis=1))for i in range(len(spd_test))])
  axs[2][1].set_title('Balanced Accurancy Max. Points')
  axs[2][1].plot([balanced_accuracy_score(Y3_train, argmax_softpoint(spd_train[i].class_probs()))for i in range(len(spd_train))])
  axs[2][1].plot([balanced_accuracy_score(Y3_test, argmax_softpoint(spd_test[i].class_probs()))for i in range(len(spd_test))])
  plt.show()
  

  max_iter=-1    
  # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
#  Y3_pred = ngbmodel3.predict_proba(X_test)
#  Y3_pred = ngb3.predict_proba(X_test, max_iter=max_iter)
  #Y3_pred = spd_test[max_iter-10].class_probs() 
  Y3_pred = np.mean([x.class_probs() for x in spd_test[100:]] , axis=0)
  Y3_pred_train = np.mean([x.class_probs() for x in spd_train[100:]] , axis=0)
  Y3_pred_new = np.mean([x.class_probs() for x in ngb3.staged_pred_dist(X_pred)[100:]], axis=0)
  
  print(pd.concat([
          print_performance(Y3_test, Y3_pred, name="ngb3 test"),
          print_performance(Y3_train, Y3_pred_train, name="ngb3 train")]))


  print(np.mean(calc_softpoints(Y3_test, Y3_pred)))
  print(beautify(Counter(invert(X_test[:,1], argmax_softpoint(Y3_pred)))))
  print(np.mean(calc_points(Y3_test, argmax_softpoint(Y3_pred))))
  print(point_dist(calc_points(Y3_test, argmax_softpoint(Y3_pred))))
  print(point_summary(calc_points(Y3_test, argmax_softpoint(Y3_pred)), invert(X_test[:,1], argmax_softpoint(Y3_pred))))
  print(accuracy_score(Y3_test, argmax_softpoint(Y3_pred)))
  print(beautify(Counter(invert(X_test[:,1], np.argmax(Y3_pred, axis=1)))))
  print(np.mean(calc_points(Y3_test, np.argmax(Y3_pred, axis=1))))
  print(point_dist(calc_points(Y3_test, np.argmax(Y3_pred, axis=1))))
  print(accuracy_score(Y3_test, np.argmax(Y3_pred, axis=1)))
  print(point_summary(calc_points(Y3_test, np.argmax(Y3_pred, axis=1)), invert(X_test[:,1], np.argmax(Y3_pred, axis=1))))

  
#  print(np.argmax(Y3_pred, axis=1))
#  print(confusion_matrix(Y3_test, np.argmax(Y3_pred, axis=1)))
#  #print(confusion_matrix(Y_test, np.argmax(Y_pred, axis=1), normalize='all'))
#  print(classification_report(Y3_test, np.argmax(Y3_pred, axis=1)))
#  print(np.mean(Y3_pred, axis=0)*100)    
#  print(np.sqrt(np.var(Y3_pred, axis=0))*100)    
#  print(np.max(Y3_pred, axis=0)*100)    
#  print(np.min(Y3_pred, axis=0)*100)    
#  print(np.sqrt(np.var(Y3_pred, axis=0))/np.mean(Y3_pred, axis=0))    
#
#  print(classification_report(Y3_test, argmax_softpoint(Y3_pred)))
  
#  Y3_pred_train = ngb3.predict_proba(X_train)
#  Y3_pred_train = ngb3.predict_proba(X_train, max_iter=max_iter)
  #Y3_pred_train = spd_train[max_iter].class_probs() 
   
  
  print(np.mean(calc_softpoints(Y3_train, Y3_pred_train)))
  print(beautify(Counter(invert(X_train[:,1], argmax_softpoint(Y3_pred_train)))))
  print(np.mean(calc_points(Y3_train, argmax_softpoint(Y3_pred_train))))
  print(point_dist(calc_points(Y3_train, argmax_softpoint(Y3_pred_train))))
  print(point_summary(calc_points(Y3_train, argmax_softpoint(Y3_pred_train)), invert(X_train[:,1], argmax_softpoint(Y3_pred_train))))
  print(accuracy_score(Y3_train, argmax_softpoint(Y3_pred_train)))
  print(beautify(Counter(invert(X_train[:,1], np.argmax(Y3_pred_train, axis=1)))))
  print(np.mean(calc_points(Y3_train, np.argmax(Y3_pred_train, axis=1))))
  print(point_dist(calc_points(Y3_train, np.argmax(Y3_pred_train, axis=1))))
  print(point_summary(calc_points(Y3_train, np.argmax(Y3_pred_train, axis=1)), invert(X_train[:,1], np.argmax(Y3_pred_train, axis=1))))
  print(accuracy_score(Y3_train, np.argmax(Y3_pred_train, axis=1)))

#  print(confusion_matrix(Y3_train, np.argmax(Y_pred_train, axis=1)))
#  print(confusion_matrix(Y3_train, argmax_softpoint(Y_pred_train)))
#  #print(confusion_matrix(Y_train, np.argmax(Y_pred_train, axis=1), normalize='all'))
#  print(classification_report(Y3_train, np.argmax(Y_pred_train, axis=1)))
#  print(np.mean(Y_pred_train, axis=0)*100)    
#  print(np.sqrt(np.var(Y_pred_train, axis=0))*100)    
#  print(np.max(Y_pred_train, axis=0)*100)    
#  print(np.min(Y_pred_train, axis=0)*100)    
#  print(classification_report(Y3_train, argmax_softpoint(Y_pred_train)))

  
  feature_importance = pd.DataFrame({'feature':feature_names , 
                                   'importance':ngb3.feature_importances_[0]})\
    .sort_values('importance',ascending=False).reset_index().drop(columns='index')
  fig, ax = plt.subplots(figsize=(20, 20))
  plt.title('Feature Importance Plot')
  sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance.iloc[0:100])

  def print_evaluation(labels, probs, where):  
      print(np.mean(calc_softpoints(labels, probs)))
    #  print(beautify(Counter(invert(X_test[:,1], argmax_softpoint(Y7_pred)))))
      print(np.mean(calc_points(labels, argmax_softpoint(probs))))
    #  print(point_dist(calc_points(Y3_test, argmax_softpoint(Y7_pred))))
      print(point_summary(calc_points(labels, argmax_softpoint(probs)), invert(where, argmax_softpoint(probs))))

  def print_evaluation_2max(labels, probs, where):  
      print(np.mean(calc_softpoints(labels, probs)))
      # descending order indexes
      probs_ind = np.argsort( probs, axis=1 )[:, ::-1]
      c1=probs_ind[:,0]
      c2=probs_ind[:,1]
      sp1=calc_softpoints(c1, probs)
      sp2=calc_softpoints(c2, probs)
      pred = np.where(sp1>sp2, c1, c2)
      print(np.mean(calc_points(labels, pred)))
      print(point_summary(calc_points(labels, pred), invert(where, pred)))

  def print_evaluation_3max(labels, probs, where):  
      print(np.mean(calc_softpoints(labels, probs)))
      # descending order indexes
      probs_ind = np.argsort( probs, axis=1 )[:, ::-1]
      c1=probs_ind[:,0]
      c2=probs_ind[:,1]
      c3=probs_ind[:,2]
      sp1=calc_softpoints(c1, probs)
      sp2=calc_softpoints(c2, probs)
      sp3=calc_softpoints(c3, probs)
      pred_idx = np.argmax([sp1, sp2, sp3], axis=0)
      pred = probs_ind[range(len(pred_idx)),pred_idx]
      print(np.mean(calc_points(labels, pred)))
      print(point_summary(calc_points(labels, pred), invert(where, pred)))

  def combine_probs_3_49(probs3, probs49):
      Y3_bw_pred_train0 = np.stack([np.sum([probs49[:, i]  for i in range(49) if i//7<np.mod(i,7)], axis=0),
                                   np.sum([probs49[:, i]  for i in range(49) if i//7==np.mod(i,7)], axis=0),
                                   np.sum([probs49[:, i]  for i in range(49) if i//7>np.mod(i,7)], axis=0)]
        , axis=1)
      Y3_bw_pred_train = np.stack([probs3[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] * probs49[:,i] / Y3_bw_pred_train0[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] for i in range(49)], axis=1)
      return Y3_bw_pred_train 
  
    
  def combine_probs(probsarray):    
      Y_bwin_pred_train, Y1_pred_train, Y2_pred_train, Y3_pred_train, Y4_pred_train, Y5_pred_train, Y6_pred_train, Y8_pred_train = probsarray 
    

      Y3_bw_pred_train0 = np.stack([np.sum([Y3_pred_train[:, i]  for i in range(49) if i//7<np.mod(i,7)], axis=0),
                                   np.sum([Y3_pred_train[:, i]  for i in range(49) if i//7==np.mod(i,7)], axis=0),
                                   np.sum([Y3_pred_train[:, i]  for i in range(49) if i//7>np.mod(i,7)], axis=0)]
        , axis=1)
      Y3_bw_pred_train = np.stack([Y_bwin_pred_train[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] * Y3_pred_train[:,i] / Y3_bw_pred_train0[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] for i in range(49)], axis=1)

#      Y3_bw_pred_train = np.stack([Y_bwin_pred_train[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] for i in range(49)], axis=1)
#      Y3_bw_pred_train = np.stack([Y3_bw_pred_train[:, i]/7 if i//7==np.mod(i,7) else Y3_bw_pred_train[:, i]/21 for i in range(49)], axis=1)

      Y3_bw_pred_train_pure = np.stack([Y_bwin_pred_train[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] for i in range(49)], axis=1)
      Y3_bw_pred_train_pure = np.stack([Y3_bw_pred_train_pure[:, i] if (i//7==1 or np.mod(i,7)==1) and ((i//7+np.mod(i,7))==3 or (i//7+np.mod(i,7))==2) else Y3_bw_pred_train[:, i]*0 for i in range(49)], axis=1)
        
      Y3_1_pred_train = np.stack([Y1_pred_train[:, np.sign(i//7-np.mod(i,7)).astype(int)+1 ] for i in range(49)], axis=1)
      Y3_1_pred_train = np.stack([Y3_1_pred_train[:, i]/7 if i//7==np.mod(i,7) else Y3_1_pred_train[:, i]/21 for i in range(49)], axis=1)
    
      Y3_8_pred_train = np.stack([Y8_pred_train[:,2] /7 if i//7-np.mod(i,7)==0 else 
                            Y8_pred_train[:,1] /6 if i//7-np.mod(i,7)==-1 else 
                            Y8_pred_train[:,3] /6 if i//7-np.mod(i,7)==1 else 
                            Y8_pred_train[:,0] /5 if i//7-np.mod(i,7)==-2 else 
                            Y8_pred_train[:,4] /5 if i//7-np.mod(i,7)==2 else 
                            0.0*Y8_pred_train[:,4] for i in range(49)], axis=1)
    
      Y3_2_pred_train = np.stack([Y2_pred_train[:, (i//7-np.mod(i,7)).astype(int)+6 ] for i in range(49)], axis=1)
      Y3_2_pred_train = np.stack([Y3_2_pred_train[:, i]/(7-np.abs(i//7-np.mod(i,7)))  for i in range(49)], axis=1)

#      Y3_4_pred_train = np.stack([Y4_pred_train[:, (i//7)]/7 for i in range(49)], axis=1)
#      Y3_5_pred_train = np.stack([Y5_pred_train[:, np.mod(i,7)]/7 for i in range(49)], axis=1)
      Y3_45_pred_train = np.stack([Y4_pred_train[:, (i//7)]*Y5_pred_train[:, np.mod(i,7)] for i in range(49)], axis=1)
      Y3_45_pred_train = Y3_45_pred_train / np.sum(Y3_45_pred_train, axis=1, keepdims=True)
      #Y3_6_pred_train = np.stack([Y3_6_pred_train[:, i]/(7-np.abs(6 - i//7 - np.mod(i,7)))  for i in range(49)], axis=1)
      
      Y3_6_pred_train = np.stack([Y6_pred_train[:, (i//7+np.mod(i,7)).astype(int)] for i in range(49)], axis=1)
    
      Y3_2b_pred_train = np.stack([Y3_2_pred_train[:, i]/
                                   np.sum([ Y3_2_pred_train[:, j] for j in range(49) if (i//7+np.mod(i,7))==(j//7+np.mod(j,7)) ], 
                                           axis=0)  for i in range(49)], 
                                   axis=1)
      Y3_26_pred_train = Y3_2b_pred_train * Y3_6_pred_train
      for k in range(10):
          Y3_26a_pred_train = np.stack([np.sum(np.stack([Y3_26_pred_train[:,j] for j in range(49) if j//7-np.mod(j,7)+6==i]).T, axis=1) for i in range(13)], axis=1)
          Y3_26_pred_train = np.stack([Y3_26_pred_train[:,i] * Y2_pred_train[:, (i//7)-np.mod(i,7)+6] / Y3_26a_pred_train[:, (i//7)-np.mod(i,7)+6] for i in range(49)], axis=1)

          Y3_26b_pred_train = np.stack([np.sum(np.stack([Y3_26_pred_train[:,j] for j in range(49) if (i//7+np.mod(i,7))==(j//7+np.mod(j,7))]).T, axis=1) for i in range(13)], axis=1)
          Y3_26_pred_train = np.stack([Y3_26_pred_train[:,i] * Y6_pred_train[:, i//7+np.mod(i,7)] / Y3_26b_pred_train[:, i//7+np.mod(i,7)] for i in range(49)], axis=1)

      
#      Y3_26_pred_train = np.stack([Y2_pred_train[:, (i//7-np.mod(i,7)+6).astype(int)] * 
#                                                 Y6_pred_train[:, max(i//7, np.mod(i,7))] / 
#                                                 np.sum(Y6_pred_train[:, (abs(i//7-np.mod(i,7)).astype(int)) : 7], axis=1) for i in range(49)], axis=1)
#      for k in range(10):
#          Y3_26b_pred_train = np.stack([np.sum(np.stack([Y3_26_pred_train[:,j] for j in range(49) if max(j//7, np.mod(j,7))==i]).T, axis=1) for i in range(7)], axis=1)
#    
#          Y3_26_pred_train = np.stack([Y3_26_pred_train[:,i] * Y6_pred_train[:, max(i//7, np.mod(i,7))] / Y3_26b_pred_train[:, max(i//7, np.mod(i,7))] for i in range(49)], axis=1)
#          
#          Y3_26a_pred_train = np.stack([np.sum(np.stack([Y3_26_pred_train[:,j] for j in range(49) if j//7-np.mod(j,7)+6==i]).T, axis=1) for i in range(13)], axis=1)
#    
#          Y3_26_pred_train = np.stack([Y3_26_pred_train[:,i] * Y2_pred_train[:, (i//7)-np.mod(i,7)+6] / Y3_26a_pred_train[:, (i//7)-np.mod(i,7)+6] for i in range(49)], axis=1)
#
##      Y3_26_pred_train = np.stack([Y2_pred_train[:, (i//7-np.mod(i,7)+6).astype(int)] * 
##                                                 Y6_pred_train[:, max(i//7, np.mod(i,7))] / 
##                                                 np.sum(Y2_pred_train[:, 6-max(i//7, np.mod(i,7)) : 7+max(i//7, np.mod(i,7))], axis=1) for i in range(49)], axis=1)
      Y7_pred_train = (Y3_bw_pred_train+Y3_pred_train+Y3_1_pred_train+Y3_8_pred_train+Y3_45_pred_train+Y3_26_pred_train)/6
      
      combined_probs = [Y3_bw_pred_train, Y3_1_pred_train, Y3_8_pred_train, Y3_26_pred_train, Y3_45_pred_train, Y3_bw_pred_train_pure, Y7_pred_train]
      return combined_probs
  
  #Y3_26_pred_train = np.stack([(Y2_pred_train[:, (i//7-np.mod(i,7)).astype(int)+6 ])*(Y6_pred_train[:, (i//7+np.mod(i,7)).astype(int)])  for i in range(49)], axis=1)
  #Y3_26_pred_train = Y3_26_pred_train / np.sum(Y3_26_pred_train, axis=1, keepdims=True)

  Y3_bw_pred_train, Y3_1_pred_train, Y3_8_pred_train, Y3_26_pred_train, Y3_45_pred_train, Y3_bw_pred_train_pure, Y7_pred_train = combine_probs([Y_bwin_pred_train, Y1_pred_train, Y2_pred_train, Y3_pred_train, Y4_pred_train, Y5_pred_train, Y6_pred_train, Y8_pred_train])
  Y3_bw_pred, Y3_1_pred, Y3_8_pred, Y3_26_pred, Y3_45_pred, Y3_bw_pred_pure, Y7_pred = combine_probs([Y_bwin_pred, Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, Y8_pred])
  Y3_bw_pred_new, Y3_1_pred_new, Y3_8_pred_new, Y3_26_pred_new, Y3_45_pred_new, Y3_bw_pred_new_pure, Y7_pred_new = combine_probs([Y_bwin_pred_new, Y1_pred_new, Y2_pred_new, Y3_pred_new, Y4_pred_new, Y5_pred_new, Y6_pred_new, Y8_pred_new])
  
  #combine_probs_3_49(Y1_pred, Y3_pred)
  #combine_probs_3_49(Y_spi_pred, Y3_pred)
  
  print(pd.concat([
          print_performance(Y3_test, combine_probs_3_49(Y1_pred, Y3_pred), name="ngb1+3 test"),
          print_performance(Y3_train, combine_probs_3_49(Y1_pred_train, Y3_pred_train), name="ngb1+3 train"),
          print_performance(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_pred), name="spi +Y3 test"),
          print_performance(Y3_train, combine_probs_3_49(Y_spi_pred_train, Y3_pred_train), name="spi+Y3 train"),
          print_performance(Y3_test, combine_probs_3_49(Y_bwin_raw_test, Y3_pred), name="bwin raw +Y3 test"),
          print_performance(Y3_train, combine_probs_3_49(Y_bwin_raw_train, Y3_pred_train), name="bwin raw +Y3 train"),
          print_performance(Y3_test, Y3_pred, name="Y3 test"),
          print_performance(Y3_train, Y3_pred_train, name="Y3 train")]))

  print("Y eg")  
  print_evaluation(Y3_test, Y_eg_raw_test, X_test[:,1])
  print_evaluation(Y3_train, Y_eg_raw_train, X_train[:,1])

  print("Y3 eg")  
  print_evaluation(Y3_test, Y3_eg_raw_test, X_test[:,1])
  print_evaluation(Y3_train, Y3_eg_raw_train, X_train[:,1])

  print("Y3 test")  
  print_evaluation(Y3_test, Y3_pred, X_test[:,1])
  print("Y3 test 2max")  
  print_evaluation_2max(Y3_test, Y3_pred, X_test[:,1])
  print("Y3 test 3max")  
  print_evaluation_3max(Y3_test, Y3_pred, X_test[:,1])
  print("Y45 test")  
  print_evaluation(Y3_test, Y3_45_pred, X_test[:,1])
  print("Y45 test_2max")  
  print_evaluation_2max(Y3_test, Y3_45_pred, X_test[:,1])
  print("Y26 test")  
  print_evaluation(Y3_test, Y3_26_pred, X_test[:,1])
  print("Y26 test_2max")  
  print_evaluation_2max(Y3_test, Y3_26_pred, X_test[:,1])
  print("Y26 +Y3 test")  
  print_evaluation(Y3_test, (Y3_26_pred+9*Y3_pred)/10, X_test[:,1])
  print("Y1+Y3 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y1_pred, Y3_pred), X_test[:,1])
  print("Y1+Y eg test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y1_pred, Y_eg_raw_test), X_test[:,1])
  print("Y1+Y3 eg test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y1_pred, Y3_eg_raw_test), X_test[:,1])
  print("Y1+Y45 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y1_pred, Y3_45_pred), X_test[:,1])
  print("Y1+Y26 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y1_pred, Y3_26_pred), X_test[:,1])
  print("SPI+Y3 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_pred), X_test[:,1])
  print("SPI+Y45 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_45_pred), X_test[:,1])
  print("SPI+Y26 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_26_pred), X_test[:,1])
  print("SPI+Y26 test 2max")  
  print_evaluation_2max(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_26_pred), X_test[:,1])
  print("SPI+Y26 test 3max")  
  print_evaluation_3max(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_26_pred), X_test[:,1])
  print("BWIN+Y45 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_pred, Y3_45_pred), X_test[:,1])
  print("BWIN+Y26 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_pred, Y3_26_pred), X_test[:,1])
  print("BWIN raw +Y26 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_raw_test, Y3_26_pred), X_test[:,1])
  print("BWIN raw +Y eg test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_raw_test, Y_eg_raw_test), X_test[:,1])
  print("BWIN raw +Y3 eg test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_raw_test, Y3_eg_raw_test), X_test[:,1])
  print("SPI raw +Y26 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_spi_raw_test, Y3_26_pred), X_test[:,1])
  print("SPI +Y eg test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_spi_pred, Y_eg_raw_test), X_test[:,1])
  print("SPI +Y3 eg test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_spi_pred, Y3_eg_raw_test), X_test[:,1])
  print("BWIN raw+Y3 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_raw_test, Y3_pred), X_test[:,1])
  print("BWIN raw+Y3+Y8 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_raw_test, (Y3_pred+Y3_8_pred)/2), X_test[:,1])
  print("BWIN raw+Y45 test")  
  print_evaluation(Y3_test, combine_probs_3_49(Y_bwin_raw_test, Y3_45_pred), X_test[:,1])
  print("Blend 1")  
  print_evaluation(Y3_test, combine_probs_3_49((Y_bwin_raw_test+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_26_pred)/2), X_test[:,1])
  print_evaluation(Y3_train, combine_probs_3_49((Y_bwin_raw_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train)/2), X_train[:,1])
  print("Blend 2")  
  print_evaluation(Y3_test, combine_probs_3_49((Y_bwin_pred+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_26_pred+Y3_8_pred)/3), X_test[:,1])
  print_evaluation(Y3_train, combine_probs_3_49((Y_bwin_pred_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train+Y3_8_pred_train)/3), X_train[:,1])
  print("Blend 2 2max")  
  print_evaluation_2max(Y3_test, combine_probs_3_49((Y_bwin_pred+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_26_pred+Y3_8_pred)/3), X_test[:,1])
  print_evaluation_2max(Y3_train, combine_probs_3_49((Y_bwin_pred_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train+Y3_8_pred_train)/3), X_train[:,1])

  print("Blend 2 3max")  
  print_evaluation_3max(Y3_test, combine_probs_3_49((Y_bwin_pred+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_26_pred+Y3_8_pred)/3), X_test[:,1])
  print_evaluation_3max(Y3_train, combine_probs_3_49((Y_bwin_pred_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train+Y3_8_pred_train)/3), X_train[:,1])
  print("Blend 3")  
  print_evaluation(Y3_test, combine_probs_3_49((Y_bwin_raw_test+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_26_pred+Y3_45_pred+Y3_8_pred)/4), X_test[:,1])
  print_evaluation(Y3_train, combine_probs_3_49((Y_bwin_raw_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train+Y3_45_pred_train+Y3_8_pred_train)/4), X_train[:,1])

  print("Blend 4")  
  print_evaluation(Y3_test, combine_probs_3_49((Y_spi_raw_test+Y_bwin_raw_test+Y_bwin_pred+Y1_pred+Y_spi_pred)/5, 
                                               (Y3_pred+Y3_26_pred)/2), X_test[:,1])
  print_evaluation(Y3_train, combine_probs_3_49((Y_spi_raw_train+Y_bwin_raw_train+Y_bwin_pred_train+Y1_pred_train+Y_spi_pred_train)/5, 
                                               (Y3_pred_train+Y3_26_pred_train)/2), X_train[:,1])

  print("Blend 5")  
  print_evaluation(Y3_test, combine_probs_3_49((Y_bwin_raw_test+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_26_pred+Y_eg_raw_test+Y3_eg_raw_test)/6), X_test[:,1])
  print_evaluation(Y3_train, combine_probs_3_49((Y_bwin_raw_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train+Y_eg_raw_train+Y3_eg_raw_train)/6), X_train[:,1])

  print("Blend 6")  
  print_evaluation(Y3_test, combine_probs_3_49((Y_bwin_raw_test+Y1_pred+Y_spi_pred)/3, 
                                               (Y3_pred+Y3_8_pred+Y3_eg_raw_test+Y3_26_pred+Y3_45_pred)/5), X_test[:,1])
  print_evaluation(Y3_train, combine_probs_3_49((Y_bwin_raw_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_8_pred_train+Y3_eg_raw_train+Y3_26_pred_train+Y3_45_pred_train)/5), X_train[:,1])

  Y7_pred = combine_probs_3_49(Y_spi_pred, Y3_26_pred)  
  Y7_pred_train = combine_probs_3_49(Y_spi_pred_train, Y3_26_pred_train)  
  Y7_pred_new = combine_probs_3_49(Y_spi_pred_new, Y3_26_pred_new)  

  Y7_pred = combine_probs_3_49(Y1_pred, Y3_eg_raw_test)  
  Y7_pred_train = combine_probs_3_49(Y1_pred_train, Y3_eg_raw_train)  
  Y7_pred_new = combine_probs_3_49(Y1_pred_new, Y3_eg_raw_new)  

  Y7_pred = combine_probs_3_49((Y_bwin_pred+Y1_pred+Y_spi_pred)/3, (Y3_pred+Y3_26_pred+Y3_8_pred)/3)
  Y7_pred_train = combine_probs_3_49((Y_bwin_pred_train+Y1_pred_train+Y_spi_pred_train)/3, 
                                               (Y3_pred_train+Y3_26_pred_train+Y3_8_pred_train)/3)
  Y7_pred_new = combine_probs_3_49((Y_bwin_pred_new+Y1_pred_new+Y_spi_pred_new)/3, 
                                               (Y3_pred_new+Y3_26_pred_new+Y3_8_pred_new)/3)
  index=-4
  print(pd.DataFrame({"Y_bwin_raw_test":Y_bwin_raw_test[index], "Y_bwin_pred":Y_bwin_pred[index], "Y1_pred":Y1_pred[index], "Y_spi_pred":Y_spi_pred[index], "Y_spi_raw_test":Y_spi_raw_test[index]}))
  plot_probs(probs=Y3_pred[index], softpoints=np.matmul(Y3_pred, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y3_pred")
  plot_probs(probs=Y7_pred[index], softpoints=np.matmul(Y7_pred, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y7_pred")
  print(Y4_pred[index])
  print(Y5_pred[index])
  plot_probs(probs=Y3_45_pred[index], softpoints=np.matmul(Y3_45_pred, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y3_45_pred")
  print(np.round(Y2_pred[index]*100, 1))
  print(np.round(Y6_pred[index]*100, 1))
  plot_probs(probs=Y3_26_pred[index], softpoints=np.matmul(Y3_26_pred, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y3_26_pred")
  plot_probs(probs=Y3_8_pred[index], softpoints=np.matmul(Y3_8_pred, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y3_8_pred")
  print(X_test_eg[index])
  plot_probs(probs=Y_eg_raw_test[index], softpoints=np.matmul(Y_eg_raw_test, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y_eg_raw_test")
  plot_probs(probs=Y3_eg_raw_test[index], softpoints=np.matmul(Y3_eg_raw_test, point_matrix)[index], gs=Y3_test[index]//7, gc=np.mod(Y3_test[index], 7), title="Y3_eg_raw_test")

  
  print_evaluation(Y3_train, Y7_pred_train, X_train[:,1])
  print_evaluation(Y3_test, Y7_pred, X_test[:,1])

  if False:  
      # point estimate ...  
      prednew = argmax_softpoint(Y7_pred_new)
      sp_prednew = calc_softpoints(prednew, Y7_pred_new)
      
      all_quotes =  pd.read_csv(FLAGS.data_dir+"/all_quotes_bwin.csv")
    
      prednewdf = pd.DataFrame({
              "GS":prednew//7, "GC":np.mod(prednew, 7), 
              "SP":sp_prednew
              })
      prednewdf.GS = prednewdf.GS.astype(str).str.cat(prednewdf.GC.astype(str), sep=":")
      prednewdf.drop(["GC"], axis=1, inplace=True)
      print(pd.concat([prednewdf, all_quotes], axis=1))


  if True:  
      # point estimate ...  
      Y7_pred_new2 =   (Y7_pred_new[::2] +   Y7_pred_new[1::2].reshape((-1,7,7)).transpose((0,2,1)).reshape((-1,49)))/2
      prednew = argmax_softpoint(Y7_pred_new)
      prednew2 = argmax_softpoint(Y7_pred_new2)
      sp_prednew = calc_softpoints(prednew, Y7_pred_new)
      sp_prednew2 = calc_softpoints(prednew2, Y7_pred_new2)
      #Y7_pred_new[1::2].reshape((-1,7,7)).transpose((0,2,1)).reshape((-1,49))
      
      all_quotes =  pd.read_csv(FLAGS.data_dir+"/all_quotes_bwin.csv")
    
      prednewdf = pd.DataFrame({
              "GS":prednew[::2]//7, "GC":np.mod(prednew[::2], 7), 
              "GSA":np.mod(prednew[1::2], 7), "GCA":prednew[1::2]//7,
              "GS2":prednew2//7, "GC2":np.mod(prednew2, 7),
              "SP":sp_prednew[::2],
              "SPA":sp_prednew[1::2],
              "SP2":sp_prednew2
              })
      prednewdf.GS = prednewdf.GS.astype(str).str.cat(prednewdf.GC.astype(str), sep=":")
      prednewdf.GSA = prednewdf.GSA.astype(str).str.cat(prednewdf.GCA.astype(str), sep=":")
      prednewdf.GS2 = prednewdf.GS2.astype(str).str.cat(prednewdf.GC2.astype(str), sep=":")
      prednewdf.drop(["GC", "GCA", "GC2"], axis=1, inplace=True)
      print(pd.concat([prednewdf, all_quotes], axis=1))

  #Y7_pred[1000].reshape((7,7))*100
  #Y2_pred[0]

  # print("Y3_bw_pred")
  # print_evaluation(Y3_train, Y3_bw_pred_train, X_train[:,1])
  # print_evaluation(Y3_test, Y3_bw_pred, X_test[:,1])
  # print("(Y3_bw_pred+Y3_pred)/2")
  # print_evaluation(Y3_train, (Y3_bw_pred_train+Y3_pred_train)/2, X_train[:,1])
  # print_evaluation(Y3_test, (Y3_bw_pred+Y3_pred)/2, X_test[:,1])
  # print("Y3_bw_pred_pure")
  # print_evaluation(Y3_train, Y3_bw_pred_train_pure, X_train[:,1])
  # print_evaluation(Y3_test, Y3_bw_pred_pure, X_test[:,1])
  # #  inspect
  # m=(Y3_bw_pred+Y3_pred)/2
  # #m=Y3_bw_pred
  # #m=Y7_pred
  # mm=Y_bwin_pred[argmax_softpoint(m)==8]
  # #mm=Y1_pred[argmax_softpoint(m)==8]
  # plt.figure(figsize=(15,10))        
  # plt.scatter(mm[:,0], mm[:,2], alpha=0.1)
  # plt.show()
  


  # plt.figure(figsize=(15,10))        
  # plt.plot([np.mean(calc_points(Y3_train, argmax_softpoint((spd_train[i].class_probs())/1))) for i in range(len(spd_train))])
  # plt.plot([np.mean(calc_points(Y3_train, argmax_softpoint((spd_train[i].class_probs()+Y3_1_pred_train+Y3_8_pred_train+Y3_45_pred_train+Y3_26_pred_train)/5))) for i in range(len(spd_train))])
  # plt.plot([np.mean(calc_points(Y3_train, argmax_softpoint((spd_train[i].class_probs()+Y3_45_pred_train+Y3_26_pred_train)/3))) for i in range(len(spd_train))])
  # plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint((spd_test[i].class_probs())/1))) for i in range(len(spd_test))])
  # plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint((spd_test[i].class_probs()+Y3_bw_pred)/2))) for i in range(len(spd_test))], label="BW")
  # plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint((spd_test[i].class_probs()+Y3_bw_pred+Y3_1_pred+Y3_8_pred+Y3_45_pred+Y3_26_pred)/6))) for i in range(len(spd_test))], label="all")
  # plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint((spd_test[i].class_probs()+Y3_26_pred+Y3_45_pred)/3))) for i in range(len(spd_test))])
  # plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint((2*spd_test[i].class_probs()+2*Y3_bw_pred+Y3_1_pred+Y3_8_pred+Y3_45_pred+Y3_26_pred)/8))) for i in range(len(spd_test))], label="xyz")
  # plt.legend()
  # plt.show()
  
  plt.figure(figsize=(15,10))        
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, Y1_pred, Y2_pred, spd_test[i].class_probs(), Y4_pred, Y5_pred, Y6_pred, Y8_pred])[-1])))
        for i in range(len(spd_test))], label="Y3")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, spd1_test[i].class_probs(), Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, Y8_pred])[-1])))
        for i in range(len(spd1_test))], label="Y1")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, Y1_pred, spd2_test[i].class_probs(), Y3_pred, Y4_pred, Y5_pred, Y6_pred, Y8_pred])[-1])))
        for i in range(len(spd2_test))], label="Y2")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, spd6_test[i].class_probs(), Y8_pred])[-1])))
        for i in range(len(spd6_test))], label="Y6")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, Y1_pred, Y2_pred, Y3_pred, spd4_test[i].class_probs(), Y5_pred, Y6_pred, Y8_pred])[-1])))
        for i in range(len(spd4_test))], label="Y4")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, Y1_pred, Y2_pred, Y3_pred, Y4_pred, spd5_test[i].class_probs(), Y6_pred, Y8_pred])[-1])))
        for i in range(len(spd5_test))], label="Y5")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [Y3_bw_pred, Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, spd8_test[i].class_probs()])[-1])))
        for i in range(len(spd8_test))], label="Y8")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
          [Y3_bw_pred, Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, spd8_test[i].class_probs()])[-1])))
        for i in range(len(spd8_test))], label="Y8")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
          [spd_bwin_test[i].class_probs(), Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, Y8_pred])[-1])))
        for i in range(len(spd_bwin_test))], label="Y_bwin")
  plt.legend()
  plt.show()

  #plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(spd_bwin_test[i].class_probs()))) for i in range(len(spd_bwin_test))], label="Y_bwin")
  plt.plot([np.mean(calc_points(Y3_test, argmax_softpoint(combine_probs(
          [spd_bwin_test[i].class_probs(), Y1_pred, Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, Y8_pred])[0])))
        for i in range(len(spd_bwin_test))], label="Y_bwin")

  plt.plot([np.mean(calc_softpoints(Y3_test, combine_probs(
                  [Y3_bw_pred, spd1_test[i].class_probs(), Y2_pred, Y3_pred, Y4_pred, Y5_pred, Y6_pred, Y8_pred])[-1]))
        for i in range(len(spd1_test))])

  sample_len = 300
  distribution_array=[spd_bwin_test, spd1_test, spd2_test, spd_test, spd4_test, spd5_test, spd6_test, spd8_test]
  dsi = [np.random.randint(50, len(d), size=sample_len) for d in distribution_array] # 50, len(d)
  points_per_match = [calc_points(Y3_test, argmax_softpoint(combine_probs(
                  [spd_bwin_test[dsi[0][i]].class_probs(), 
                   spd1_test[dsi[1][i]].class_probs(), 
                   spd2_test[dsi[2][i]].class_probs(), 
                   spd_test[dsi[3][i]].class_probs(), 
                   spd4_test[dsi[4][i]].class_probs(), 
                   spd5_test[dsi[5][i]].class_probs(), 
                   spd6_test[dsi[6][i]].class_probs(), 
                   spd8_test[dsi[7][i]].class_probs(), 
                   ])[-1]))
        for i in range(sample_len)]

  points = np.array(points_per_match)
  homepoints = np.mean(points[:, ::2], axis=1)
  awaypoints = np.mean(points[:, 1::2], axis=1)
  allpoints = np.mean(points, axis=1)
  plt.figure(figsize=(15,10))        
  plt.plot(allpoints)
  plt.hlines(y=np.mean(allpoints), xmin=0, xmax=sample_len)
  plt.hlines(colors="r", y=np.mean(allpoints)+2*np.sqrt(np.var(allpoints)), xmin=0, xmax=sample_len)
  plt.hlines(colors="r", y=np.mean(allpoints)-2*np.sqrt(np.var(allpoints)), xmin=0, xmax=sample_len)
  plt.show()
  print(np.mean(allpoints))      
  print(np.sqrt(np.var(allpoints)))
  plt.boxplot(allpoints)
  plt.violinplot(allpoints, showmedians=True)
  plt.violinplot(awaypoints, showmedians=True)
  plt.violinplot(homepoints, showmedians=True)
  plt.show()
  df = pd.DataFrame({"points":allpoints, "dsi":(np.mean(np.array(dsi), axis=0)//50)*50})
  sns.violinplot(x='dsi', y='points', data=df, scale="count")
  plt.show()
  df = pd.concat([pd.DataFrame({"points":homepoints, "dsi":(np.mean(np.array(dsi), axis=0)//50)*50, "where":"Home"}),
                  pd.DataFrame({"points":awaypoints, "dsi":(np.mean(np.array(dsi), axis=0)//50)*50, "where":"Away"})])
  sns.violinplot(x='dsi', y='points', data=df, hue="where", split=True, scale="count", inner="stick", scale_hue=False)
  plt.show()

  softpoints_per_match = [calc_softpoints(Y3_test, combine_probs(
                  [spd_bwin_test[dsi[0][i]].class_probs(), 
                   spd1_test[dsi[1][i]].class_probs(), 
                   spd2_test[dsi[2][i]].class_probs(), 
                   spd_test[dsi[3][i]].class_probs(), 
                   spd4_test[dsi[4][i]].class_probs(), 
                   spd5_test[dsi[5][i]].class_probs(), 
                   spd6_test[dsi[6][i]].class_probs(), 
                   spd8_test[dsi[7][i]].class_probs(), 
                   ])[-1])
        for i in range(sample_len)]
  points = np.array(softpoints_per_match)
  homepoints = np.mean(points[:, ::2], axis=1)
  awaypoints = np.mean(points[:, 1::2], axis=1)
  allsoftpoints = np.mean(points, axis=1)
  plt.figure(figsize=(15,10))        
  plt.plot(allsoftpoints)
  plt.hlines(y=np.mean(allsoftpoints), xmin=0, xmax=sample_len)
  plt.hlines(colors="r", y=np.mean(allsoftpoints)+2*np.sqrt(np.var(allsoftpoints)), xmin=0, xmax=sample_len)
  plt.hlines(colors="r", y=np.mean(allsoftpoints)-2*np.sqrt(np.var(allsoftpoints)), xmin=0, xmax=sample_len)
  plt.show()
  print(np.mean(allsoftpoints))      
  print(np.sqrt(np.var(allsoftpoints)))
  plt.boxplot(allsoftpoints)
  plt.violinplot(allsoftpoints, showmedians=True)
  plt.violinplot(awaypoints, showmedians=True)
  plt.violinplot(homepoints, showmedians=True)
  plt.show()
  df = pd.DataFrame({"points":allsoftpoints, "dsi":(np.mean(np.array(dsi), axis=0)//50)*50})
  sns.violinplot(x='dsi', y='points', data=df, scale="count")
  plt.show()
  df = pd.concat([pd.DataFrame({"points":homepoints, "dsi":(np.mean(np.array(dsi), axis=0)//50)*50, "where":"Home"}),
                  pd.DataFrame({"points":awaypoints, "dsi":(np.mean(np.array(dsi), axis=0)//50)*50, "where":"Away"})])
  sns.violinplot(x='dsi', y='points', data=df, hue="where", split=True, scale="count", inner="stick", scale_hue=False)
  plt.show()


  spd_bw_new = ngb_bwin.staged_pred_dist(X_pred_bwin)
  spd1_new = ngb.staged_pred_dist(X_pred)
  spd2_new = ngb2.staged_pred_dist(X_pred)
  spd3_new = ngb3.staged_pred_dist(X_pred)
  spd4_new = ngb4.staged_pred_dist(X_pred)
  spd5_new = ngb5.staged_pred_dist(X_pred)
  spd6_new = ngb6.staged_pred_dist(X_pred)
  spd8_new = ngb8.staged_pred_dist(X_pred)
  new_preds_per_match = [combine_probs(
                  [spd_bw_new[dsi[0][i]].class_probs(), 
                   spd1_new[dsi[1][i]].class_probs(), 
                   spd2_new[dsi[2][i]].class_probs(), 
                   spd3_new[dsi[3][i]].class_probs(), 
                   spd4_new[dsi[4][i]].class_probs(), 
                   spd5_new[dsi[5][i]].class_probs(), 
                   spd6_new[dsi[6][i]].class_probs(), 
                   spd8_new[dsi[7][i]].class_probs(), 
                   ])[-1]
        for i in range(sample_len)]
  len(new_preds_per_match)      
  np.array(new_preds_per_match).transpose((1,2,0)).shape
  
  
  sp_new_dist = np.matmul(np.array(new_preds_per_match).reshape((-1, 49)), point_matrix)
  new_preds = np.argmax(sp_new_dist, axis=1).reshape((sample_len, -1))
  new_preds_maxsp = np.max(sp_new_dist, axis=1).reshape((sample_len, -1))
  sp_new_dist2 = sp_new_dist.copy()
  sp_new_dist2[range(sp_new_dist2.shape[0]),new_preds.flatten()]=0
  new_preds2 = np.argmax(sp_new_dist2, axis=1).reshape((sample_len, -1))
  new_preds2_maxsp = np.max(sp_new_dist2, axis=1).reshape((sample_len, -1))
  sp_new_dist3 = sp_new_dist2.copy()
  sp_new_dist3[range(sp_new_dist3.shape[0]),new_preds2.flatten()]=0
  new_preds3 = np.argmax(sp_new_dist3, axis=1).reshape((sample_len, -1))
  new_preds3_maxsp = np.max(sp_new_dist3, axis=1).reshape((sample_len, -1))
  for i in range(9):
      print(all_quotes.HomeTeam.iloc[i]+" - "+all_quotes.AwayTeam.iloc[i])
      prednewdfhome = pd.DataFrame({
              "GS":new_preds[:,2*i]//7, "GC":np.mod(new_preds[:,2*i], 7), 
              "SP":new_preds_maxsp[:,2*i], 
              "HomeTeam":all_quotes.HomeTeam.iloc[i],
              "AwayTeam":all_quotes.AwayTeam.iloc[i],
              "points":allpoints,
              "softpoints":allsoftpoints,
              "where":"Home"
              })
      prednewdfaway = pd.DataFrame({
              "GS":np.mod(new_preds[:,2*i+1], 7), "GC":new_preds[:,2*i+1]//7,
              "SP":new_preds_maxsp[:,2*i+1],
              "HomeTeam":all_quotes.AwayTeam.iloc[i],
              "AwayTeam":all_quotes.HomeTeam.iloc[i],
              "points":allpoints,
              "softpoints":allsoftpoints,
              "where":"Away"
              })
      prednewdfhome2 = pd.DataFrame({
              "GS":new_preds2[:,2*i]//7, "GC":np.mod(new_preds2[:,2*i], 7), 
              "SP":new_preds2_maxsp[:,2*i], 
              "HomeTeam":all_quotes.HomeTeam.iloc[i],
              "AwayTeam":all_quotes.AwayTeam.iloc[i],
              "points":allpoints,
              "softpoints":allsoftpoints,
              "where":"Home"
              })
      prednewdfaway2 = pd.DataFrame({
              "GS":np.mod(new_preds2[:,2*i+1], 7), "GC":new_preds2[:,2*i+1]//7,
              "SP":new_preds2_maxsp[:,2*i+1],
              "HomeTeam":all_quotes.AwayTeam.iloc[i],
              "AwayTeam":all_quotes.HomeTeam.iloc[i],
              "points":allpoints,
              "softpoints":allsoftpoints,
              "where":"Away"
              })
      prednewdfhome3 = pd.DataFrame({
              "GS":new_preds3[:,2*i]//7, "GC":np.mod(new_preds3[:,2*i], 7), 
              "SP":new_preds3_maxsp[:,2*i], 
              "HomeTeam":all_quotes.HomeTeam.iloc[i],
              "AwayTeam":all_quotes.AwayTeam.iloc[i],
              "points":allpoints,
              "softpoints":allsoftpoints,
              "where":"Home"
              })
      prednewdfaway3 = pd.DataFrame({
              "GS":np.mod(new_preds3[:,2*i+1], 7), "GC":new_preds3[:,2*i+1]//7,
              "SP":new_preds3_maxsp[:,2*i+1],
              "HomeTeam":all_quotes.AwayTeam.iloc[i],
              "AwayTeam":all_quotes.HomeTeam.iloc[i],
              "points":allpoints,
              "softpoints":allsoftpoints,
              "where":"Away"
              })
      prednewdf = pd.concat([prednewdfhome, prednewdfaway, prednewdfhome2, prednewdfaway2, prednewdfhome3, prednewdfaway3], axis=0)      
      prednewdf.GS = prednewdf.GS.astype(str).str.cat(prednewdf.GC.astype(str), sep=":")
      prednewdf.drop(["GC"], axis=1, inplace=True)
      # sns.scatterplot(prednewdf.softpoints, prednewdf.points)
      # sns.scatterplot(prednewdf.SP, prednewdf.points)
      # sns.scatterplot(prednewdf.softpoints, prednewdf.SP)
      # plt.show()
      fig,ax = plt.subplots(3,1,figsize=(15,9))        
      #plt.figure(figsize=(15,9))
      fig.suptitle(all_quotes.HomeTeam.iloc[i]+" - "+all_quotes.AwayTeam.iloc[i], fontsize=16)
      ax[0].set_title("Softpoints")
      sns.violinplot(y="softpoints",data=prednewdf, x="GS", scale="count", hue="where", split=True, scale_hue=False, ax=ax[0])    
      #plt.show()
      #plt.figure(figsize=(15,3))        
      ax[1].set_title("Points")
      sns.violinplot(y="points",data=prednewdf, x="GS", scale="count", hue="where", split=True, scale_hue=False, ax=ax[1])
      # plt.show()
      # plt.figure(figsize=(15,3))        
      ax[2].set_title("Estimated Softpoints")
      sns.violinplot(y="SP",data=prednewdf, x="GS", scale="count", hue="where", split=True, scale_hue=False, ax=ax[2])
      plt.show()
  

  
  i = 13   
  print(np.round(Y7_pred_new [i].reshape((7,7))*100, 1))
  print(np.round(np.matmul(Y7_pred_new [i:(i+1)], point_matrix).reshape((7,7)), 3))
  
  i = 3  
  print(np.round(Y7_pred_new2 [i].reshape((7,7))*100, 1))
  print(np.round(np.matmul(Y7_pred_new2 [i:(i+1)], point_matrix).reshape((7,7)), 3))

  if point_scheme == point_scheme_pistor:
    point_matrix =  np.array([[3 if (i//7 == j//7) and (np.mod(i, 7)==np.mod(j, 7)) else 
                               2 if (i//7 - np.mod(i, 7) == j//7 - np.mod(j, 7)) else
                               1 if (np.sign(i//7 - np.mod(i, 7)) == np.sign(j//7 - np.mod(j, 7))) else
                               0  for i in range(49)]  for j in range(49)] )
  elif point_scheme == point_scheme_sky:
    point_matrix =  np.array([[5 if (i//7 == j//7) and (np.mod(i, 7)==np.mod(j, 7)) else 
                               2 if (np.sign(i//7 - np.mod(i, 7)) == np.sign(j//7 - np.mod(j, 7))) else
                               0  for i in range(49)]  for j in range(49)] )
  if point_scheme == point_scheme_goal_diff:
    point_matrix =  np.array([[np.maximum(5, 10 - np.abs(i//7 - j//7) - np.abs(np.mod(i, 7) - np.mod(j, 7))) if (np.sign(i//7 - np.mod(i, 7)) == np.sign(j//7 - np.mod(j, 7))) else
                               0  for i in range(49)]  for j in range(49)] )

#  
#  print(np.mean(calc_softpoints(Y3_test, Y3_2_pred)))
#  print(beautify(Counter(invert(X_test[:,1], argmax_softpoint(Y3_2_pred)))))
#  print(np.mean(calc_points(Y3_test, argmax_softpoint(Y3_2_pred))))
#  print(point_dist(calc_points(Y3_test, argmax_softpoint(Y3_2_pred))))
#
#  Y3_1_pred[0].reshape((7,7))
#  np.mean(np.sum(Y3_1_pred, axis=1) )
#  Y3_5_pred[0].reshape((7,7))
#  np.mean(np.sum(Y3_2_pred, axis=1) )
#  np.mean(np.sum(Y3_4_pred, axis=1) )
#  np.mean(np.sum(Y3_5_pred, axis=1) )
#  np.mean(np.sum(Y3_6_pred, axis=1) )
#  np.mean(np.sum(Y3_pred, axis=1) )
#  np.mean(np.sum(Y1_pred, axis=1) )
#  np.mean(np.sum(Y2_pred, axis=1) )
#  np.mean(np.sum(Y4_pred, axis=1) )
#
#  Y1_pred[0]
#  
#  print(np.mean(calc_softpoints(Y3_test, Y3_1_pred)))
#  print(beautify(Counter(invert(X_test[:,1], argmax_softpoint(Y3_1_pred)))))
#  print(np.mean(calc_points(Y3_test, argmax_softpoint(Y3_1_pred))))
#  print(point_dist(calc_points(Y3_test, argmax_softpoint(Y3_1_pred))))
#
#  print(np.mean(calc_softpoints(Y3_test, Y3_2_pred)))
#  print(beautify(Counter(invert(X_test[:,1], argmax_softpoint(Y3_2_pred)))))
#  print(np.mean(calc_points(Y3_test, argmax_softpoint(Y3_2_pred))))
#  print(point_dist(calc_points(Y3_test, argmax_softpoint(Y3_2_pred))))
#
#  print(np.mean(calc_softpoints(Y3_test, Y3_4_pred)))
#  print(beautify(Counter(invert(X_test[:,1], argmax_softpoint(Y3_4_pred)))))
#  print(np.mean(calc_points(Y3_test, argmax_softpoint(Y3_4_pred))))
#  print(point_dist(calc_points(Y3_test, argmax_softpoint(Y3_4_pred))))


#
#
#
#X, Y = load_boston(True)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#
#ngb = NGBRegressor().fit(X_train, Y_train)
#Y_preds = ngb.predict(X_test)
#Y_dists = ngb.pred_dist(X_test)
#
#
## test Mean Squared Error
#test_MSE = mean_squared_error(Y_preds, Y_test)
#print('Test MSE', test_MSE)
#
## test Negative Log Likelihood
#test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
#print('Test NLL', test_NLL)
#
#Y_preds_train = ngb.predict(X_train)
#Y_dists_train = ngb.pred_dist(X_train)
#
## test Mean Squared Error
#train_MSE = mean_squared_error(Y_preds_train, Y_train)
#print('Train MSE', train_MSE)
#
## test Negative Log Likelihood
#train_NLL = -Y_dists_train.logpdf(Y_train.flatten()).mean()
#print('Train NLL', train_NLL)
#
#
