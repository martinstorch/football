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

import pandas as pd
pd.set_option('expand_frame_repr', False)

import numpy as np
#np.set_printoptions(threshold=50)

from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import matplotlib.markers as markers
#from Estimators import LinearModel as lm
#from Estimators import PoissonModel_teamcentric as pm
#from Estimators import DiscreteModel as dm
#from Estimators import DiscreteModelMulti as dmm
#from Estimators import DiscreteLayeredModel as dlm
#from Estimators import DiscreteRNNModel as drm
#from Estimators import LSTMModel as lstm
#from Estimators import LSTM_m21_Model as lstm_m21
from Estimators import GRU4_Model as gru
from Estimators import Utilities as utils

#from tensorflow.python.training.session_run_hook import SessionRunHook
#from tensorflow.contrib.layers import l2_regularizer

from collections import Counter
#from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import random
import itertools

Feature_COLUMNS = ["HomeTeam","AwayTeam"]
Label_COLUMNS = ["FTHG","FTAG"]
CSV_COLUMNS = Feature_COLUMNS + Label_COLUMNS
Derived_COLUMNS = ["t1goals", "t2goals", "t1goals_where", "t2goals_where"]
COLS = ["HGFT","AGFT","HGHT","AGHT","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR"]
Meta_COLUMNS = ["t1games", "t2games", "t1games_where", "t2games_where"]
COLS_Extended = COLS + ['HWin', 'AWin', 'HLoss', 'ALoss', 'HDraw', 'ADraw']

SEQ_LENGTH = 10
TIMESERIES_COL = 'rawdata'

DEVELOPER_MODE = False

def download_data(model_dir, season, skip_download):
    """Maybe downloads training data and returns train and test file names."""
    file_name = model_dir + "/" + season + ".csv"
    if (not skip_download):    
  #    urllib.request.urlretrieve(
  #        "http://217.160.223.109/mmz4281/"+season+"/D1.csv",
  #        file_name)  # pylint: disable=line-too-long
      
      url = "http://www.football-data.co.uk/mmz4281/"+season+"/D1.csv"
      print("Downloading %s" % url)
      urllib.request.urlretrieve(
          url,
          file_name)  # pylint: disable=line-too-long
      print("Data is downloaded to %s" % file_name)
    data = pd.read_csv(
      tf.gfile.Open(file_name),
      skipinitialspace=True,
      engine="python",
      skiprows=0)
    data["Season"]= season
    return data

def replace_teamnames(data, season):
  def _replace_teamnames(relegated_season, teamname):
    if season <= relegated_season:
      data.loc[data["HomeTeam"]==teamname,"HomeTeam"]=teamname+"_"+relegated_season
      data.loc[data["AwayTeam"]==teamname,"AwayTeam"]=teamname+"_"+relegated_season
  _replace_teamnames("1415", "Freiburg")  
  _replace_teamnames("1516", "Hannover")  
  _replace_teamnames("1516", "Stuttgart")  
  return(data)

def get_train_test_data(model_dir, train_seasons, test_seasons, skip_download):
  train_data = []
  for s in train_seasons:
    newdata = download_data(model_dir, s, skip_download) 
    replace_teamnames(newdata, s)
    train_data.append(newdata)
  train_data = pd.concat(train_data, ignore_index=True)
  
  test_data = []
  for s in test_seasons:
    test_data.append(replace_teamnames(download_data(model_dir, s, skip_download), s) )
  test_data = pd.concat(test_data, ignore_index=True)

  new_data =  pd.read_csv(
      tf.gfile.Open(model_dir + "/NewGames.csv"),
      skipinitialspace=True,
      engine="python",
      skiprows=0)
  new_data["Season"]= "1718"

  print(train_data.shape)  
  print(test_data.shape)  
  print(new_data.shape)  
  teamnames = [] 
  teamnames.extend(train_data["HomeTeam"].tolist())
  teamnames.extend(train_data["AwayTeam"].tolist())
  teamnames.extend(test_data["HomeTeam"].tolist())
  teamnames.extend(test_data["AwayTeam"].tolist())
  teamnames.extend(new_data["HomeTeam"].tolist())
  teamnames.extend(new_data["AwayTeam"].tolist())
  print(Counter(teamnames))
  teamnames = np.unique(teamnames).tolist()
  train_data["Train"]=True
  train_data["Predict"]=False
  test_data["Train"]=False
  test_data["Predict"]=False
  new_data["Predict"]=True
  new_data["Train"]=False

  if DEVELOPER_MODE:
    train_data = train_data[0:9]
    test_data = test_data[0:9]
    
  all_data = pd.concat([train_data, test_data, new_data], ignore_index=True)
  all_data = all_data.fillna(0)
  
  return all_data, teamnames



def build_features(df_data, teamnames, mode=tf.estimator.ModeKeys.TRAIN):
#  if mode==tf.estimator.ModeKeys.TRAIN:
#    df_data = df_data[df_data["Train"]==True].copy()
  
  print("Build features - {}".format(mode))
  print(df_data.shape) 
  
  team_encoder = preprocessing.LabelEncoder()
  team_encoder.fit(teamnames)

  df_data.rename(columns={
      'FTHG': 'HGFT', 'FTAG': 'AGFT',
      'HTHG': 'HGHT', 'HTAG': 'AGHT'
      }, inplace=True)

  df1 = pd.DataFrame()
  df1["Team1"] = df_data["HomeTeam"]
  df1["Team2"] = df_data["AwayTeam"]
  df1["Where"] = 1
  df1['OpponentGoals'] = df_data["AGFT"]
  df1['OwnGoals'] = df_data["HGFT"]
  df1['HomeTeam'] = df1["Team1"]
  df1['Season'] = df_data["Season"]
  df1["Train"] = df_data["Train"]
  df1["Date"]= df_data["Date"]
  #df1["zGameFinalScore"] = [str(int(hg))+":"+str(int(ag)) for hg,ag in zip(df_data["HGFT"], df_data["AGFT"])]
    
  df2 = pd.DataFrame()
  df2["Team1"] = df_data["AwayTeam"]
  df2["Team2"] = df_data["HomeTeam"]
  df2["Where"] = 0
  df2['OpponentGoals'] = df_data["HGFT"]
  df2['OwnGoals'] = df_data["AGFT"]
  df2['HomeTeam'] = df1["Team2"]
  df2['Season'] = df_data["Season"]
  df2["Train"] = df_data["Train"]
  df2["Date"]= df_data["Date"]
  #df2["zGameFinalScore"] = [str(int(hg))+":"+str(int(ag)) for hg,ag in zip(df_data["HGFT"], df_data["AGFT"])]
  
  columns = [c[1:] for c in COLS[::2]]
#  feature_column_names_fixed = ["Team1", "Team2", "Where", "Season", "Train"]
  label_column_names = []
  for colname in columns:
    homecol="H"+colname
    awaycol="A"+colname
    df1["T1_"+colname] = df_data[homecol].astype(int)
    df1["T2_"+colname] = df_data[awaycol].astype(int)
    df2["T1_"+colname] = df_data[awaycol].astype(int)
    df2["T2_"+colname] = df_data[homecol].astype(int)
    label_column_names += ["T1_"+colname, "T2_"+colname]

  lb1 = pd.DataFrame()
  lb1['Goals'] = df_data["HGFT"]
  lb2 = pd.DataFrame()
  lb2['Goals'] = df_data["AGFT"]
  lb1.index=lb1.index*2
  lb2.index=lb1.index+1
  labels = pd.concat([lb1,lb2], ignore_index=False)
  labels = labels.sort_index()
  
  df1.index=df1.index*2
  df2.index=df1.index+1
  features = pd.concat([df1,df2], ignore_index=False )
  features = features.sort_index()
  
  # derived feature 2nd half goals
  features["T1_GH2"] = features["T1_GFT"] - features["T1_GHT"]
  features["T2_GH2"] = features["T2_GFT"] - features["T2_GHT"]
  label_column_names += ["T1_GH2", "T2_GH2"]
  columns += ["GH2"]

  # derived feature full time win/loss/draw
  features["zGameResult"] = [np.sign(x1-x2) for x1,x2 in zip(features["T1_GFT"], features["T2_GFT"])]
  features["zGameHTResult"] = [np.sign(x1-x2) for x1,x2 in zip(features["T1_GHT"], features["T2_GHT"])]
  features["zGameFinalScore"] = [str(x1)+":"+str(x2) for x1,x2 in zip(features["T1_GFT"], features["T2_GFT"])]
  
  gr = pd.get_dummies(features["zGameResult"])
  col_names = ['Loss', 'Draw', 'Win']
  gr.columns = col_names
  features = pd.concat([features, gr], axis=1)
  label_column_names += col_names

  # derived feature half-time win/loss/draw
  gr2 = pd.get_dummies(features["zGameHTResult"])
  col_names = ['HTLoss', 'HTDraw', 'HTWin']
  gr2.columns = col_names
  features = pd.concat([features, gr2], axis=1)
  label_column_names += col_names
  
  features["Team1_index"] = team_encoder.transform(features["Team1"])
  features["Team2_index"] = team_encoder.transform(features["Team2"])
  

  final_score_enum = ["0:3", "1:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0", "2:2", "3:2", "2:3"]
  final_score_encoder = preprocessing.LabelBinarizer(sparse_output=False)
  final_score_encoder.fit_transform(final_score_enum)
  label_column_names += ["zScore"+s for s in final_score_enum]
  print(Counter(features["zGameFinalScore"]))
  for s in final_score_enum :
    features["zScore"+s] =  [1 if r==s else 0 for r in features["zGameFinalScore"]]  
  
  gt1 = features.groupby(["Season", "Team1"])
  gt2 = features.groupby(["Season", "Team2"])
  gt1w = features.groupby(["Season", "Team1","Where"])
  gt2w = features.groupby(["Season", "Team2","Where"])
  
  features["t1games"] = gt1.cumcount()
  features["t2games"] = gt2.cumcount()
  features["t1goals"] = (gt1["OwnGoals"].cumsum()-features["OwnGoals"])/(features["t1games"]+2)
  features["t2goals"] = (gt2["OpponentGoals"].cumsum()-features["OpponentGoals"])/(features["t2games"]+2)
  features["t12goals"] = features["t1goals"] - features["t2goals"]
  features.loc[features["t1games"]==0, "t1goals"] = 1.45
  features.loc[features["t2games"]==0, "t2goals"] = 1.45
  
  features["t1games_where"] = gt1w.cumcount()
  features["t2games_where"] = gt2w.cumcount()

  feature_column_names = []
  for colname in columns:
    features["T1_CUM_T1_"+colname] = (gt1["T1_"+colname].cumsum()-features["T1_"+colname])/(features["t1games"]+2)
    features["T2_CUM_T2_"+colname] = (gt2["T2_"+colname].cumsum()-features["T2_"+colname])/(features["t2games"]+2)
    features["T1_CUM_T1_W_"+colname] = (gt1w["T1_"+colname].cumsum()-features["T1_"+colname])/(features["t1games_where"]+1)
    features["T2_CUM_T2_W_"+colname] = (gt2w["T2_"+colname].cumsum()-features["T2_"+colname])/(features["t2games_where"]+1)
    features["T1_CUM_T2_"+colname] = (gt2["T1_"+colname].cumsum()-features["T1_"+colname])/(features["t1games"]+2)
    features["T2_CUM_T1_"+colname] = (gt1["T2_"+colname].cumsum()-features["T2_"+colname])/(features["t2games"]+2)
    features["T1_CUM_T2_W_"+colname] = (gt2w["T1_"+colname].cumsum()-features["T1_"+colname])/(features["t1games_where"]+1)
    features["T2_CUM_T1_W_"+colname] = (gt1w["T2_"+colname].cumsum()-features["T2_"+colname])/(features["t2games_where"]+1)
    features["T12_CUM_T1_"+colname] = features["T1_CUM_T1_"+colname] - features["T2_CUM_T1_"+colname]
    features["T12_CUM_T1_W_"+colname] = features["T1_CUM_T1_W_"+colname] - features["T2_CUM_T1_W_"+colname]
    features["T21_CUM_T2_"+colname] = features["T2_CUM_T2_"+colname] - features["T1_CUM_T2_"+colname]
    features["T21_CUM_T2_W_"+colname] = features["T2_CUM_T2_W_"+colname] - features["T1_CUM_T2_W_"+colname]
    features["T12_CUM_T12_"+colname] = features["T1_CUM_T1_"+colname] - features["T2_CUM_T2_"+colname]
    features["T12_CUM_T12_W_"+colname] = features["T1_CUM_T1_W_"+colname] - features["T2_CUM_T2_W_"+colname]
    features["T1221_CUM_"+colname] = features["T12_CUM_T1_"+colname] - features["T21_CUM_T2_"+colname]
    features["T1221_CUM_W_"+colname] = features["T12_CUM_T1_W_"+colname] - features["T21_CUM_T2_W_"+colname]
    features.loc[features["t1games"]==0, "T1_CUM_T1_"+colname] = features.loc[features["t1games"]>0, "T1_CUM_T1_"+colname].mean()
    features.loc[features["t2games"]==0, "T2_CUM_T2_"+colname] = features.loc[features["t2games"]>0, "T2_CUM_T2_"+colname].mean()
    features.loc[features["t1games_where"]==0, "T1_CUM_T1_W_"+colname] = features.loc[features["t1games_where"]>0, "T1_CUM_T1_W_"+colname].mean()
    features.loc[features["t2games_where"]==0, "T2_CUM_T2_W_"+colname] = features.loc[features["t2games_where"]>0, "T2_CUM_T2_W_"+colname].mean()
    features.loc[features["t1games"]==0, "T1_CUM_T2_"+colname] = features.loc[features["t1games"]>0, "T1_CUM_T2_"+colname].mean()
    features.loc[features["t2games"]==0, "T2_CUM_T1_"+colname] = features.loc[features["t2games"]>0, "T2_CUM_T1_"+colname].mean()
    features.loc[features["t1games_where"]==0, "T1_CUM_T2_W_"+colname] = features.loc[features["t1games_where"]>0, "T1_CUM_T2_W_"+colname].mean()
    features.loc[features["t2games_where"]==0, "T2_CUM_T1_W_"+colname] = features.loc[features["t2games_where"]>0, "T2_CUM_T1_W_"+colname].mean()
    feature_column_names += ["T1_CUM_T1_"+colname, "T2_CUM_T2_"+colname,
                             "T1_CUM_T1_W_"+colname, "T2_CUM_T2_W_"+colname,
                             "T1_CUM_T2_"+colname, "T2_CUM_T1_"+colname,
                             "T1_CUM_T2_W_"+colname, "T2_CUM_T1_W_"+colname,
                             "T12_CUM_T1_"+colname, "T12_CUM_T1_W_"+colname,
                             "T21_CUM_T2_"+colname, "T21_CUM_T2_W_"+colname,
                             "T12_CUM_T12_"+colname, "T12_CUM_T12_W_"+colname,
                             "T1221_CUM_"+colname, "T1221_CUM_W_"+colname]
    
  for colname in ['Win', 'HTWin', 'Loss', 'HTLoss', 'Draw', 'HTDraw']:
    features["T1_CUM_T1_"+colname] = (gt1[colname].cumsum()-features[colname])/(features["t1games"]+2)
    features["T2_CUM_T2_"+colname] = (gt2[colname].cumsum()-features[colname])/(features["t2games"]+2)
    features["T1_CUM_T1_W_"+colname] = (gt1w[colname].cumsum()-features[colname])/(features["t1games_where"]+1)
    features["T2_CUM_T2_W_"+colname] = (gt2w[colname].cumsum()-features[colname])/(features["t2games_where"]+1)
    features.loc[features["t1games"]==0, "T1_CUM_T1_"+colname] = features.loc[features["t1games"]>0, "T1_CUM_T1_"+colname].mean()
    features.loc[features["t2games"]==0, "T2_CUM_T2_"+colname] = features.loc[features["t2games"]>0, "T2_CUM_T2_"+colname].mean()
    features.loc[features["t1games_where"]==0, "T1_CUM_T1_W_"+colname] = features.loc[features["t1games_where"]>0, "T1_CUM_T1_W_"+colname].mean()
    features.loc[features["t2games_where"]==0, "T2_CUM_T2_W_"+colname] = features.loc[features["t2games_where"]>0, "T2_CUM_T2_W_"+colname].mean()
    feature_column_names += ["T1_CUM_T1_"+colname, "T2_CUM_T2_"+colname,
                             "T1_CUM_T1_W_"+colname, "T2_CUM_T2_W_"+colname]


  batches1 = [features[features["Team1"]==t].copy() for t in teamnames]
  batches2 = [features[features["Team2"]==t].copy() for t in teamnames]
  
  print(label_column_names) 
  print(feature_column_names) 
  print(features[label_column_names].mean()) 
  
  team_onehot_encoder = preprocessing.LabelBinarizer(sparse_output=False)
  team_onehot_encoder .fit_transform(teamnames)

  steps = 10
  tn = len(teamnames)
  lc = len(label_column_names)
  fc = len(feature_column_names)
  newgame = np.ndarray(shape=[len(features), 4+2*tn+fc], dtype=np.float32)
  labels = np.ndarray(shape=[len(features), lc], dtype=np.float32)
  match_history_t1 = np.ndarray(shape=[len(features), steps, 2+2*tn+lc+fc], dtype=np.float32)
  match_history_t2 = np.ndarray(shape=[len(features), steps, 2+2*tn+lc+fc], dtype=np.float32)
  match_history_t12 = np.ndarray(shape=[len(features), steps, 2+2*tn+lc+fc], dtype=np.float32)

  def build_history_data(match_history, teamdata, index):
    teamdata = teamdata[teamdata.index < index] # include only matches from the past
    teamdata = teamdata[-steps:] # include up to "steps" previous matches if available
    seq_len = len(teamdata)
    j = 0
    match_history[index, :, j] = seq_len 
    if seq_len>0:
      # Features
      j = j+1
      match_history[index, :seq_len, j] = teamdata["Where"]
      j = j+1
      match_history[index, :seq_len, j:j+tn] = team_onehot_encoder.transform(teamdata["Team1"])
      j = j+tn
      match_history[index, :seq_len, j:j+tn] = team_onehot_encoder.transform(teamdata["Team2"])
      j = j+tn
      match_history[index, :seq_len, j:j+lc] = teamdata [label_column_names]
      j = j+lc
      match_history[index, :seq_len, j:j+fc] = teamdata [feature_column_names]
      
    return seq_len
  features_by_teams = features.copy()
  features_by_teams["gameindex"]=features_by_teams.index
  features_by_teams.set_index(["Team1", "Team2","gameindex"], inplace=True)
#  print(features_by_teams.loc["Hannover", "Augsburg",:])
#  print(features_by_teams.loc["Hannover", "Augsburg",:].columns)
#  print(features_by_teams.loc["Hannover", "Augsburg",:].index)
  for index, f in features.iterrows():
    team1 = f["Team1_index"]
    team2 = f["Team2_index"]
    new_label = f[label_column_names]
    labels[index,:] = new_label
    # 
    t1_hist_len = build_history_data(match_history_t1, batches1[team1], index)
    t2_hist_len = build_history_data(match_history_t2, batches2[team2], index)
    batches12 = features_by_teams.loc[f["Team1"],f["Team2"],:].copy()
    batches12["Team1"]= f["Team1"]
    batches12["Team2"]= f["Team2"]
    t12_hist_len = build_history_data(match_history_t12, batches12, index)
      
    newgame[index,0] = t1_hist_len
    newgame[index,1] = t2_hist_len
    newgame[index,2] = f["Where"]
    newgame[index,3] = t12_hist_len
    newgame[index, 4:4+tn] = team_onehot_encoder.transform([f["Team1"]])
    newgame[index, 4+tn:4+2*tn] = team_onehot_encoder.transform([f["Team2"]])
    newgame[index, 4+2*tn:4+2*tn+fc] = f[feature_column_names]
  
  
#  print(labels[0:1,:])
#  print(labels.shape)
#  print(match_history_t1[0,:,:])
#  print(match_history_t2[600,:,:])
#  print(match_history_t1.shape)
#  print(newgame.shape)
  
  #output.tofile("features.txt", format="%d")
  #labels.dump("labels.txt")
  print("t1_hist_len: {}".format(Counter(newgame[:,0])))
  print("t2_hist_len: {}".format(Counter(newgame[:,1])))
  print("t12_hist_len: {}".format(Counter(newgame[:,3])))
  return {
      "newgame": newgame,
      "match_history_t1": match_history_t1,
      "match_history_t2": match_history_t2,
      "match_history_t12": match_history_t12,
      }, labels, team_onehot_encoder, label_column_names

def input_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN):
  print(mode)
#  print({k:v.shape for k,v in features.items()})
#  print("Labels: {}".format(labels.shape))
  return tf.estimator.inputs.numpy_input_fn(
    x=features,
    y=labels,
    batch_size=len(labels),
    num_epochs=None if mode==tf.estimator.ModeKeys.TRAIN else 1,
    shuffle= (mode==tf.estimator.ModeKeys.TRAIN),
    num_threads=1
    )

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
  spt = pred[prefix+"ev_points"]
  gs = min(gs,6)
  gc = min(gc,6)
  margin_pred_prob1 = pred[prefix+"p_marg_1"]
  margin_poisson_prob1 = pred[prefix+"p_poisson_1"]
  margin_pred_prob2 = pred[prefix+"p_marg_2"]
  margin_poisson_prob2 = pred[prefix+"p_poisson_2"]
  margin_pred_expected1 = pred[prefix+"ev_goals_1"] 
  margin_pred_expected2 = pred[prefix+"ev_goals_2"] 
  g=[0,1,2,3,4,5,6]
  g1=[0]*7+[1]*7+[2]*7+[3]*7+[4]*7+[5]*7+[6]*7
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
  ax[0].scatter(max_sp_index//7, np.mod(max_sp_index, 7), s=max_sp*10000, facecolors='none', edgecolors='black', linewidth='2')
  max_spt = max(spt)
  max_spt_index = np.argmax(spt) 
  ax[1].scatter(max_spt_index//7, np.mod(max_spt_index, 7), s=max_spt*500, facecolors='none', edgecolors='black', linewidth='2')
  
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


def plot_predictions_2(predictions, features, labels, team_onehot_encoder, prefix="sp/", is_prediction=False, dataset = "Test", skip_plotting=False):

  features = features["newgame"]
  
  features = features[:len(predictions)] # cut off features if not enough predictions are present
  labels = labels[:len(predictions)] # cut off labels if not enough predictions are present
  
  default_color = "blue"
  default_cmap=plt.cm.Blues
  if prefix=="p1/" or prefix=="p3/":
    default_color = "darkolivegreen"
    default_cmap=plt.cm.Greens
  if prefix=="p2/" or prefix=="p4/":
    default_color = "darkmagenta"
    default_cmap=plt.cm.Purples # RdPu # PuRd
  
  if prefix=="ens/": # only home games
    features = features[::2]
    labels = labels[::2]
    predictions = predictions[::2]
  
  # sample input and output values for printing
  predictions[0].update({"features": features[0,:]})
  predictions[1].update({"features": features[1,:]})
  predictions[0].update({"labels": labels[0,:]})
  predictions[1].update({"labels": labels[1,:]})
  
#  print(predictions[0:2])

  df = pd.DataFrame()  
  df["GS"] = labels[:,0].astype(np.int)
  df["GC"] = labels[:,1].astype(np.int)
  #print(len(df["GS"]))
  #print(len([p[prefix+"pred"][0] for p in predictions]))
  df['pGS'] = [p[prefix+"pred"][0] for p in predictions]
  df['pGC'] = [p[prefix+"pred"][1] for p in predictions]

  if prefix!="ens/":
    est1 = pd.Series([p[prefix+"ev_goals_1"] for p in predictions], name="est1")
    est2 = pd.Series([p[prefix+"ev_goals_2"] for p in predictions], name="est2")
    df['est1'] = est1
    df['est2'] = est2
  else:
    strategy_list = ["p1", "p2","p3","p4","p5","p6","sp","sm","smhb"]
    strategy_list = [s+" home" for s in strategy_list] + [s+" away" for s in strategy_list]
    strategy_index = [p[prefix+"selected_strategy"] for p in predictions]
    df['Strategy'] = [strategy_list[ind] for ind in strategy_index]
    print(Counter(df['Strategy']))
    
  print(team_onehot_encoder.classes_)
  tn = len(team_onehot_encoder.classes_)
  df['Team1']=team_onehot_encoder.inverse_transform(features[:, 4:4+tn])
  df['Team2']=team_onehot_encoder.inverse_transform(features[:, 4+tn:4+2*tn])
  df['Where']=['Home' if h==1 else 'Away' for h in features[:, 2]]
  df["act"]  = [str(gs)+':'+str(gc) for gs,gc in zip(df["GS"],df["GC"]) ]
  df["pred"] = [str(gs)+':'+str(gc) for gs,gc in zip(df["pGS"],df["pGC"]) ]
  
  tensor = tf.constant(df[[ "pGS", "pGC", "GS", "GC"]].as_matrix(), dtype = tf.int64)
  is_home = tf.equal(features[:,2] , 1)
  with tf.Session() as sess:
    points_tensor = gru.calc_points(tensor[:,0],tensor[:,1], tensor[:,2], tensor[:,3], is_home)[0]
    df['Pt'] = sess.run(tf.cast(points_tensor, tf.int8))
    df['Prefix'] = prefix[:-1]
    print()
    print("{}({}):{} ".format(prefix[:-1], dataset, np.sum(df['Pt'])/len(df)))

  def preparePrintData(prefix, df):
    if prefix!="ens/":
      return df[["Team1", "Team2", "act", "pred", "Where", "est1","est2","Pt", "Prefix"]]
    else:
      return df[["Team1", "Team2", "act", "pred", "Pt", "Prefix", "Strategy"]]
      
  print(preparePrintData(prefix, df).head(80))
    
  if skip_plotting:
    return df
  if not is_prediction:
    print(preparePrintData(prefix, df).tail(200))
  if is_prediction:
    for s in range(len(df)):
      plot_softprob(predictions[s], df["GS"][s], df["GC"][s], df["Team1"][s]+" - "+df["Team2"][s]+" ("+df["Where"][s]+")", prefix=prefix)
    return
  else:
    s = random.sample(range(len(df)), 1)[0]
    plot_softprob(predictions[s], df["GS"][s], df["GC"][s], df["Team1"][s]+" - "+df["Team2"][s]+" ("+df["Where"][s]+")", prefix=prefix)
    s = random.sample(range(len(df)), 1)[0]
    plot_softprob(predictions[s], df["GS"][s], df["GC"][s], df["Team1"][s]+" - "+df["Team2"][s]+" ("+df["Where"][s]+")", prefix=prefix)

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
  df["total_points"]+=[6 if t==0 and gof=="3_full" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[2 if t==0 and gof=="2_diff" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[2 if t==0 and gof=="1_tendency" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[2 if t==1 and gof=="1_tendency" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[3 if t==1 and gof=="2_diff" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[4 if t==1 and gof=="3_full" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[4 if t==-1 and gof=="1_tendency" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[5 if t==-1 and gof=="2_diff" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]
  df["total_points"]+=[7 if t==-1 and gof=="3_full" else 0 for t, gof in zip(np.sign(df["pFTHG"]-df["pFTAG"]), df["gof"])]

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

  t1 = df.pivot_table(index = [df["sort_idx"], 'pGoals'], columns=['gof'], values=["Team1"], aggfunc=len, margins=False, fill_value=0)
  t1.columns = ["None", "Tendency", "Diff", "Full"][:len(t1.columns)]
  t1.index = t1.index.droplevel(level=0)
  t1.plot(kind='bar', stacked=True, ax=ax1)
  ax1.set_title(prefix[:-1]).set_size(12)
  
  _,_,autotexts = ax2.pie(pie_chart_values, 
          labels=["None", "Tendency", "Diff", "Full"], autopct='%1.1f%%', startangle=90)
  for t in autotexts:
    t.set_color("white")
  ax2.set_title("{}: {:.04f} ({})".format(prefix[:-1], avg_points, dataset)).set_size(20)

  percentages = [pie_chart_values[0], # None
                 pie_chart_values[1]+pie_chart_values[2]+pie_chart_values[3], #Tendency
                 pie_chart_values[2]+pie_chart_values[3], #GDiff
                 pie_chart_values[3], #Full
                 ]    
  for t,p in zip(autotexts, percentages):
    t.set_text("{:.01f}%".format(100.0 * p / len(labels)))
  
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
      100.0 * tendency_values[2]/len(labels),
      100.0 * tendency_values[1]/len(labels),
      100.0 * tendency_values[0]/len(labels)
    ))

  plt.show()
  plt.close()

  print()
  print("Points: {0:.4f}, Tendency: {1:.2f}, Diff: {2:.2f}, Full: {3:.2f},    Home: {4:.1f}, Draw: {5:.1f}, Away: {6:.1f}".format(
      avg_points,
      100.0 * (1-pie_chart_values[0]/len(labels)),
      100.0 * (pie_chart_values[2]+pie_chart_values[3])/len(labels),
      100.0 * pie_chart_values[3]/len(labels),
      100.0 * tendency_values[2]/len(labels),
      100.0 * tendency_values[1]/len(labels),
      100.0 * tendency_values[0]/len(labels)
    ))
  
  c_home = df["Where"]=="Home"
  c_win  = df['pGS'] > df['pGC']
  c_loss = df['pGS'] < df['pGC']
  c_draw = df['pGS'] == df['pGC']
  c_tendency = np.sign(df['pGS']- df['pGC']) == np.sign(df["GS"] - df["GC"]) 
 
  def createTitle(series1, series2):
    return "pearson: {:.4f}, spearman: {:.4f}".format(
        series1.corr(series2, method="pearson"), 
        series1.corr(series2, method="spearman") 
        )

  if prefix!="ens/":
  
      fig, ax = plt.subplots(1,2,figsize=(12,4))
      ax[0].set_title(createTitle(df["est1"], df["GS"]))
      ax[0].scatter(df[~c_home]["est1"], df[~c_home]["GS"], alpha=0.1, color="red")
      ax[0].scatter(df[c_home]["est1"], df[c_home]["GS"], alpha=0.1, color=default_color)
      ax[1].set_title(createTitle(df["est2"], df["GC"]))
      ax[1].scatter(df[~c_home]["est2"], df[~c_home]["GC"], alpha=0.1, color="red")
      ax[1].scatter(df[c_home]["est2"], df[c_home]["GC"], alpha=0.1, color=default_color)
      plt.show()
      fig, ax = plt.subplots(1,2,figsize=(12,4))
      ax[0].set_title(createTitle(df["est1"]-df["est2"], df["GS"]-df["GC"]))
      ax[0].scatter(df[~c_home]["est1"]-df[~c_home]["est2"], df[~c_home]["GS"]-df[~c_home]["GC"],alpha=0.1, color="red")
      ax[0].scatter(df[c_home]["est1"]-df[c_home]["est2"], df[c_home]["GS"]-df[c_home]["GC"],alpha=0.1, color=default_color)
    
    #  fig, ax = plt.subplots(1,2,figsize=(12,4))
    #  ax[0].scatter(est1.loc[c_home], df[c_home]["GS"], alpha=0.1, color="blue")
    #  ax[0].scatter(est1.loc[~c_home], df[~c_home]["GS"], alpha=0.1, color="red")
    #  ax[1].scatter(est2.loc[c_home], df[c_home]["GC"], alpha=0.1, color="blue")
    #  ax[1].scatter(est2.loc[~c_home], df[~c_home]["GC"], alpha=0.1, color="red")
    #  plt.show()
    #  fig, ax = plt.subplots(1,2,figsize=(12,4))
    #  ax[0].scatter(est_diff.loc[c_home], df[c_home]["GS"]-df[c_home]["GC"],alpha=0.1, color="blue")
    #  ax[0].scatter(est_diff.loc[~c_home], df[~c_home]["GS"]-df[~c_home]["GC"],alpha=0.1, color="red")
    
      def plotEstimates(cond1, cond2, color, alpha=0.1):
        ax[1].scatter(est1[cond1 & c_tendency], est2[cond1 & c_tendency],alpha=alpha,color=color, marker='o')
        ax[1].scatter(est1[cond1 & ~c_tendency], est2[cond1 & ~c_tendency],alpha=alpha,color=color, marker='x')
        ax[1].scatter(est2[cond2 & c_tendency], est1[cond2 & c_tendency],alpha=alpha,color=color, marker='o')
        ax[1].scatter(est2[cond2 & ~c_tendency], est1[cond2 & ~c_tendency],alpha=alpha,color=color, marker='x')
      
      plotEstimates(c_win & c_home, c_loss & ~c_home, "blue")
      plotEstimates(c_loss & c_home, c_win & ~c_home, "red")
      plotEstimates(c_draw, c_draw, "green", 0.3)
      ax[1].set_title("Points: {:.4f} ({:.2f}%), H: {:.1f}, D: {:.1f}, A: {:.1f}".format(
          np.sum(t2["Contribution"]),
          100.0 * (1-pie_chart_values[0]/len(labels)),
          100.0 * tendency_values[2]/len(labels),
          100.0 * tendency_values[1]/len(labels),
          100.0 * tendency_values[0]/len(labels)
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

  return df

def prepare_label_fit(predictions, features, labels, team_onehot_encoder, label_column_names):                             
  features = features["newgame"]
  
  features = features[:len(predictions)] # cut off features if not enough predictions are present
  labels = labels[:len(predictions)] # cut off labels if not enough predictions are present

  tn = len(team_onehot_encoder.classes_)
  df = pd.DataFrame()
  df['Team1']=team_onehot_encoder.inverse_transform(features[:, 4:4+tn])
  df['Team2']=team_onehot_encoder.inverse_transform(features[:, 4+tn:4+2*tn])
  df['Where']=['Home' if h==1 else 'Away' for h in features[:, 2]]

#  df = pd.DataFrame().from_csv("file:///C:/tmp/Football/models_multi_2017/train_outputs_poisson.csv")
#  tn=36
#  label_column_names = df.columns[4::2]
  #print(label_column_names)
  fig, ax = plt.subplots(1+(len(label_column_names)//4), 4, figsize=(20,70))
  for i,col in enumerate(label_column_names):
    #print(i)
    df["p_"+col]=[np.exp(p["outputs_poisson"][i]) for p in predictions]
    df[col]=labels[:,i]
    ax0 = ax[i//4,np.mod(i,4)]
    df.boxplot(column = "p_"+col, 
               by=col, ax=ax0, fontsize=10, grid=False)
    dfcorr = df[["p_"+col, col]]
    cor_p = dfcorr[col].corr(dfcorr["p_"+col], method='pearson') 
    cor_s = dfcorr[col].corr(dfcorr["p_"+col], method='spearman') 
#    print(col)
#    print(cor_p)
#    print(cor_s)
    ax0.set_title('cor_p={:.4f}, cor_s={:.4f}'.format(cor_p,cor_s))
  
  #plt.show()  
  #fig.savefig("C:/tmp/Football/models_multi_2017/train_outputs_poisson.pdf")
  return df, fig

def evaluate_metrics_and_predict(model, features, labels, outputname, skip_predictions):
    
  eval_results = model.evaluate(
      input_fn=input_fn(features, labels, mode=tf.estimator.ModeKeys.EVAL),
      steps=1, name=outputname)
      
  if skip_predictions:
    return eval_results, None
  
  predictions = model.predict(
      input_fn=input_fn(features, labels, mode=tf.estimator.ModeKeys.PREDICT)
  )
  predictions = list(predictions)
  return eval_results, predictions

def train_and_eval(model_dir, train_steps, train_data, test_data, 
                   predict_new, save_steps, skip_download, max_to_keep, evaluate_after_steps, skip_predictions):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
  
  all_data, teamnames = get_train_test_data(model_dir, train_data, test_data, skip_download)
  features_arrays, labels_array, team_onehot_encoder, label_column_names = build_features(all_data.copy(), teamnames)

#  train_idx = range(2*306*len(train_data))
#  test_idx = range(2*306*len(train_data), 2*306*len(train_data)+2*306*len(test_data))
#  print(train_idx)
#  print(test_idx)
  print(labels_array.shape)

#  print(feature_columns)
#  print(teamnames)

  train_idx = all_data.index[all_data['Train']].tolist()
  test_idx = all_data.index[~all_data['Train'] & ~all_data['Predict']].tolist()
  pred_idx = all_data.index[all_data['Predict']].tolist()
  train_idx = [[2*i, 2*i+1] for i in train_idx ]
  test_idx = [[2*i, 2*i+1] for i in test_idx ]
  pred_idx = [[2*i, 2*i+1] for i in pred_idx ]
  train_idx = [val for sublist in train_idx for val in sublist]
  test_idx = [val for sublist in test_idx for val in sublist]
  pred_idx = [val for sublist in pred_idx for val in sublist]
  print("Train index {}-{}".format(np.min(train_idx), np.max(train_idx)))
  print("Test index {}-{}".format(np.min(test_idx), np.max(test_idx)))
  print("Prediction index {}-{}".format(np.min(pred_idx), np.max(pred_idx)))
  
  
  train_X = {k: v[train_idx] for k, v in features_arrays.items()}
  #print(train_X)
  train_y = labels_array[train_idx]
  #print(train_y)
  test_X = {k: v[test_idx] for k, v in features_arrays.items()}
  test_y = labels_array[test_idx]
  pred_X = {k: v[pred_idx] for k, v in features_arrays.items()}
  pred_y = labels_array[pred_idx]

  tf.reset_default_graph()

  model = gru.create_estimator(model_dir, label_column_names, save_steps, max_to_keep)

  if predict_new:
    new_predictions = model.predict(
      input_fn=input_fn(pred_X, pred_y, mode=tf.estimator.ModeKeys.PREDICT)
    )
    new_predictions = list(new_predictions)
    #print(new_predictions)
    df, _ = prepare_label_fit(new_predictions, pred_X, pred_y, team_onehot_encoder, label_column_names)                             
    df.to_csv(model_dir+"/prediction_outputs_poisson.csv")  
    
    results = [plot_predictions_2(new_predictions, pred_X, pred_y, team_onehot_encoder, prefix, True, skip_plotting=True) 
      for prefix in ["ens/", "p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]]
    results = pd.concat(results)
    results.to_csv("new_predictions_df.csv")
    print(results)
    return 
  
  with tf.Session() as sess:
    sess = sess # dummy to avoid syntax warning
    plot_softprob(gru.makeStaticPrediction(train_X, train_y),6,6,"Static Prediction Train Data")
    plot_softprob(gru.makeStaticPrediction(test_X, test_y),6,6,"Static Prediction Test Data")

  if False: # change to True if model structure has been changed
    utils.upgrade_estimator_model(model_dir, model, train_X, train_y)
    
  for i in range(train_steps//evaluate_after_steps):
  
    model.train(
        input_fn=input_fn(train_X, train_y, mode=tf.estimator.ModeKeys.TRAIN),
        steps=evaluate_after_steps)

    # set steps to None to run evaluation until all data consumed.
    test_result, test_prediction = evaluate_metrics_and_predict(model, test_X, test_y, "test", skip_predictions)
    train_result, train_prediction = evaluate_metrics_and_predict(model, train_X, train_y, "train", skip_predictions)

    if not skip_predictions:
      for prefix in ["ens/", "p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]:
        plot_predictions_2(test_prediction, test_X, test_y, team_onehot_encoder, prefix, dataset = "Test")                             
        plot_predictions_2(train_prediction, train_X, train_y, team_onehot_encoder, prefix, dataset = "Train")                             

  #    test_result = run_evaluation(all_data.copy(), teamnames, model, outputname="test")
  #    train_result = run_evaluation(all_data.copy(), teamnames, model, outputname="train")
  
      df, fig = prepare_label_fit(train_prediction, train_X, train_y, team_onehot_encoder, label_column_names)                             
      df.to_csv(model_dir+"/train_outputs_poisson.csv")  
      fig.savefig(model_dir+"/train_outputs_poisson.pdf")
      plt.close(fig)
      
      df, fig = prepare_label_fit(test_prediction, test_X, test_y, team_onehot_encoder, label_column_names)                             
      df.to_csv(model_dir+"/test_outputs_poisson.csv")  
      fig.savefig(model_dir+"/test_outputs_poisson.pdf")
      plt.close(fig)
    
    results = pd.DataFrame()
    results["Measure"] = test_result.keys()
    results["Train"] = train_result.values()
    results["Test"] = test_result.values()
    results["Diff abs"] = results["Train"] - results["Test"]
    results["Test %"] = results["Test"] / results["Train"] *100
    results = results.sort_values(by="Measure")
    with pd.option_context('display.max_rows', None):
      print(results)


#  print(feature_columns)
#  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
#  print(feature_spec)
#  sir = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
#  print(sir)
#  m.export_savedmodel("C:/tmp/Football/models", sir, as_text=False,checkpoint_path=None)
#  m.export_savedmodel("C:/tmp/Football/models", sir, as_text=True,checkpoint_path=None)
  # Manual cleanup
  #shutil.rmtree(model_dir)


def rolling_train_and_eval(model_dir, train_data, test_data, 
                   predict_new, save_steps, skip_download, max_to_keep, evaluate_after_steps, skip_predictions):
  tf.logging.set_verbosity(tf.logging.INFO)
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
  
  all_data, teamnames = get_train_test_data(model_dir, train_data, test_data, skip_download)
  features_arrays, labels_array, team_onehot_encoder, label_column_names = build_features(all_data.copy(), teamnames)

  print(labels_array.shape)
  print(teamnames)

  train_idx = all_data.index[all_data['Train']].tolist()
  test_idx = all_data.index[~all_data['Train'] & ~all_data['Predict']].tolist()
  pred_idx = all_data.index[all_data['Predict']].tolist()
  train_idx = [[2*i, 2*i+1] for i in train_idx ]
  test_idx = [[2*i, 2*i+1] for i in test_idx ]
  pred_idx = [[2*i, 2*i+1] for i in pred_idx ]
  train_idx = [val for sublist in train_idx for val in sublist]
  test_idx = [val for sublist in test_idx for val in sublist]
  pred_idx = [val for sublist in pred_idx for val in sublist]
  print("Train index {}-{}".format(np.min(train_idx), np.max(train_idx)))
  print("Test index {}-{}".format(np.min(test_idx), np.max(test_idx)))
  print("Prediction index {}-{}".format(np.min(pred_idx), np.max(pred_idx)))
  
  
  train_X = {k: v[train_idx] for k, v in features_arrays.items()}
  train_y = labels_array[train_idx]
  test_X = {k: v[test_idx] for k, v in features_arrays.items()}
  test_y = labels_array[test_idx]
  pred_X = {k: v[pred_idx] for k, v in features_arrays.items()}
  pred_y = labels_array[pred_idx]

  if True:  
    tf.reset_default_graph()
    model = gru.create_estimator(model_dir, label_column_names, save_steps, max_to_keep)
  
    # pretraining
    model.train(
        input_fn=input_fn(train_X, train_y, mode=tf.estimator.ModeKeys.TRAIN),
  #      steps=10)
        steps=5000)
    predictions = []  
    for season in test_data:
      print(season)
      is_current_season = (all_data['Season']==season)
      seasons_idx = all_data.index[is_current_season & ~all_data['Train'] & ~all_data['Predict']].tolist()
      seasons_idx = [[2*i, 2*i+1] for i in seasons_idx  ]
      seasons_idx  = [val for sublist in seasons_idx for val in sublist]

      print("Seasons index {}-{}".format(np.min(seasons_idx), np.max(seasons_idx)))
      #print(seasons_idx)
      season_predictions = []  
      season_pred_X = {k: v[seasons_idx] for k, v in features_arrays.items()}
      season_pred_y = labels_array[seasons_idx]
  
      #chunk_size = 12*9*2
      chunk_size = 5*9*2
      chunk_list = [seasons_idx[i:i+chunk_size] for i  in range(0, len(seasons_idx), chunk_size)]
      
      for chunk_idx in chunk_list:
        print(season)
        print("Chunk index {}-{}".format(np.min(chunk_idx), np.max(chunk_idx)))
      
        chunk_train_X = {k: v[0:np.min(chunk_idx)] for k, v in features_arrays.items()}
        chunk_train_y = labels_array[0:np.min(chunk_idx)]
        chunk_pred_X = {k: v[chunk_idx] for k, v in features_arrays.items()}
        chunk_pred_y = labels_array[chunk_idx]
  
        model.train(
            input_fn=input_fn(chunk_train_X, chunk_train_y, mode=tf.estimator.ModeKeys.TRAIN),
            steps=evaluate_after_steps)
        
        model.evaluate(
            input_fn=input_fn(chunk_pred_X, chunk_pred_y, mode=tf.estimator.ModeKeys.EVAL),
            steps=1, name="Chunk"+season)
        
        model.evaluate(
            input_fn=input_fn(chunk_train_X, chunk_train_y, mode=tf.estimator.ModeKeys.EVAL),
            steps=1, name="Train")
        
        chunk_predictions = model.predict(
            input_fn=input_fn(chunk_pred_X, chunk_pred_y, mode=tf.estimator.ModeKeys.PREDICT)
        )
        chunk_predictions = list(chunk_predictions)
        predictions.extend(chunk_predictions)
        season_predictions.extend(chunk_predictions)
        
        print(season)
        print("Chunk index {}-{}".format(np.min(chunk_idx), np.max(chunk_idx)))
        print("Seasons Prediction File length: {}".format(len(season_predictions)))
        print("Prediction File length: {}".format(len(predictions)))

        with open(model_dir+'/prediction.dmp', 'wb') as fp:
          pickle.dump(predictions, fp)
          
  #    for prefix in ["p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]:
      for prefix in ["ens/"]:
        print(season)
        plot_predictions_2(season_predictions, season_pred_X, season_pred_y, team_onehot_encoder, prefix, dataset = "Test")
    

  with open (model_dir+'/prediction.dmp', 'rb') as fp:
    predictions = pickle.load(fp)

  #print(predictions)
  print(len(predictions))
    
  for prefix in ["ens/", "p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]:
    print("Prediction for rolling seasons {}".format(', '.join(test_data)))
    plot_predictions_2(predictions, test_X, test_y, team_onehot_encoder, prefix, dataset = "Test")
    
  df, fig = prepare_label_fit(predictions, test_X, test_y, team_onehot_encoder, label_column_names)                             
  df.to_csv(model_dir+"/test_outputs_poisson.csv")  
  fig.savefig(model_dir+"/test_outputs_poisson.pdf")
  plt.close(fig)

  if predict_new:
    new_predictions = model.predict(
      input_fn=input_fn(pred_X, pred_y, mode=tf.estimator.ModeKeys.PREDICT)
    )
    new_predictions = list(new_predictions)
    #print(new_predictions)
    df, _ = prepare_label_fit(new_predictions, pred_X, pred_y, team_onehot_encoder, label_column_names)                             
    df.to_csv(model_dir+"/prediction_outputs_poisson.csv")  

    for prefix in ["ens/", "p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]:
      print(season)
      plot_predictions_2(new_predictions, pred_X, pred_y, team_onehot_encoder, prefix, is_prediction=True, dataset = "Predict")


def eval_rolling_prediction(model_dir, train_data, test_data, skip_download):
  print(model_dir)
  
  all_data, teamnames = get_train_test_data(model_dir, train_data, test_data, skip_download)
  features_arrays, labels_array, team_onehot_encoder, label_column_names = build_features(all_data.copy(), teamnames)

  print(labels_array.shape)
  print(teamnames)

  train_idx = all_data.index[all_data['Train']].tolist()
  test_idx = all_data.index[~all_data['Train'] & ~all_data['Predict']].tolist()
  pred_idx = all_data.index[all_data['Predict']].tolist()
  train_idx = [[2*i, 2*i+1] for i in train_idx ]
  test_idx = [[2*i, 2*i+1] for i in test_idx ]
  pred_idx = [[2*i, 2*i+1] for i in pred_idx ]
  train_idx = [val for sublist in train_idx for val in sublist]
  test_idx = [val for sublist in test_idx for val in sublist]
  pred_idx = [val for sublist in pred_idx for val in sublist]
  print("Train index {}-{}".format(np.min(train_idx), np.max(train_idx)))
  print("Test index {}-{}".format(np.min(test_idx), np.max(test_idx)))
  print("Prediction index {}-{}".format(np.min(pred_idx), np.max(pred_idx)))
  
#  train_X = {k: v[train_idx] for k, v in features_arrays.items()}
#  train_y = labels_array[train_idx]
  test_X = {k: v[test_idx] for k, v in features_arrays.items()}
  test_y = labels_array[test_idx]
#  pred_X = {k: v[pred_idx] for k, v in features_arrays.items()}
#  pred_y = labels_array[pred_idx]

  with open (model_dir+'/prediction.dmp', 'rb') as fp:
    predictions = pickle.load(fp)

#  print(len(predictions))
#  print(len(test_y))
#  print({k:len(v) for k,v in test_X.items()})

  ens_results = [plot_predictions_2(predictions, test_X, test_y, team_onehot_encoder, prefix, False, dataset = "Test", skip_plotting=False) 
    for prefix in ["ens/"]]
  ens_results = pd.concat(ens_results)
  ens_results.to_csv(model_dir+"/ens_rolling_predictions_df.csv")
  print(ens_results)


  results = [plot_predictions_2(predictions, test_X, test_y, team_onehot_encoder, prefix, False, dataset = "Test", skip_plotting=False) 
    for prefix in ["p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]]
  results = pd.concat(results)
  results.to_csv(model_dir+"/rolling_predictions_df.csv")
  print(results)

  df, fig = prepare_label_fit(predictions, test_X, test_y, team_onehot_encoder, label_column_names)                             
  df.to_csv(model_dir+"/test_outputs_poisson.csv")  
  fig.savefig(model_dir+"/test_outputs_poisson.pdf")
  plt.close(fig)

  for season in test_data:
    print(season)
    is_current_season = (all_data['Season']==season)
    seasons_idx = all_data.index[is_current_season & ~all_data['Train'] & ~all_data['Predict']].tolist()
    seasons_idx = [[2*i, 2*i+1] for i in seasons_idx  ]
    seasons_idx  = [val for sublist in seasons_idx for val in sublist]

    print("Seasons index {}-{}".format(np.min(seasons_idx), np.max(seasons_idx)))
    season_predictions = predictions[np.min(seasons_idx)-len(train_idx):np.max(seasons_idx)-len(train_idx)+1]  
    season_pred_X = {k: v[seasons_idx] for k, v in features_arrays.items()}
    season_pred_y = labels_array[seasons_idx]

    for prefix in ["ens/", "p1/", "p3/", "p2/", "p5/", "p4/", "p6/", "sp/", "sm/", "smhb/"]:
      print(season)
      plot_predictions_2(season_predictions, season_pred_X, season_pred_y, team_onehot_encoder, prefix, dataset = "Test")
    
    df, fig = prepare_label_fit(season_predictions, season_pred_X, season_pred_y, team_onehot_encoder, label_column_names)                             
    df.to_csv(model_dir+"/test_outputs_poisson"+season+".csv")  
    fig.savefig(model_dir+"/test_outputs_poisson"+season+".pdf")
    plt.close(fig)


FLAGS = None

def main(_):
  
#  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/Mapping_Layer/WM", target_file_name="mapping.csv", all_tensor_names=False, all_tensors=False)
#  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel", target_file_name="rnn_candidate_kernel.csv", all_tensor_names=False, all_tensors=False)
#  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel", target_file_name="rnn_gates_kernel.csv", all_tensor_names=False, all_tensors=False)

#  train_and_eval(FLAGS.model_dir, FLAGS.train_steps,
#                 FLAGS.train_data, FLAGS.test_data, FLAGS.predict_new,
#                 FLAGS.save_steps, FLAGS.skip_download, FLAGS.max_to_keep, 
#                 FLAGS.evaluate_after_steps, FLAGS.skip_predictions)
#  
#  rolling_train_and_eval(FLAGS.model_dir, FLAGS.train_data, FLAGS.test_data, FLAGS.predict_new, FLAGS.save_steps, FLAGS.skip_download, FLAGS.max_to_keep, FLAGS.evaluate_after_steps, FLAGS.skip_predictions)
  eval_rolling_prediction(FLAGS.model_dir, FLAGS.train_data, FLAGS.test_data, FLAGS.skip_download)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--skip_download", type=bool,
      default=True, 
      help="Use input files in model_dir without downloading"
  )
  parser.add_argument(
      "--skip_predictions", type=bool,
      default=False, 
      help="Print plots of predicted data"
  )
  parser.add_argument(
      "--predict_new", type=str,
      #default=False, 
      default=True, 
      help="Predict new games only"
  )
  parser.add_argument(
      "--train_steps", type=int,
      default=30000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--save_steps", type=int,
      #default=500,
      default=500,
      help="Number of training steps between checkpoint files."
  )
  parser.add_argument(
      "--evaluate_after_steps", type=int,
      #default=500,
      default=1000,
      help="Number of training steps after which to run evaluation. Should be a multiple of save_steps"
  )
  parser.add_argument(
      "--max_to_keep", type=int,
      default=60,
      help="Number of checkpoint files to keep."
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      #default="C:/tmp/Football/models_gru4_1415", # always use UNIX-style path names!!!
      #default="C:/tmp/Football/models_1516", # always use UNIX-style path names!!!
      #default="C:/tmp/Football/models_1617", # always use UNIX-style path names!!!
      #default="C:/tmp/Football/models_gru4_1718",
      #default="C:/tmp/Football/models_1718_new2", # always use UNIX-style path names!!!
      default="C:/tmp/Football/rolling_gru4_1418",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--train_data", type=str,
      default=["1112", "1213", "1314"], #
      #default=["1213", "1314", "1415"], #
      #default=["1314", "1415", "1516"], #
      #default=["1415", "1516", "1617"], #
      #default=["1415", "1516", "1617", "1718"], #
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data", type=str,
      #default=["1415"],
      #default=["1516"],
      #default=["1617"],
      #default=["1718"],
      default=["1415", "1516", "1617", "1718"], #
      help="Path to the test data."
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# Path('C:/tmp/Football/models/reset.txt').touch()
  
if False:  
  p = pd.read_csv("new_predictions_df.csv")
  p.groupby(['Prefix','Where']).agg(['mean', 'sum'])
  p.groupby(['Prefix','Where']).sum()
  p.groupby(['Prefix','Where']).mean()
  p.loc[p["Where"]=="Home"]
  
  p = pd.read_csv("C:/tmp/Football/rolling_gru4_1418/rolling_predictions_df.csv")
  #p = pd.read_csv("C:/tmp/Football/rolling_gru3_1418_full2/rolling_predictions_df.csv")
  p.groupby(['Prefix','Where']).sum()
  p1 = p[['Unnamed: 0', 'GS', 'GC', 'pGS', 'pGC', 'est1', 'est2', 'Team1', 'Team2', 'Where', 'act', 'pred', 'Pt', 'Prefix']].copy()
  p1["Season"]=p1["Unnamed: 0"]//612
  p1.groupby(['Prefix','Where']).mean()
  p1.sort_values('Unnamed: 0')
  p1.groupby(['Season','Prefix']).mean()
  p2 = p1[['Season','Prefix',"Pt",'Where' ]]
  p2.pivot_table(index=['Season', "Where"], values="Pt", aggfunc=np.mean, columns="Prefix", margins=True)

  p2.pivot_table(index=['Season'], values="Pt", aggfunc=np.mean, columns="Prefix", margins=True)

  p3 = pd.read_csv("C:/tmp/Football/rolling_gru4_1418/ens_rolling_predictions_df.csv")
  #p = pd.read_csv("C:/tmp/Football/rolling_gru3_1418_full2/rolling_predictions_df.csv")
  p4 = p3[['Unnamed: 0', 'GS', 'GC', 'pGS', 'pGC', 'Team1', 'Team2', 'Where', 'act', 'pred', 'Pt', 'Prefix', "Strategy"]].copy()
  p4["Season"]=p4["Unnamed: 0"]//306
  p4 = p4[['Season','Prefix',"Pt",'Strategy' ]]
  p4.pivot_table(index=['Season', "Strategy"], values="Pt", aggfunc=np.mean, columns="Prefix", margins=True)

  p4.pivot_table(index=['Season'], values="Pt", aggfunc=np.mean, columns="Prefix", margins=False)
