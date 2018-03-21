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

import pandas as pd
pd.set_option('expand_frame_repr', False)

import numpy as np
#np.set_printoptions(threshold=50)

from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
#import matplotlib.markers as markers
#from Estimators import LinearModel as lm
#from Estimators import PoissonModel_teamcentric as pm
#from Estimators import DiscreteModel as dm
#from Estimators import DiscreteModelMulti as dmm
#from Estimators import DiscreteLayeredModel as dlm
#from Estimators import DiscreteRNNModel as drm
#from Estimators import LSTMModel as lstm
#from Estimators import LSTM_m21_Model as lstm_m21
from Estimators import LSTM_multihead_Model as lstm_multihead
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

skip_download = True

def download_data(model_dir, season):
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
  _replace_teamnames("1112", "FC Koln")  
  _replace_teamnames("1415", "Freiburg")  
  _replace_teamnames("1516", "Hannover")  
  _replace_teamnames("1516", "Stuttgart")  
  return(data)

def get_train_test_data(model_dir, train_seasons, test_seasons):
  train_data = []
  for s in train_seasons:
    newdata = download_data(model_dir, s) 
    replace_teamnames(newdata, s)
    train_data.append(newdata)
  train_data = pd.concat(train_data, ignore_index=True)
  
  test_data = []
  for s in test_seasons:
    test_data.append(replace_teamnames(download_data(model_dir, s), s) )
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
  features["zGameFinalScore"] = [str(x1)+":"+str(x2) if w==1 else str(x2)+":"+str(x1) for x1,x2,w in zip(features["T1_GFT"], features["T2_GFT"], features["Where"])]
  
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
  

  final_score_enum = ["0:3", "1:3", "0:2", "1:2", "0:1", "0:0", "1:1", "2:2", "1:0", "2:1", "2:0", "3:1", "3:0"]
  final_score_encoder = preprocessing.LabelBinarizer(sparse_output=False)
  final_score_encoder.fit_transform(final_score_enum)
  label_column_names += ["zScore"+s for s in final_score_enum]
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
  
  team_onehot_encoder = preprocessing.LabelBinarizer(sparse_output=False)
  team_onehot_encoder .fit_transform(teamnames)

  steps = 10
  tn = len(teamnames)
  lc = len(label_column_names)
  fc = len(feature_column_names)
  tfc = 2*tn+fc+1 # total_feature_columns
  newgame = np.ndarray(shape=[len(features), 3+tfc], dtype=np.float32)
  labels = np.ndarray(shape=[len(features), lc], dtype=np.float32)
  match_history_t1 = np.ndarray(shape=[len(features), steps, 1+tfc+lc], dtype=np.float32)
  match_history_t2 = np.ndarray(shape=[len(features), steps, 1+tfc+lc], dtype=np.float32)
  match_history_t12 = np.ndarray(shape=[len(features), steps, 1+tfc+lc], dtype=np.float32)

#  goal_history_t1 = np.ndarray(shape=[len(features), 2*steps, 12+tfc], dtype=np.float32)
#  goal_history_t2 = np.ndarray(shape=[len(features), 2*steps, 12+tfc], dtype=np.float32)

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
      match_history[index, :seq_len, j:j+fc] = teamdata [feature_column_names]
      j = j+fc
      match_history[index, :seq_len, j:j+lc] = teamdata [label_column_names]
      
    return seq_len

  def build_goal_history_data(match_history, teamdata, index):
    teamdata = teamdata[teamdata.index < index] # include only matches from the past
    teamdata = teamdata[-steps:] # include up to "steps" previous matches if available
    seq_len = len(teamdata)
    j = 0
    match_history[index, 0::2, j] = seq_len*2 
    match_history[index, 1::2, j] = seq_len*2 
    if seq_len>0:
      # 1H Features
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["Where"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j:j+tn] = team_onehot_encoder.transform(teamdata["Team1"])
      j = j+tn
      match_history[index, 0:2*seq_len:2, j:j+tn] = team_onehot_encoder.transform(teamdata["Team2"])
      j = j+tn
      match_history[index, 0:2*seq_len:2, j:j+fc] = teamdata [feature_column_names]
      j = j+fc
      match_history[index, 0:2*seq_len:2, j] = 0
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = 0
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = 0
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = 1
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = 0
      j = j+1
      # Labels
      match_history[index, 0:2*seq_len:2, j] = 0 # 1H
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["T1_GHT"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["T2_GHT"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["HTLoss"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["HTDraw"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["HTWin"]
      j = j+1
      
      # 2H Features
      j = 0 
      j = j+1
      match_history[index, 1:2*seq_len+1:2, j] = teamdata["Where"]
      j = j+1
      match_history[index, 1:2*seq_len+1:2, j:j+tn] = team_onehot_encoder.transform(teamdata["Team1"])
      j = j+tn
      match_history[index, 1:2*seq_len+1:2, j:j+tn] = team_onehot_encoder.transform(teamdata["Team2"])
      j = j+tn
      match_history[index, 1:2*seq_len+1:2, j:j+fc] = teamdata [feature_column_names]
      j = j+fc
      match_history[index, 1:2*seq_len+1:2, j] = teamdata["T1_GHT"]
      j = j+1
      match_history[index, 1:2*seq_len+1:2, j] = teamdata["T2_GHT"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["HTLoss"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["HTDraw"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["HTWin"]
      j = j+1
      # Labels
      match_history[index, 0:2*seq_len:2, j] = 1 # 2H
      j = j+1
      match_history[index, 1:2*seq_len+1:2, j] = teamdata["T1_GH2"]
      j = j+1
      match_history[index, 1:2*seq_len+1:2, j] = teamdata["T2_GH2"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["Loss"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["Draw"]
      j = j+1
      match_history[index, 0:2*seq_len:2, j] = teamdata["Win"]
      j = j+1
      
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
      
#    build_goal_history_data(goal_history_t1, batches1[team1], index)
#    build_goal_history_data(goal_history_t2, batches2[team2], index)

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
#      "goal_history_t1": goal_history_t1,
#      "goal_history_t2": goal_history_t2,
      }, labels, team_onehot_encoder, label_column_names, tfc

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

def plot_softprob(pred, gs, gc, title="", is_softprob=False):
  print("-----------------------------------------------------")
  print(title)
  sp = pred["sm/p_pred_12"]
  spt = pred["sm/ev_points"]
  spsp = pred["sp/p_pred_12"]
    
  gs = min(gs,6)
  gc = min(gc,6)
  margin_pred_prob1 = pred["sm/p_marg_1"]
  margin_poisson_prob1 = pred["sm/p_poisson_1"]
  margin_pred_prob2 = pred["sm/p_marg_2"]
  margin_poisson_prob2 = pred["sm/p_poisson_2"]
  margin_pred_expected1 = pred["sm/ev_goals_1"] 
  margin_pred_expected2 = pred["sm/ev_goals_2"] 

  margin_pred_prob1_sp = pred["sp/p_marg_1"]
  margin_pred_prob2_sp = pred["sp/p_marg_2"]
  margin_pred_expected1_sp = pred["sp/ev_goals_1"] 
  margin_pred_expected2_sp = pred["sp/ev_goals_2"] 

  g=[0,1,2,3,4,5,6]
  g1=[0]*7+[1]*7+[2]*7+[3]*7+[4]*7+[5]*7+[6]*7
  g2=g*7
  fig, ax = plt.subplots(1,3,figsize=(15,5))
  ax[0].scatter(g1, g2, s=sp*10000, alpha=0.5)
  ax[0].scatter(gs, gc, s=sp[gs*7+gc]*10000, alpha=0.5, color='blue')
  for i, txt in enumerate(sp):
    ax[0].annotate("{:4.2f}".format(txt*100), (g1[i],g2[i]))
  max_sp = max(sp)
  max_sp_index = np.argmax(sp) 
  ax[0].scatter(max_sp_index//7, np.mod(max_sp_index, 7), s=max_sp*10000, facecolors='none', edgecolors='black', linewidth='2')

  ax[1].scatter(g1, g2, s=spt*500, alpha=0.4,color='red')
  ax[1].scatter(gs, gc, s=spt[gs*7+gc]*500, alpha=0.7,color='red')
  for i, txt in enumerate(spt):
    ax[1].annotate("{:4.2f}".format(txt), (g1[i],g2[i]))
  max_spt = max(spt)
  max_spt_index = np.argmax(spt) 
  ax[1].scatter(max_spt_index//7, np.mod(max_spt_index, 7), s=max_spt*500, facecolors='none', edgecolors='black', linewidth='2')

  ax[2].scatter(g1, g2, s=spsp*10000, alpha=0.4, color='green')
  ax[2].scatter(gs, gc, s=spsp[gs*7+gc]*10000, alpha=0.5, color='green')
  for i, txt in enumerate(spsp):
    ax[2].annotate("{:4.2f}".format(txt*100), (g1[i],g2[i]))
  max_sp = max(spsp)
  max_sp_index = np.argmax(spsp) 
  ax[2].scatter(max_sp_index//7, np.mod(max_sp_index, 7), s=max_sp*10000, facecolors='none', edgecolors='black', linewidth='2')

  plt.show()

  w=0.35
  fig, ax = plt.subplots(1,3,figsize=(15,1))
  ax[0].bar(g, margin_pred_prob1,alpha=0.6, width=w)
  ax[0].bar([x+w for x in g], margin_poisson_prob1,alpha=0.3,color="red",width=0.35)
  ax[0].bar(gs, margin_pred_prob1[gs],alpha=0.5, width=w, color='blue')
  ax[0].bar(gs+w, margin_poisson_prob1[gs],alpha=0.7,color="red",width=0.35)
  ax[0].axvline(x=margin_pred_expected1, color='red')

  ax[1].bar(g, margin_pred_prob2,alpha=0.6, width=w)
  ax[1].bar([x+w for x in g], margin_poisson_prob2,alpha=0.3,color="red",width=0.35)
  ax[1].bar(gc, margin_pred_prob2[gc],alpha=0.5, width=w, color='blue')
  ax[1].bar(gc+w, margin_poisson_prob2[gc],alpha=0.7,color="red",width=0.35)
  ax[1].axvline(x=margin_pred_expected2, color='red')

  ax[2].bar(g, margin_pred_prob1_sp,alpha=0.3, width=w, color="green")
  ax[2].bar([x+w for x in g], margin_pred_prob2_sp,alpha=0.3,color="red",width=0.35)
  ax[2].bar(gs, margin_pred_prob1_sp[gs],alpha=0.5, width=w, color='green')
  ax[2].bar(gc+w, margin_pred_prob2_sp[gc],alpha=0.5,color="red",width=0.35)
  ax[2].axvline(x=margin_pred_expected1_sp, color='green')
  ax[2].axvline(x=margin_pred_expected2_sp, color='red')
  plt.show()

def plot_predictions(predictions, features, labels, team_onehot_encoder, is_prediction=False, is_softprob=False):

  features = features["newgame"]
  
  # sample input and output values for printing
  predictions[0].update({"features": features[0,:]})
  predictions[1].update({"features": features[1,:]})
  predictions[0].update({"labels": labels[0,:]})
  predictions[1].update({"labels": labels[1,:]})
  
#  print(predictions[0:2])

  df = pd.DataFrame()  
  df["GS"] = labels[:,0].astype(np.int)
  df["GC"] = labels[:,1].astype(np.int)

  if is_softprob:
    df['pGS'] = [p["sp/pred"][0] for p in predictions]
    df['pGC'] = [p["sp/pred"][1] for p in predictions]
    est1 = pd.Series([p["sp/ev_goals_1"] for p in predictions], name="est1_sp")
    est2 = pd.Series([p["sp/ev_goals_2"] for p in predictions], name="est2_sp")
    theme_color = "green"
  else:
    df['pGS'] = [p["sm/pred"][0] for p in predictions]
    df['pGC'] = [p["sm/pred"][1] for p in predictions]
    est1 = pd.Series([p["sm/ev_goals_1"] for p in predictions], name="est1")
    est2 = pd.Series([p["sm/ev_goals_2"] for p in predictions], name="est2")
    theme_color = "blue"
    
  df['est1'] = est1
  df['est2'] = est2
  # print(team_onehot_encoder.classes_)
  tn = len(team_onehot_encoder.classes_)
  df['Team1']=team_onehot_encoder.inverse_transform(features[:, 4:4+tn])
  df['Team2']=team_onehot_encoder.inverse_transform(features[:, 4+tn:4+2*tn])
  df['Where']=['Home' if h==1 else 'Away' for h in features[:, 2]]
  df["act"]  = [str(gs)+':'+str(gc) for gs,gc in zip(df["GS"],df["GC"]) ]
  df["pred"] = [str(gs)+':'+str(gc) for gs,gc in zip(df["pGS"],df["pGC"]) ]
  
  tensor = tf.constant(df[[ "pGS", "pGC", "GS", "GC"]].as_matrix(), dtype = tf.int64)
  is_home = tf.equal(features[:,2] , 1)
  with tf.Session() as sess:
    points_tensor = lstm_multihead.calc_points(tensor[:,0],tensor[:,1], tensor[:,2], tensor[:,3], is_home)[0]
    df['Pt'] = sess.run(tf.cast(points_tensor, tf.int8))
    print(np.sum(df['Pt'])/len(df))

#  print(
#      df[["Team1", "Team2", "GS", "GC", "pGS", "pGC", "Where", "est1","est2","Pt"]]
##      .sort_values(["est1","est2"], ascending=[True, False])
#      )
  print(
      df[["Team1", "Team2", "act", "pred", "Where", "est1","est2","Pt"]]
#      .sort_values(["est1","est2"], ascending=[True, False])
      .head(80)
      )
  print(
      df[["Team1", "Team2", "act", "pred", "Where", "est1","est2","Pt"]]
#      .sort_values(["est1","est2"], ascending=[True, False])
      .tail(200)
      )
  if is_prediction:
    if is_softprob:
      return
    for s in range(len(df)):
      plot_softprob(predictions[s], df["GS"][s], df["GC"][s], df["Team1"][s]+" - "+df["Team2"][s]+" ("+df["Where"][s]+")", is_softprob)
    return
  else:
    s = random.sample(range(len(df)), 1)[0]
    plot_softprob(predictions[s], df["GS"][s], df["GC"][s], df["Team1"][s]+" - "+df["Team2"][s]+" ("+df["Where"][s]+")", is_softprob)
    s = random.sample(range(len(df)), 1)[0]
    plot_softprob(predictions[s], df["GS"][s], df["GC"][s], df["Team1"][s]+" - "+df["Team2"][s]+" ("+df["Where"][s]+")", is_softprob)

  fig = plt.figure(figsize=(14,4))
  ax1 = plt.subplot2grid((1,3), (0,0), colspan=2, rowspan=1)
  ax2 = plt.subplot2grid((1,3), (0,2), colspan=2, rowspan=1)
  ax2.axis('off')
  
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

  print()
  print("Points: {0:.4f}, Tendency: {1:.2f}, Diff: {2:.2f}, Full: {3:.2f},    Home: {4:.1f}, Draw: {5:.1f}, Away: {6:.1f}".format(
      np.sum(t2["Contribution"]),
      100.0 * (1-pie_chart_values[0]/len(labels)),
      100.0 * (pie_chart_values[2]+pie_chart_values[3])/len(labels),
      100.0 * pie_chart_values[3]/len(labels),
      100.0 * tendency_values[2]/len(labels),
      100.0 * tendency_values[1]/len(labels),
      100.0 * tendency_values[0]/len(labels)
    ))

  t1 = df.pivot_table(index = [df["sort_idx"], 'pGoals'], columns=['gof'], values=["Team1"], aggfunc=len, margins=False, fill_value=0)
  t1.columns = ["None", "Tendency", "Diff", "Full"][:len(t1.columns)]
  t1.index = t1.index.droplevel(level=0)
  t1.plot(kind='bar', stacked=True, ax=ax1)

  ax2.pie(pie_chart_values, 
          labels=["None", "Tendency", "Diff", "Full"], 
          startangle=90)
  plt.show()
  plt.close()
  
  t5 = df.pivot_table(index = 'Team1', 
            aggfunc={'0_none':np.sum, '1_tendency':np.sum, '2_diff':np.sum, '3_full':np.sum,
                    'total_points':[np.sum, np.mean],
                    "hit_tend":np.mean, "hit_diff":np.mean, "hit_goal":np.mean},
                     margins=False, fill_value=0)
  t5.columns = ["None", "Tendency", "Diff", "Full", "Diff%", "Goal%", "Tendency%", "AvgPoints", "TotalPoints"]
  t5 = t5.sort_values(by=('AvgPoints'), ascending=False)
  print(t5)
  
  c_home = df["Where"]=="Home"
  c_win  = df['pGS'] > df['pGC']
  c_loss = df['pGS'] < df['pGC']
  c_draw = df['pGS'] == df['pGC']
  c_tendency = np.sign(df['pGS']- df['pGC']) == np.sign(df["GS"] - df["GC"]) 
  fig, ax = plt.subplots(1,2,figsize=(12,4))
#  ax[0].scatter(est1, df["GS"], alpha=0.1, color=theme_color)
  ax[0].set_title(np.corrcoef(df["est1"], df["GS"])[0,1])
  ax[0].scatter(df[~c_home]["est1"], df[~c_home]["GS"], alpha=0.1, color="red")
  ax[0].scatter(df[c_home]["est1"], df[c_home]["GS"], alpha=0.1, color=theme_color)
#  ax[1].scatter(est2, df["GC"], alpha=0.1, color=theme_color)
  ax[1].set_title(np.corrcoef(df["est2"], df["GC"])[0,1])
  ax[1].scatter(df[~c_home]["est2"], df[~c_home]["GC"], alpha=0.1, color="red")
  ax[1].scatter(df[c_home]["est2"], df[c_home]["GC"], alpha=0.1, color=theme_color)
  plt.show()
  fig, ax = plt.subplots(1,2,figsize=(12,4))
#  ax[0].scatter(est_diff, df["GS"]-df["GC"],alpha=0.1, color=theme_color)
  ax[0].set_title(np.corrcoef(df["est1"]-df["est2"], df["GS"]-df["GC"])[0,1])
  ax[0].scatter(df[~c_home]["est1"]-df[~c_home]["est2"], df[~c_home]["GS"]-df[~c_home]["GC"],alpha=0.1, color="red")
  ax[0].scatter(df[c_home]["est1"]-df[c_home]["est2"], df[c_home]["GS"]-df[c_home]["GC"],alpha=0.1, color=theme_color)

  def plotEstimates(cond1, cond2, color, alpha=0.1):
    ax[1].scatter(est1[cond1 & c_tendency], est2[cond1 & c_tendency],alpha=alpha,color=color, marker='o')
    ax[1].scatter(est1[cond1 & ~c_tendency], est2[cond1 & ~c_tendency],alpha=alpha,color=color, marker='x')
    ax[1].scatter(est2[cond2 & c_tendency], est1[cond2 & c_tendency],alpha=alpha,color=color, marker='o')
    ax[1].scatter(est2[cond2 & ~c_tendency], est1[cond2 & ~c_tendency],alpha=alpha,color=color, marker='x')
  
  plotEstimates(c_win & c_home, c_loss & ~c_home, "blue")
  plotEstimates(c_loss & c_home, c_win & ~c_home, "red")
  plotEstimates(c_draw, c_draw, "green", 0.3)
  plt.show()
  plt.close()

  y_pred = [np.sign(p1-p2) if w=="Home" else np.sign(p2-p1) for p1,p2,w in zip(df["pGS"],df["pGC"],df["Where"])]
  y_pred = ["Draw" if i==0 else "HomeWin" if i==1 else "AwayWin" for i in y_pred]
  y_test = [np.sign(p1-p2) if w=="Home" else np.sign(p2-p1) for p1,p2,w in zip(df["GS"],df["GC"],df["Where"])]
  y_test = ["Draw" if i==0 else "HomeWin" if i==1 else "AwayWin" for i in y_test]

  cnf_matrix = confusion_matrix(y_test, y_pred)
  np.set_printoptions(precision=2)
  
  # Plot non-normalized confusion matrix
  fig, ax = plt.subplots(1,2,figsize=(10,4))
  plot_confusion_matrix(ax[0], cnf_matrix, classes=["AwayWin", "Draw", "HomeWin"],
                        title='Tendency')
  
  # Plot normalized confusion matrix
  plot_confusion_matrix(ax[1], cnf_matrix, classes=["AwayWin", "Draw", "HomeWin"],
                        normalize=True,
                        title='Tendency')
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
                        title='Tendency')
  
  # Plot normalized confusion matrix
  plot_confusion_matrix(ax[1], cnf_matrix, classes=np.unique(y_test).tolist(),
                        normalize=True,
                        title='Tendency')
  plt.show()
  plt.close()

def prepare_label_fit(predictions, features, labels, team_onehot_encoder, label_column_names):                             
  features = features["newgame"]
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
    df.boxplot(column = "p_"+col, by=col, ax=ax[i//4,np.mod(i,4)], fontsize=10, grid=False)
  #plt.show()  
  #fig.savefig("C:/tmp/Football/models_multi_2017/train_outputs_poisson.pdf")
  return df, fig

def evaluate_metrics_and_predict(model, features, labels, outputname):
    
  eval_results = model.evaluate(
      input_fn=input_fn(features, labels, mode=tf.estimator.ModeKeys.EVAL),
      steps=1, name=outputname)
      
#  for key in sorted(eval_results):
#    print("%s: %s" % (key, eval_results[key]))

  predictions = model.predict(
      input_fn=input_fn(features, labels, mode=tf.estimator.ModeKeys.PREDICT)
  )
  predictions = list(predictions)
  return eval_results, predictions

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data, predict_new, reset_weights):
  tf.logging.set_verbosity(tf.logging.INFO)
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
    
  all_data, teamnames = get_train_test_data(model_dir, train_data, test_data)
  features_arrays, labels_array, team_onehot_encoder, label_column_names, total_feature_columns = build_features(all_data.copy(), teamnames)

#  train_idx = range(2*306*len(train_data))
#  test_idx = range(2*306*len(train_data), 2*306*len(train_data)+2*306*len(test_data))
#  print(train_idx)
#  print(test_idx)
  print(labels_array.shape)

  num_label_columns = labels_array.shape[1]

  model = lstm_multihead.create_estimator(model_dir, num_label_columns, total_feature_columns, reset_weights)

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
  train_y = labels_array[train_idx]
  test_X = {k: v[test_idx] for k, v in features_arrays.items()}
  test_y = labels_array[test_idx]
  pred_X = {k: v[pred_idx] for k, v in features_arrays.items()}
  pred_y = labels_array[pred_idx]

  if predict_new:
    new_predictions = model.predict(
      input_fn=input_fn(pred_X, pred_y, mode=tf.estimator.ModeKeys.PREDICT)
    )
    new_predictions = list(new_predictions)
    #print(new_predictions)
    plot_predictions(new_predictions, pred_X, pred_y, team_onehot_encoder, True)                             
    plot_predictions(new_predictions, pred_X, pred_y, team_onehot_encoder, True, True)                             
    df, fig = prepare_label_fit(new_predictions, pred_X, pred_y, team_onehot_encoder, label_column_names)                             
    df.to_csv(model_dir+"/predict_outputs_poisson.csv")
    plt.close(fig)
    return
  
  if reset_weights:
    model.train(
        input_fn=input_fn(train_X, train_y, mode=tf.estimator.ModeKeys.TRAIN),
        steps=1) #, hooks=[summary_hook]) #hooks=[MyHook(teamnames)])#, 
    return
  
#  with tf.Session() as sess:
#    sess = sess # dummy to avoid syntax warning
#    plot_softprob(lstm_multihead.makeStaticPrediction(train_X, train_y),6,6,"Static Prediction Train Data")
#    plot_softprob(lstm_multihead.makeStaticPrediction(test_X, test_y),6,6,"Static Prediction Test Data")

  for i in range(train_steps//100):
  
#    model.train(
#        input_fn=input_fn(train_X, train_y, mode=tf.estimator.ModeKeys.TRAIN),
#        steps=300) #, hooks=[summary_hook]) #hooks=[MyHook(teamnames)])#, 
    # set steps to None to run evaluation until all data consumed.
    test_result, test_prediction = evaluate_metrics_and_predict(model, test_X, test_y, outputname="test")
    train_result, train_prediction = evaluate_metrics_and_predict(model, train_X, train_y, outputname="train")
    
#   if (np.mod(i, 50)==0):
    plot_predictions(test_prediction, test_X, test_y, team_onehot_encoder)                             
    plot_predictions(test_prediction, test_X, test_y, team_onehot_encoder, False, True)                             
    plot_predictions(train_prediction, train_X, train_y, team_onehot_encoder)                             
    plot_predictions(train_prediction, train_X, train_y, team_onehot_encoder, False, True)                             

    
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


FLAGS = None


def main(_):
  
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data, FLAGS.predict_new,
                 FLAGS.reset_weights)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="C:/tmp/Football/models_multi_2017", # always use UNIX-style path names!!!
      help="Base directory for output models."
  )
  parser.add_argument(
      "--predict_new",
      type=str,
      default=False, 
      help="Predict new games only"
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="own",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep', 'own', 'poisson'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=30000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      #default=["1112", "1213", "1314"], #
      default=["1314", "1415", "1516"], #
      #default=["1415", "1516", "1617"], #
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      #default=["1718"],
      default=["1617"],
      help="Path to the test data."
  )
  parser.add_argument(
      "--reset_weights",
      type=bool,
      #default=["1718"],
      default=False,
      help="Reset the model weights during training (one-off)"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# Path('C:/tmp/Football/models/reset.txt').touch()