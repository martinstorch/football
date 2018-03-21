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
import shutil
import sys
import tempfile

import pandas as pd
pd.set_option('expand_frame_repr', False)

import numpy as np
from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from Estimators import LinearModel as lm
from Estimators import PoissonModel_teamcentric as pm
from Estimators import DiscreteModel as dm
from Estimators import DiscreteModelMulti as dmm
from Estimators import DiscreteLayeredModel as dlm
from Estimators import DiscreteRNNModel as drm
from Estimators import LSTMModel as lstm
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.contrib.layers import l2_regularizer
from collections import Counter
from pathlib import Path

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

def get_train_test_data(model_dir, train_seasons, test_seasons):
  train_data = []
  for s in train_seasons:
    train_data.append(download_data(model_dir, s) )
  train_data = pd.concat(train_data, ignore_index=True)
  
  test_data = []
  for s in test_seasons:
    test_data.append(download_data(model_dir, s) )
  test_data = pd.concat(test_data, ignore_index=True)

  print(train_data.shape)  
  print(test_data.shape)  
  teamnames = [] 
  teamnames.extend(train_data["HomeTeam"].tolist())
  teamnames.extend(train_data["AwayTeam"].tolist())
  teamnames.extend(test_data["HomeTeam"].tolist())
  teamnames.extend(test_data["AwayTeam"].tolist())
  teamnames = np.unique(teamnames).tolist()
  train_data["Train"]=True
  test_data["Train"]=False
  all_data = pd.concat([train_data, test_data], ignore_index=True)
  return all_data, teamnames



def build_features(df_data, teamnames, mode=tf.estimator.ModeKeys.TRAIN):
  if mode==tf.estimator.ModeKeys.TRAIN:
    df_data = df_data[df_data["Train"]==True].copy()
  
  print("Build features - {}".format(mode))
  print(df_data.shape) 
  
  df_data.rename(columns={
      'FTHG': 'HGFT', 'FTAG': 'AGFT',
      'HTHG': 'HGHT', 'HTAG': 'AGHT'
      }, inplace=True)

  df1 = pd.DataFrame()
  df1["Team1"] = df_data["HomeTeam"]
  df1["Team2"] = df_data["AwayTeam"]
  df1["Where"] = "Home"
  df1['OpponentGoals'] = df_data["AGFT"]
  df1['OwnGoals'] = df_data["HGFT"]
  df1['HomeTeam'] = df1["Team1"]
  df1['Season'] = df_data["Season"]
  df1["Train"] = df_data["Train"]
  df1["Date"]= df_data["Date"]
    
  df2 = pd.DataFrame()
  df2["Team1"] = df_data["AwayTeam"]
  df2["Team2"] = df_data["HomeTeam"]
  df2["Where"] = "Away"
  df2['OpponentGoals'] = df_data["HGFT"]
  df2['OwnGoals'] = df_data["AGFT"]
  df2['HomeTeam'] = df1["Team2"]
  df2['Season'] = df_data["Season"]
  df2["Train"] = df_data["Train"]
  df2["Date"]= df_data["Date"]
  
  columns = [c[1:] for c in COLS[::2]]
  feature_column_names_fixed = ["Team1", "Team2", "Where", "Season", "Train"]
  feature_column_names = []
  label_column_names = []
  for colname in columns:
    homecol="H"+colname
    awaycol="A"+colname
    df1["T1_"+colname] = df_data[homecol].astype(int)
    df1["T2_"+colname] = df_data[awaycol].astype(int)
    df2["T1_"+colname] = df_data[awaycol].astype(int)
    df2["T2_"+colname] = df_data[homecol].astype(int)
    label_column_names += ["T1_"+colname, "T2_"+colname]
    feature_column_names += ["T1_Input_"+colname, "T2_Input_"+colname]

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
  feature_column_names += ["T1_Input_GH2", "T2_Input_GH2"]
  columns += ["GH2"]

  game_results_choice = {'1': 'Win', '-1': 'Loss', '0':'Draw'}
  # features = features.assign(GameResult = lambda x: game_results_choice.get(str(np.sign(x.T1_GFT - x.T2_GFT))))
  features["zGameResult"] = [game_results_choice.get(str(np.sign(x1-x2))) for x1,x2 in zip(features["T1_GFT"], features["T2_GFT"])]
#  print(features["GameResult"] )
  # features = features.assign(GameFinalScore = lambda x: str(x.T1_GFT)+':'+str(x.T2_GFT))
  features["zGameFinalScore"] = [str(x1)+":"+str(x2) for x1,x2 in zip(features["T1_GFT"], features["T2_GFT"])]
#  print(features["GameFinalScore"] )
  
  batches = [features[features["Team1"]==t].copy() for t in teamnames]
  for i,b in enumerate(batches):
    b["sample_id"]=i
    for colname in columns:
      b["T1_Input_"+colname] = b["T1_"+colname].shift().fillna(0).astype("int32")
      b["T2_Input_"+colname] = b["T2_"+colname].shift().fillna(0).astype("int32")
    b["Input_zGameResult"] = b["zGameResult"].shift().fillna('')
    b["Input_zGameFinalScore"] = b["zGameFinalScore"].shift().fillna('')
    
  t1= tf.feature_column.categorical_column_with_vocabulary_list(
    "Team1", teamnames)
  t2 = tf.feature_column.categorical_column_with_vocabulary_list(
    "Team2", teamnames)
  where = tf.feature_column.categorical_column_with_vocabulary_list(
    "Where", ["Home"])
  t1_where= tf.feature_column.categorical_column_with_vocabulary_list(
    "HomeTeam", teamnames)
  season= tf.feature_column.categorical_column_with_vocabulary_list(
    "Season", ["1314", "1415", "1516", "1617"])
  t1 = tf.feature_column.indicator_column(t1)
  t2 = tf.feature_column.indicator_column(t2)
  where = tf.feature_column.indicator_column(where)
  t1_where = tf.feature_column.indicator_column(t1_where)
  season= tf.feature_column.indicator_column(season)

  gr = tf.feature_column.categorical_column_with_vocabulary_list(
    "zGameResult", game_results_choice.values())
  gr = tf.feature_column.indicator_column(gr)
#  gr = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
#    "GameResult", v)) for v in game_results_choice.values()]
  lgr = tf.feature_column.categorical_column_with_vocabulary_list(
    "Input_zGameResult", game_results_choice.values())
  lgr = tf.feature_column.indicator_column(lgr)

  gfs = tf.feature_column.categorical_column_with_vocabulary_list(
    "zGameFinalScore", ["0:3", "1:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"])
  gfs = tf.feature_column.indicator_column(gfs)
#  gfs = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
#    "GameFinalScore", v)) for v in ["0:3", "1:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"]]
  lgfs = tf.feature_column.categorical_column_with_vocabulary_list(
    "Input_zGameFinalScore", ["0:3", "1:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"])
  lgfs = tf.feature_column.indicator_column(lgfs)


  feature_column_specs = [t1,t2,where,season]
#  feature_column_specs = [where,season]
  feature_column_specs += [tf.feature_column.numeric_column(c) for c in feature_column_names]  
  label_column_specs = [tf.feature_column.numeric_column(c) for c in label_column_names]  

  feature_column_specs += [lgr, lgfs]  
  label_column_specs += [gr, gfs]

  if mode==tf.estimator.ModeKeys.TRAIN:
    features = create_series(batches)  
    features = pd.concat(features, ignore_index=True)
    features.to_csv("train_features.csv")
  else:
    features = create_series_eval(batches)  
    features = pd.concat(features, ignore_index=True)
    features.to_csv("test_features.csv")
    
  return features, feature_column_specs, label_column_specs

def create_series(raw_data):
  series = [[b[i:i+34] for i in range(0, len(b)-33, 17)] for b in raw_data]
  #sequence_lengths = [len(item) for sublist in series for item in sublist]
  flattened_series = [item.assign(sequence_length=len(item)) for sublist in series for item in sublist]
#  for f in flattened_series:
#    f.assign(sequence_length=len(f))
  #print(sequence_lengths)
  
  return flattened_series 

def create_series_eval(raw_data):
  series = [[b[max(i-33,0):i+1] for i in range(len(b))] for b in raw_data]
  flattened_series = [item.assign(sequence_length=len(item)) for sublist in series for item in sublist if len(item)>0 and item["Train"][item.index[-1]]==False]
  flattened_series = [s.append([s.loc[s.index[-1],:]]*(34-len(s)) if len(s)<34 else None) for s in flattened_series]
  
  return flattened_series 


def input_fn(df_data, teamnames, mode=tf.estimator.ModeKeys.TRAIN):
  features, feature_column_specs, label_column_specs  = build_features(df_data, teamnames, mode)
  return tf.estimator.inputs.pandas_input_fn(
    x=features,
    y=None,
    batch_size=len(features),
    num_epochs=None if mode==tf.estimator.ModeKeys.TRAIN else 1,
    queue_capacity=None,
    shuffle=False,
    num_threads=1,
#    target_column='dummy_label'
    )

def plot_softprob(pred):
  print("-----------------------------------------------------")
  for prefix in [""]:
    print(prefix+"Prediction")
    if not prefix+"p_pred_12" in pred:
      return
    sp = pred[prefix+"p_pred_12"]
    if not prefix+"ev_points" in pred:
      spt = pred["ev_points"]
    else:
      spt = pred[prefix+"ev_points"]
    margin_pred_prob1 = pred[prefix+"p_marg_1"]
    margin_poisson_prob1 = pred[prefix+"p_poisson_1"]
    margin_pred_prob2 = pred[prefix+"p_marg_2"]
    margin_poisson_prob2 = pred[prefix+"p_poisson_2"]
    margin_pred_expected1 = pred[prefix+"ev_goals_1"] 
    margin_pred_expected2 = pred[prefix+"ev_goals_2"] 
    g=[0,1,2,3,4,5,6]
    g1=[0]*7+[1]*7+[2]*7+[3]*7+[4]*7+[5]*7+[6]*7
    g2=g*7
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].scatter(g1, g2, s=sp*10000, alpha=0.5)
    for i, txt in enumerate(sp):
      ax[0].annotate("{:4.2f}".format(txt*100), (g1[i],g2[i]))
    ax[1].scatter(g1, g2, s=spt*500, alpha=0.4,color='red')
    for i, txt in enumerate(spt):
      ax[1].annotate("{:4.2f}".format(txt), (g1[i],g2[i]))
    plt.show()
  
    w=0.35
    fig, ax = plt.subplots(1,2,figsize=(10,1))
    ax[0].bar(g, margin_pred_prob1,alpha=0.6, width=w)
    ax[0].bar([x+w for x in g], margin_poisson_prob1,alpha=0.3,color="red",width=0.35)
    ax[0].axvline(x=margin_pred_expected1, color='red')
    ax[1].bar(g, margin_pred_prob2,alpha=0.6, width=w)
    ax[1].bar([x+w for x in g], margin_poisson_prob2,alpha=0.3,color="red",width=0.35)
    ax[1].axvline(x=margin_pred_expected2, color='red')
    plt.show()


def run_evaluation(all_data, teamnames, estimator, outputname):
  # print(file_name)
  if (outputname == "train"):
    all_data = all_data[all_data["Train"]==True].copy()
    all_data["Train"]=False  # use all data for prediction and evaluation
    
  results = estimator.evaluate(
      input_fn=input_fn(all_data, teamnames, mode=tf.estimator.ModeKeys.EVAL),
      steps=1, name=outputname)
#      
#  for key in sorted(results):
#    print("%s: %s" % (key, results[key]))

  pred_input_fn = input_fn(all_data, teamnames, mode=tf.estimator.ModeKeys.PREDICT)
  predictions3 = list(estimator.predict(pred_input_fn))

  features = build_features(all_data, teamnames, mode=tf.estimator.ModeKeys.PREDICT)[0]
  
  features = features[33::34]
  features.index = range(len(features))
  print(features.index)
  features.to_csv("newfeatures.csv")

  # predictions3 = predictions3[33::34].copy()
  index = range(0, len(features)*34, 34)
  seq_len = features["sequence_length"]
  index_list = [i+s-1 for i,s in zip(index, seq_len)]
  # predictions3 = predictions3[index_list].copy()
  predictions3 = [predictions3[i] for i in index_list]
  
  predictions3[0].update({"features": features.loc[0,:].to_dict()})
  predictions3[1].update({"features": features.loc[1,:].to_dict()})
  
  #print(predictions3[0:2])

  predictions2 = pd.Series([p["pred"][1] for p in predictions3], name="predictions2")
  predictions = pd.Series([p["pred"][0] for p in predictions3], name="predictions")
  est1 = pd.Series([p["ev_goals_1"] for p in predictions3], name="est1")
  est2 = pd.Series([p["ev_goals_2"] for p in predictions3], name="est2")
  est_diff = pd.Series([p["ev_goals_1"]-p["ev_goals_2"] for p in predictions3], name="est_diff")
  #ev_goals_x1 = pd.Series([p["ev_goals_x1"] for p in predictions3])
  #ev_goals_x2 = pd.Series([p["ev_goals_x2"] for p in predictions3])
  
  #print(predictions3[0])
  #print(predictions3[20])
#  print(predictions3[20]["p_pred_1"])
#  print(predictions3[20]["p_pred_2"])
#  print(predictions3[20]["p3"])  
#  print(predictions3[20]["probMatrix"])
#  print(predictions3[20]["p_pred_12_raw"])
#  print(predictions3[20]["p_pred_12"])
#  print(features.shape)
#  print(features)
  labels  = features["OwnGoals"] 
  labels2 = features["OpponentGoals"] 

  features.loc[:,'Predictions'] = predictions
  features.loc[:,'Predictions2'] = predictions2
  features.loc[:,'Goals'] = labels
  features.loc[:,'est1'] = est1
  features.loc[:,'est2'] = est2

  tensor = tf.constant(features[[ "Predictions", "Predictions2","OwnGoals", "OpponentGoals"]].as_matrix(), dtype = tf.int64)
  tensor2 = tf.constant(features["Where"].as_matrix(), dtype = tf.string)
  is_home = tf.equal(tensor2 , "Home")
  with tf.Session() as sess:
    points_tensor = dlm.calc_points(tensor[:,0],tensor[:,1], tensor[:,2], tensor[:,3], is_home)[0]
    features.loc[:,'Pt'] = sess.run(tf.cast(points_tensor, tf.int8))
#    staticpred = dlm.makeStaticPrediction(features)
#    plot_softprob(staticpred)
  #print(features.iloc[20])

  print(
      features[["Team1", "Team2", "OwnGoals", "OpponentGoals", "Predictions", "Predictions2", "Where", "est1","est2","Pt"]]
      .rename(columns={'OwnGoals': 'GS', 'OpponentGoals': 'GC', "Predictions":"pGS", "Predictions2":"pGC"})
#      .sort_values(["est1","est2"], ascending=[True, False])
      )
  print(
      features[["Team1", "Team2", "OwnGoals", "OpponentGoals", "Predictions", "Predictions2", "Where", "est1","est2","Pt"]]
      .rename(columns={'OwnGoals': 'GS', 'OpponentGoals': 'GC', "Predictions":"pGS", "Predictions2":"pGC"})
#      .sort_values(["est1","est2"], ascending=[True, False])
      .head(60)
      )
  print(
      features[["Team1", "Team2", "OwnGoals", "OpponentGoals", "Predictions", "Predictions2", "Where", "est1","est2","Pt"]]
      .rename(columns={'OwnGoals': 'GS', 'OpponentGoals': 'GC', "Predictions":"pGS", "Predictions2":"pGC"})
#      .sort_values(["est1","est2"], ascending=[True, False])
      .tail(200)
      )

  plot_softprob(predictions3[0])
  plot_softprob(predictions3[20])

  fig = plt.figure(figsize=(14,4))
  ax1 = plt.subplot2grid((1,3), (0,0), colspan=2, rowspan=1)
  ax2 = plt.subplot2grid((1,3), (0,2), colspan=2, rowspan=1)
  ax2.axis('off')
  
  goalstats = pd.Series([(gs,gc) for gs,gc in zip(labels, labels2)])
  goal_cnt = Counter(goalstats)
  
  #result = zip(zip(predictions,predictions2),features["Where"])#,zip(features["OwnGoals"],features["OpponentGoals"]))
  hp = ["3_full" if p1==gs and p2==gc else 
        "2_diff" if p1-p2==gs-gc else
        "1_tendency" if np.sign(p1-p2)==np.sign(gs-gc) else
        "0_none"
        for p1,p2, gs, gc in zip(predictions,predictions2,labels,labels2)]
  result = [((p1,p2),hp) if w=="Home" else ((p2,p1),hp) for p1,p2,w,hp in zip(predictions,predictions2,features["Where"],hp)]
  result = pd.DataFrame(result)
  result.columns = ['pred', 'gof']
  t1 = result.pivot_table(index = 'pred', columns=['gof'], aggfunc=len, margins=False)
  t1.loc[:,"idx"]=[(x[0]-x[1])*(1+0.1*x[0])+0.01*x[0] for x in t1.index]
  t1 = t1.sort_values(by=('idx'), ascending=True)
  t1 = t1.drop(['idx'], axis=1)
  t2 = t1.fillna(0)
  t2.loc[:,"hit%"]=t2["3_full"] *100 / (t2["0_none"]+t2["1_tendency"]+t2["2_diff"]+t2["3_full"])
  t2.loc[:,"rand%"]=[goal_cnt.get(g)*100/len(labels) if goal_cnt.get(g) else 0.0 for g in t2.index]
  t2.loc[:,"edge"]=t2.loc[:,"hit%"]-t2.loc[:,"rand%"]
  
  #tendency = [-1, -1, -1, 1, 1, 1, 1]
  tendency = [np.sign(g[0]-g[1]) for g in t2.index]
  t2.loc[:,"total_points"]=0
  t2.loc[:,"total_points"]+=[6*c3+2*c2+2*c1 if t==0 else 0 for t,c1,c2,c3 in zip(tendency, t2["1_tendency"], t2["2_diff"], t2["3_full"])]
  t2.loc[:,"total_points"]+=[4*c3+3*c2+2*c1 if t==1 else 0 for t,c1,c2,c3 in zip(tendency, t2["1_tendency"], t2["2_diff"], t2["3_full"])]
  t2.loc[:,"total_points"]+=[7*c3+5*c2+4*c1 if t==-1 else 0 for t,c1,c2,c3 in zip(tendency, t2["1_tendency"], t2["2_diff"], t2["3_full"])]
  t2.loc[:,"avg_points"]=t2["total_points"] / (t2["0_none"]+t2["1_tendency"]+t2["2_diff"]+t2["3_full"])
  t2.loc[:,"contribution"]=t2["total_points"] / len(labels)
  print(t2)
  t1.plot(kind='bar', stacked=True, ax=ax1)

  cnt = Counter(hp)
  #plt.figure(figsize=(3,3))
  #plt.pie([float(v) for v in cnt.values()], labels=cnt.keys(), startangle=90)
  pie_chart_values = [cnt.get("0_none"), cnt.get("1_tendency"), cnt.get("2_diff"), cnt.get("3_full")] 
  pie_chart_values = [0 if v is None else v for v in pie_chart_values ]
  ax2.pie(pie_chart_values, 
          labels=["None", "Tendency", "Diff", "Full"], 
          startangle=90)
  plt.show()
  plt.close()
  
  
  fig, ax = plt.subplots(1,2,figsize=(12,4))
  ax[0].scatter(est1, labels,alpha=0.1)
  ax[1].scatter(est2, labels2,alpha=0.1)
  plt.show()
  fig, ax = plt.subplots(1,2,figsize=(12,4))
  ax[0].scatter(est_diff, labels-labels2,alpha=0.1)
  c_home = features["Where"]=="Home"
  c_win  = features['Predictions'] > features['Predictions2']
  c_loss = features['Predictions'] < features['Predictions2']
  c_draw = features['Predictions'] == features['Predictions2']
  c_tendency = np.sign(features['Predictions']- features['Predictions2']) == np.sign(labels - labels2) 

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
  
  return results



def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  tf.logging.set_verbosity(tf.logging.INFO)
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
  
  all_data, teamnames = get_train_test_data(model_dir, train_data, test_data)
  _, feature_columns, label_columns = build_features(all_data.copy(), teamnames)
  #test_input_function = input_fn(all_data, teamnames, mode=tf.estimator.ModeKeys.TRAIN)
  #print(test_input_function())
  model = lstm.create_estimator(model_dir, feature_columns, label_columns, teamnames)
  print(model)
#  with tf.Session() as sess:
#    plot_softprob(drm.makeStaticPrediction(build_features(train_data)[0]))
#    plot_softprob(drm.makeStaticPrediction(build_features(test_data)[0]))
#  print(feature_columns)
#  print(teamnames)

  for i in range(train_steps//100):
  
    model.train(
        input_fn=input_fn(all_data.copy(), teamnames, mode=tf.estimator.ModeKeys.TRAIN),
        steps=500) #, hooks=[summary_hook]) #hooks=[MyHook(teamnames)])#, 
    # set steps to None to run evaluation until all data consumed.
    
    test_result = run_evaluation(all_data.copy(), teamnames, model, outputname="test")
    train_result = run_evaluation(all_data.copy(), teamnames, model, outputname="train")
    
    results = pd.DataFrame()
    results["Measure"] = test_result.keys()
    results["Train"] = train_result.values()
    results["Test"] = test_result.values()
    results["Diff abs"] = results["Train"] - results["Test"]
    results["Test %"] = results["Test"] / results["Train"] *100
    results = results.sort_values(by="Measure")
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
                 FLAGS.train_data, FLAGS.test_data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="C:/tmp/Football/models", # always use UNIX-style path names!!!
      help="Base directory for output models."
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
      default=["1314", "1415", "1516"],
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default=["1617"],
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# Path('C:/tmp/Football/models/reset.txt').touch()