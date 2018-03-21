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
import numpy as np
from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
from Estimators import LinearModel as lm
from Estimators import PoissonModel_teamcentric as pm
from Estimators import DiscreteModel as dm
from Estimators import DiscreteModelMulti as dmm
from tensorflow.python.training.session_run_hook import SessionRunHook
from collections import Counter

Feature_COLUMNS = ["HomeTeam","AwayTeam"]
Label_COLUMNS = ["FTHG","FTAG"]
CSV_COLUMNS = Feature_COLUMNS + Label_COLUMNS


def makeColumns(teamnames):
  t1= tf.feature_column.categorical_column_with_vocabulary_list(
    "Team1", teamnames)
  t2 = tf.feature_column.categorical_column_with_vocabulary_list(
    "Team2", teamnames)
  where = tf.feature_column.categorical_column_with_vocabulary_list(
    "Where", ["Home"])
  t1_where= tf.feature_column.categorical_column_with_vocabulary_list(
    "HomeTeam", teamnames)
  #t1_where =  tf.feature_column.crossed_column(["Team1","Where"], hash_bucket_size=len(teamnames))
  #t2_where =  tf.feature_column.crossed_column(["Team2","Where"], hash_bucket_size=len(teamnames))

  t1 = tf.feature_column.indicator_column(t1)
  t2 = tf.feature_column.indicator_column(t2)
  where = tf.feature_column.indicator_column(where)
  t1_where = tf.feature_column.indicator_column(t1_where)
  #t2_where = tf.feature_column.indicator_column(t2_where)
  # Continuous base columns.
  hg = tf.feature_column.numeric_column("FTHG")
  ag = tf.feature_column.numeric_column("FTAG")

  opg = tf.feature_column.numeric_column("OpponentGoals")
  opg = tf.feature_column.bucketized_column(opg, boundaries = [1,2,3,4,5,10])
  og = tf.feature_column.numeric_column("OwnGoals")
  draw = tf.feature_column.numeric_column("IsDraw")
  return [[t1, t2,where,t1_where], [hg, ag]]


## Transformations.
#age_buckets = tf.feature_column.bucketized_column(
#    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
#
## Wide columns and deep columns.
#base_columns = [
#    gender, education, marital_status, relationship, workclass, occupation,
#    native_country, age_buckets,
#]
#
#crossed_columns = [
#    tf.feature_column.crossed_column(
#        ["education", "occupation"], hash_bucket_size=1000),
#    tf.feature_column.crossed_column(
#        [age_buckets, "education", "occupation"], hash_bucket_size=1000),
#    tf.feature_column.crossed_column(
#        ["native_country", "occupation"], hash_bucket_size=1000)
#]

def download_data(model_dir, season):
    """Maybe downloads training data and returns train and test file names."""
    file_name = model_dir + "/" + season + ".csv"
    urllib.request.urlretrieve(
        "http://www.football-data.co.uk/mmz4281/"+season+"/D1.csv",
        file_name)  # pylint: disable=line-too-long
    
    #train_file_name = train_file.name
    #train_file.close()
    print("Data is downloaded to %s" % file_name)
    data = pd.read_csv(
      tf.gfile.Open(file_name),
      skipinitialspace=True,
      engine="python",
      skiprows=0)
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
  return train_data, test_data, teamnames


def build_estimator(model_dir, model_type, columns):
 # r = tf.estimator.RunConfig() #.replace(save_checkpoints_steps=10,save_summary_steps=1,log_step_count_steps=1,keep_checkpoint_max=100)
  
  #tf.estimator.RunConfig(save_checkpoints_steps=10,save_summary_steps=1)
#  r.replace({'save_checkpoints_steps': 10})
#  r.replace('save_summary_steps', 1)
  """Build an estimator."""
  if model_type == "wide":
#    m = tf.estimator.LinearClassifier(
#        model_dir=model_dir, feature_columns=columns)
    m = tf.estimator.LinearRegressor(
        model_dir=model_dir, feature_columns=columns)
  elif model_type == "deep":
    m = tf.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=columns,
        hidden_units=[100, 50])
  elif model_type == "own":
    m = dmm.create_estimator(model_dir=model_dir, columns=columns)
  elif model_type=='poisson':
    m = pm.create_estimator(model_dir=model_dir, columns=columns)
  else:
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=columns,
        dnn_feature_columns=columns,
        dnn_hidden_units=[100, 50])
  return m

def build_features(df_data):
  # features, labels = fetch_data(data_file)
  df1 = pd.DataFrame()
  df1["Team1"] = df_data["HomeTeam"]
  df1["Team2"] = df_data["AwayTeam"]
  df1["Where"] = "Home"
  df1['OpponentGoals'] = df_data["FTAG"]
  df1['OwnGoals'] = df_data["FTHG"]
  df1['HomeTeam'] = df1["Team1"]
  
  df2 = pd.DataFrame()
  df2["Team1"] = df_data["AwayTeam"]
  df2["Team2"] = df_data["HomeTeam"]
  df2["Where"] = "Away"
  df2['OpponentGoals'] = df_data["FTHG"]
  df2['OwnGoals'] = df_data["FTAG"]
  df2['HomeTeam'] = df1["Team2"]
  #df2['HomeTeam'] = ""
  #df2.assign(IsDraw = lambda x: pd.Categorical(x["OwnGoals"]==x["OpponentGoals"]))

  lb1 = pd.DataFrame()
  lb1['Goals'] = df_data["FTHG"]
  lb2 = pd.DataFrame()
  lb2['Goals'] = df_data["FTAG"]
  
  features = pd.concat([df1,df2], ignore_index=True)
  features['IsDraw'] = features["OwnGoals"]==features["OpponentGoals"]
  labels = pd.concat([lb1,lb2], ignore_index=True)
  return features, labels  

def input_fn(df_data, num_epochs, shuffle):
  """Input builder function."""
  features, labels = build_features(df_data)
  print(features.shape)
  print(labels.shape)
  return tf.estimator.inputs.pandas_input_fn(
      x=features,
      y=labels,
      batch_size=300,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=1,
      target_column='Goals'
      )
    
class MyHook(SessionRunHook):
  def __init__(self, teamnames):
    self._teamnames = teamnames

  #def _teamnames 
  
  def after_create_session(self, session,coord):
    print("hello")
    Wt1 = session.graph.get_tensor_by_name("Linear/WTeam1_indicator:0")
    Wt2 = session.graph.get_tensor_by_name("Linear/WTeam2_indicator:0")
    Wt1w = session.graph.get_tensor_by_name("Linear/WHomeTeam_indicator:0")
#    Wt2w = session.graph.get_tensor_by_name("Linear/WTeam2_X_Where_indicator:0")
    W = session.graph.get_tensor_by_name("Linear/WWhere_indicator:0")
    b = session.graph.get_tensor_by_name("Linear/b:0")
    W,Wt1,Wt2,Wt1w,b = session.run([W,Wt1,Wt2,Wt1w,b])
#    W = W[:,0]  
    print(b)
    print(W)
#    print(Wt1)
#    print(Wt2)
#    print(Wt1w)
#    print(Wt2w)
    print(pd.DataFrame({'Team1': self._teamnames, 'Attack':Wt1[:,0], 'Defense':Wt2[:,0], 'HomeAttack':Wt1w[:,0]}))
    print(pd.DataFrame({'Team2': self._teamnames, 'Defense':Wt1[:,5], 'Attack':Wt2[:,5], 'AwayDefense':Wt1w[:,5]}))
#    print(b.shape)
#    print(W.shape)
#    print(pd.DataFrame({'AwayTeam': self._teamnames, 'weights':(W[0:len(self._teamnames)])}))
#    print()
#    print(pd.DataFrame({'HomeTeam': self._teamnames, 'weights':(W[len(self._teamnames):len(self._teamnames)*2])}))


def run_evaluation(df_data, estimator, outputname):
  # print(file_name)
  results = estimator.evaluate(
      input_fn=input_fn(df_data, num_epochs=1, shuffle=False),
      steps=None, name=outputname)
      
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

  pred_fn = input_fn(df_data, num_epochs=1, shuffle=False)
  predictions3 = list(estimator.predict(pred_fn))
  print(predictions3[1])

  predictions2 = pd.Series([p["predictions2"] for p in predictions3], name="predictions2")
  predictions = pd.Series([p["predictions"] for p in predictions3], name="predictions")
  est1 = pd.Series([p["est1"] for p in predictions3], name="est1")
  est2 = pd.Series([p["est2"] for p in predictions3], name="est2")
  
  features, labels = build_features(df_data)
  labels2 = features["OpponentGoals"]

  result = zip(zip(predictions,predictions2),features["Where"])#,zip(features["OwnGoals"],features["OpponentGoals"]))
  print(result)
  
  freq = Counter(result)
  print(freq)
  df = pd.DataFrame.from_dict(freq, orient='index')
  df.plot(kind='bar')
  plt.show()
  #predictions = 
  # features, labels = fetch_data(file_name)
#  plt.scatter(est1[:,0], labels,alpha=0.2)
#  plt.savefig('plot.png')
#  plt.show()
#  plt.scatter(est2[:,0], labels2,alpha=0.2)
#  plt.show()
  plt.close()
  features['Predictions'] = predictions
  features['Predictions2'] = predictions2
  features['Goals'] = labels
  
  print(features.iloc[0])
  print(
      features[["Team1", "Team2", "OwnGoals", "OpponentGoals", "Predictions", "Predictions2", "Where"]]
      .rename(columns={'OwnGoals': 'GS', 'OpponentGoals': 'GC', "Predictions":"pGS", "Predictions2":"pGC"}))
  

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  tf.logging.set_verbosity(tf.logging.INFO)
  """Train and evaluate the model."""
  #train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print(model_dir)
  
  train_data, test_data, teamnames = get_train_test_data(model_dir, train_data, test_data)
  
  feature_columns, label_columns = makeColumns(teamnames)
#  print(feature_columns)
#  print(teamnames)
  
  m = build_estimator(model_dir, model_type, feature_columns)
  # set num_epochs to None to get infinite stream of data.
  #summary_hook = tf.train.SummarySaverHook(save_steps=10,output_dir=FLAGS.model_dir,summary_op = tf.summary.merge_all())
  #checkpoint_hook = tf.train.CheckpointSaverHook(FLAGS.model_dir, save_steps=1000)
  m.train(
      input_fn=input_fn(train_data, num_epochs=None, shuffle=True),
      steps=train_steps, hooks=[MyHook(teamnames)])#, hooks=[summary_hook])
  # set steps to None to run evaluation until all data consumed.
  
  run_evaluation(train_data, m, outputname="train")
  run_evaluation(test_data, m, outputname="test")

#  print(feature_columns)
  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
#  print(feature_spec)
  sir = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
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
      default=200,
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
