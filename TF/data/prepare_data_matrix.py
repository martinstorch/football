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
import pandas as pd
pd.set_option('expand_frame_repr', False)

import numpy as np

#np.set_printoptions(threshold=50)
from datetime import datetime
import os

from collections import Counter
#from pathlib import Path
from sklearn import preprocessing

dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)


Feature_COLUMNS = ["HomeTeam","AwayTeam"]
Label_COLUMNS = ["FTHG","FTAG"]
CSV_COLUMNS = Feature_COLUMNS + Label_COLUMNS
Derived_COLUMNS = ["t1goals", "t2goals", "t1goals_where", "t2goals_where"]
COLS = ["HGFT","AGFT","HGHT","AGHT","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR", "HxG", "AxG", "HGFTa","AGFTa","Hxsg","Axsg","Hxnsg","Axnsg", "Hspi","Aspi","Himp","Aimp","HGFTe","AGFTe","H_PT","A_PT"]
Meta_COLUMNS = ["t1games", "t2games", "t1games_where", "t2games_where"]
COLS_Extended = COLS + ['HWin', 'AWin', 'HLoss', 'ALoss', 'HDraw', 'ADraw']

SEQ_LENGTH = 10
TIMESERIES_COL = 'rawdata'

#df_data = pd.read_csv(dir_path+"/full_data.csv")


def build_features(df_data):
  print(df_data.shape)
  print(df_data["HomeTeam"].unique())
  team_encoder = preprocessing.LabelEncoder()
  team_encoder.fit(df_data["HomeTeam"])
  print(team_encoder.classes_)
  print(team_encoder.transform(team_encoder.classes_))

  print(df_data["Date"].tail(18))
  df_data["DateOrig"] = df_data["Date"]
  df_data["Date"]= [(datetime.strptime(dt, '%d.%m.%Y').toordinal()-734138) for dt in df_data["Date"]]
  df_data.sort_values("Date").reset_index(drop=True)
  df_data.replace(-1, np.nan, inplace=True)
  df_data.replace("-1", np.nan, inplace=True)

  # use FTHG / FTAG as xHG/xAG where expected goals are not available
  df_data.loc[pd.isna(df_data.xHG), "xHG"]=df_data.loc[pd.isna(df_data.xHG), "FTHG"]
  df_data.loc[pd.isna(df_data.xAG), "xAG"]=df_data.loc[pd.isna(df_data.xAG), "FTAG"]
  df_data.loc[pd.isna(df_data.HGFTa), "HGFTa"]=df_data.loc[pd.isna(df_data.HGFTa), "FTHG"]*1.05
  df_data.loc[pd.isna(df_data.AGFTa), "AGFTa"]=df_data.loc[pd.isna(df_data.AGFTa), "FTAG"]*1.05
  df_data.loc[pd.isna(df_data.Hxsg), "Hxsg"]=df_data.loc[pd.isna(df_data.Hxsg), "FTHG"]/2
  df_data.loc[pd.isna(df_data.Axsg), "Axsg"]=df_data.loc[pd.isna(df_data.Axsg), "FTAG"]/2
  df_data.loc[pd.isna(df_data.Hxnsg), "Hxnsg"]=df_data.loc[pd.isna(df_data.Hxnsg), "FTHG"]/2
  df_data.loc[pd.isna(df_data.Axnsg), "Axnsg"]=df_data.loc[pd.isna(df_data.Axnsg), "FTAG"]/2
  df_data.loc[pd.isna(df_data.Time), "Time"]="15:30"
  df_data["DateOrig"] = pd.to_datetime(df_data['DateOrig'] + ' ' + df_data['Time'])
  df_data["Time"] = df_data["Time"].str.slice(0,2).astype(float)+1/60*df_data["Time"].str.slice(3,5).astype(float)-14.5
  df_data.loc[pd.isna(df_data.H_PT), "H_PT"] = 50
  df_data.loc[pd.isna(df_data.A_PT), "A_PT"] = 50
  df_data.H_PT /= 100 # percentage to 0..1
  df_data.A_PT /= 100 # percentage to 0..1
  def to_logit(p):
    return np.log(p / (1 - p))
  df_data.H_PT = to_logit(df_data.H_PT)
  df_data.A_PT = to_logit(df_data.A_PT)

  #fill NA with mean() of each column in boston dataset
  #print(df_data.isna().any())
  #df_data = df_data.apply(lambda x: x.fillna(x.mean()),axis=0)
  df_data.fillna(value=0, inplace=True)

  df_data.rename(columns={
      'FTHG': 'HGFT', 'FTAG': 'AGFT',
      'HTHG': 'HGHT', 'HTAG': 'AGHT',
      'xHG':'HxG', 'xAG':'AxG'
      }, inplace=True)

  df1 = pd.DataFrame()
  df1["Team1"] = df_data["HomeTeam"]
  df1["Team2"] = df_data["AwayTeam"]
  df1["Where"] = 1
  df1['OpponentGoals'] = df_data["AGFT"]
  df1['OwnGoals'] = df_data["HGFT"]
  df1['HomeTeam'] = df1["Team1"]
  df1['Season'] = df_data["Season"]
  #df1["Train"] = df_data["Train"]
  df1["Date"]= df_data["Date"]
  df1["DateOrig"]= df_data["DateOrig"]
  df1["Dow"]= df_data["Dow"]
  df1["Time"]= df_data["Time"]
  df1["Predict"]= df_data["Predict"]

  df1['BW1'] = 1/df_data["BWH"]
  df1['BW2'] = 1/df_data["BWA"]
  df1['BW0'] = 1/df_data["BWD"]

  df1['pp1'] = df_data["ppH"]
  df1['pp2'] = df_data["ppA"]
  df1['pp0'] = df_data["ppD"]
#  df1["T1_xG"] = df_data["xHG"]
#  df1["T2_xG"] = df_data["xAG"]

  df2 = pd.DataFrame()
  df2["Team1"] = df_data["AwayTeam"]
  df2["Team2"] = df_data["HomeTeam"]
  df2["Where"] = 0
  df2['OpponentGoals'] = df_data["HGFT"]
  df2['OwnGoals'] = df_data["AGFT"]
  df2['HomeTeam'] = df1["Team2"]
  df2['Season'] = df_data["Season"]
  #df2["Train"] = df_data["Train"]
  df2["Date"]= df_data["Date"]
  df2["DateOrig"]= df_data["DateOrig"]
  df2["Dow"]= df_data["Dow"]
  df2["Time"]= df_data["Time"]
  df2["Predict"]= df_data["Predict"]

  df2['BW2'] = 1/df_data["BWH"]
  df2['BW1'] = 1/df_data["BWA"]
  df2['BW0'] = 1/df_data["BWD"]

  df2['pp1'] = df_data["ppA"]
  df2['pp2'] = df_data["ppH"]
  df2['pp0'] = df_data["ppD"]

#  df2["T1_xG"] = df_data["xAG"]
#  df2["T2_xG"] = df_data["xHG"]

  columns = [c[1:] for c in COLS[::2]]
#  feature_column_names_fixed = ["Team1", "Team2", "Where", "Season", "Train"]
  label_column_names = []
  for colname in columns:
    print(colname)
    homecol="H"+colname
    awaycol="A"+colname
    df1["T1_"+colname] = df_data[homecol]
    df1["T2_"+colname] = df_data[awaycol]
    df2["T1_"+colname] = df_data[awaycol]
    df2["T2_"+colname] = df_data[homecol]
    label_column_names += ["T1_"+colname, "T2_"+colname]

  label_column_names = label_column_names[:-6] # cut off ["Hspi","Aspi","Himp","Aimp","HGFTe","AGFTe"] # these are not features, not labels

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
  features = pd.concat([df1,df2], ignore_index=False)
  features = features.sort_index()

  print(features[["BW1", "BW0", "BW2"]])
  #print(features[["B365_1", "B365_0", "B365_2"]])
  print(np.sum(features[["BW1", "BW0", "BW2"]], axis=1))
  # normalized Bet&Win probabilities
  bwsum = np.sum(features[["BW1", "BW0", "BW2"]], axis=1).values
  features["BW1"] /= bwsum
  features["BW0"] /= bwsum
  features["BW2"] /= bwsum

  # derived feature 2nd half goals
  features["T1_GH2"] = features["T1_GFT"] - features["T1_GHT"]
  features["T2_GH2"] = features["T2_GFT"] - features["T2_GHT"]
  label_column_names += ["T1_GH2", "T2_GH2"]
  columns += ["GH2"]

  features["t1cards"] = features["T1_R"]+0.25*features["T1_Y"]
  features["t2cards"] = features["T2_R"]+0.25*features["T2_Y"]

  print(features[["BW1", "BW0", "BW2"]])
  #print(features[["B365_1", "B365_0", "B365_2"]])

  # derived feature full time win/loss/draw
  features["zGameResult"] = [np.sign(x1-x2) for x1,x2 in zip(features["T1_GFT"], features["T2_GFT"])]
  features["zGameHTResult"] = [np.sign(x1-x2) for x1,x2 in zip(features["T1_GHT"], features["T2_GHT"])]
  features["zGameHT2Result"] = [np.sign(x1-x2) for x1,x2 in zip(features["T1_GH2"], features["T2_GH2"])]
  features["zGameFinalScore"] = [str(x1)+":"+str(x2) for x1,x2 in zip(features["T1_GFT"].astype(int), features["T2_GFT"].astype(int))]
  features["zGamePoints1"] = [x+1 if x<=0 else 3 for x in features["zGameResult"]]
  features["zGamePoints2"] = [1-x if x>=0 else 3 for x in features["zGameResult"]]

  # feature scaling
#  features["T1_S"] /= 15.0
#  features["T2_S"] /= 15.0
#  features["T1_ST"] /= 5.0
#  features["T2_ST"] /= 5.0
#  features["T1_F"] /= 15.0
#  features["T2_F"] /= 15.0
#  features["T1_C"] /= 5.0
#  features["T2_C"] /= 5.0
  features["Date"] /= 1000.0

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

  gr2h = pd.get_dummies(features["zGameHT2Result"])
  col_names = ['HT2Loss', 'HT2Draw', 'HT2Win']
  gr2h.columns = col_names
  features = pd.concat([features, gr2h], axis=1)
  label_column_names += col_names

  features["Team1_index"] = team_encoder.transform(features["Team1"])
  features["Team2_index"] = team_encoder.transform(features["Team2"])


  final_score_enum = ["0:3", "1:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0", "2:2", "3:2", "2:3", "3:3"]
  final_score_encoder = preprocessing.LabelBinarizer(sparse_output=False)
  final_score_encoder.fit_transform(final_score_enum)
  label_column_names += ["zScore"+s for s in final_score_enum]
  print(Counter(features["zGameFinalScore"]))
  for s in final_score_enum :
    g1=int(s[0])
    g2=int(s[2])
    matches1H = [1 if g1<=t1 and g2<=t2 else 0 for t1, t2 in zip(features["T1_GHT"], features["T2_GHT"])]
    matches2H = [1 if t11<=g1 and g1<=t12 and t21<=g2 and g2<=t22 else 0 for t11, t21, t12, t22 in zip(features["T1_GHT"], features["T2_GHT"], features["T1_GFT"], features["T2_GFT"])]
    features["zScore"+s] =  [(1 if m1==1 or m2==1 else 0)+(1 if fs==s else 0) for m1, m2, fs in zip(matches1H, matches2H, features["zGameFinalScore"])]
  #features["zScore0:0"].value_counts()
#  print(features.iloc[-20])
#  print(features.iloc[-21])
#  print(features.iloc[-22])
#  print(features.iloc[-23])
  # derived feature >3 goals
  features["FTG4"] = [1 if t1+t2>=4 else 0 for t1,t2 in zip(features["T1_GFT"], features["T2_GFT"])]
  label_column_names += ["FTG4"]
  features["FTG0"] = [1 if t1==0 or t2==0 else 0 for t1,t2 in zip(features["T1_GFT"], features["T2_GFT"])]
  label_column_names += ["FTG0"]
  features["HTG0"] = [1 if t1==0 or t2==0 else 0 for t1,t2 in zip(features["T1_GHT"], features["T2_GHT"])]
  label_column_names += ["HTG0"]

  #features["Season"]
  #features["Team1_index"]
  gt1 = features.groupby(["Season", "Team1_index"])
  gt2 = features.groupby(["Season", "Team2_index"])
  gt1w = features.groupby(["Season", "Team1_index","Where"])
  gt2w = features.groupby(["Season", "Team2_index","Where"])
  gtt1 = features.groupby(["Team1_index"])
  gtt2 = features.groupby(["Team2_index"])

  halftime_ema = "30 days"
  features["t1games"] = gt1.cumcount()
  features["t2games"] = gt2.cumcount()
  features["roundsleft"] = 34-features["t1games"]
  features["t1goals"] = (gt1["OwnGoals"].cumsum()-features["OwnGoals"])/(features["t1games"]+2)
  features["t2goals"] = (gt2["OpponentGoals"].cumsum()-features["OpponentGoals"])/(features["t2games"]+2)
  features["t12goals"] = features["t1goals"] - features["t2goals"]
  features.loc[features["t1games"]==0, "t1goals"] = 1.45
  features.loc[features["t2games"]==0, "t2goals"] = 1.45
  features["t1points"] = gt1["zGamePoints1"].cumsum()-features["zGamePoints1"]
  features["t2points"] = gt2["zGamePoints2"].cumsum()-features["zGamePoints2"]
  features["t1rank"] = features.groupby(["Season", "t1games"])["t1points"].rank(ascending=False)
  features["t2rank"] = features.groupby(["Season", "t2games"])["t2points"].rank(ascending=False)
  features["t1rank6_attention"] = [np.exp(-0.1*np.square(x-6)) for x in features["t1rank"]]
  features["t2rank6_attention"] = [np.exp(-0.1*np.square(x-6)) for x in features["t2rank"]]
  features["t1rank16_attention"] = [np.exp(-0.1*np.square(x-16)) for x in features["t1rank"]]
  features["t2rank16_attention"] = [np.exp(-0.1*np.square(x-16)) for x in features["t2rank"]]
  features["t1goals_ema"] = gtt1.OwnGoals.apply(lambda x: x.shift(1).ewm(halflife=halftime_ema, times=features.loc[x.index].DateOrig.shift(1,  fill_value=datetime(2000, 1, 1))).mean())
  features["t2goals_ema"] = gtt2.OwnGoals.apply(lambda x: x.shift(1).ewm(halflife=halftime_ema, times=features.loc[x.index].DateOrig.shift(1,  fill_value=datetime(2000, 1, 1))).mean())

  features["t1cards_ema"] = gt1.t1cards.apply(lambda x: x.shift(1).ewm(halflife=2).mean())
  features["t2cards_ema"] = gt2.t2cards.apply(lambda x: x.shift(1).ewm(halflife=2).mean())
  features.fillna(0,  inplace=True)

  features["t1games_where"] = gt1w.cumcount()
  features["t2games_where"] = gt2w.cumcount()

  feature_column_names = ["Where", "Predict", "Team1", "Team2", "Team1_index", "Team2_index", "Date", "Time", "t1games",
                          "t1dayssince", "t2dayssince", "t1dayssince_ema", "t2dayssince_ema",
                          "roundsleft", "t1promoted", "t2promoted"]
  feature_column_names += ["t1points", "t2points", "t1rank", "t2rank",
                           "t1rank6_attention", "t2rank6_attention", "t1rank16_attention", "t2rank16_attention",
                           "t1cards_ema", "t2cards_ema", "t1goals_ema", "t2goals_ema"]

  feature_column_names += ["T1_spi", "T2_spi", "T1_imp", "T2_imp", "T1_GFTe", "T2_GFTe", "pp1", "pp0", "pp2"]
#  print(features[["Date", "t1games", "t1points", "t2points", "t1rank", "t2rank",
#                           "t1cards", "t2cards",
#                           "t1cards_ema", "t2cards_ema",
#                           "Team1", "Team2"]])

  use_bwin_statistics = True
  if use_bwin_statistics:
    feature_column_names += ["BW1", "BW0", "BW2"]

  print(columns)
  print(features.columns)
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
    features["T1_EMA_"+colname] = gtt1["T1_"+colname].apply(
      lambda x: x.shift(1).ewm(halflife=halftime_ema, times=features.loc[x.index].DateOrig.shift(1,  fill_value=datetime(2000, 1, 1))).mean())
    features["T2_EMA_"+colname] = gtt2["T2_"+colname].apply(
      lambda x: x.shift(1).ewm(halflife=halftime_ema, times=features.loc[x.index].DateOrig.shift(1,  fill_value=datetime(2000, 1, 1))).mean())

    feature_column_names += ["T1_CUM_T1_"+colname, "T2_CUM_T2_"+colname,
                             "T1_EMA_" + colname, "T2_EMA_"+colname,
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


  #features.columns.values
  # feature scaling
  features["t1games"] /= 34.0
  features["t2games"] /= 34.0
  features["roundsleft"] /= 34.0
  # build feature such that winter games near mid-season can be distinguished from summer. Likelyhood of draw results increases in mid-season ...
  features["t1games"] = (features["t1games"]-0.5)**2

  gt1_total = features.groupby(["Team1_index"])
  gt2_total = features.groupby(["Team2_index"])

  features["t1games_total"] = gt1_total.cumcount()
  features["t2games_total"] = gt2_total.cumcount()

  team_onehot_encoder = preprocessing.LabelBinarizer(sparse_output=False)
  team_onehot_encoder .fit(features["Team1"])
  team1oh = team_onehot_encoder.transform(features["Team1"])
  team2oh = team_onehot_encoder.transform(features["Team2"])
  teamsoh = np.concatenate([team1oh, team2oh], axis=1)

#  features["t1dayssince"] = [d-features.loc[(features["Team1_index"]==t)&(features["t1games_total"]==i-1), "Date"] for t,d,i in zip(features["Team1_index"],features["Date"],features["t1games_total"])]
#  features["t1dayssince"] = [x.iloc[0] if len(x)==1 else 0.0 for x in features["t1dayssince"]]
#
#  features["t2dayssince"] = [d-features.loc[(features["Team2_index"]==t)&(features["t2games_total"]==i-1), "Date"] for t,d,i in zip(features["Team2_index"],features["Date"],features["t2games_total"])]
#  features["t2dayssince"] = [x.iloc[0] if len(x)==1 else 0.0 for x in features["t2dayssince"]]

  t1ds = features["Date"]-features.groupby("Team1_index").Date.shift(1)
  t1ds.fillna(t1ds.mean(), inplace=True)
  t1ds = np.minimum(t1ds, 0.03)
  features["t1dayssince"] = t1ds

  t2ds = features["Date"]-features.groupby("Team2_index").Date.shift(1)
  t2ds.fillna(t2ds.mean(), inplace=True)
  t2ds = np.minimum(t2ds, 0.03)
  features["t2dayssince"] = t2ds

  features["t1dayssince_ema"] = gt1.t1dayssince.apply(lambda x: x.shift(1).ewm(halflife=2).mean())
  features["t2dayssince_ema"] = gt2.t2dayssince.apply(lambda x: x.shift(1).ewm(halflife=2).mean())
  features.fillna(0,  inplace=True)

  t1new = features["Season"]-features.groupby("Team1_index").Season.shift(34)
  t1new.fillna(0, inplace=True)
  t1new = 1.0-(t1new==101)
  #print(features.loc[t1new==1, ["Season","Team1"]].drop_duplicates())
  features["t1promoted"]=t1new

  t2new = features["Season"]-features.groupby("Team2_index").Season.shift(34)
  t2new.fillna(0, inplace=True)
  t2new = 1.0-(t2new==101)
  #print(features.loc[t2new==1, ["Season","Team2"]].drop_duplicates())
  features["t2promoted"]=t2new

  dow = pd.get_dummies(features["Dow"])
  features = pd.concat([features, dow], axis=1)
  feature_column_names += list(dow.columns)

  gr_all = features.loc[:, label_column_names + ["Where"]].reset_index().groupby(["Where"])
  rs = gr_all.rolling(11*9).sum() - gr_all.rolling(1*9).sum()
  rs = rs.fillna(rs.groupby(level=0).mean())/ 90  # impute group-level means for first matches
  rolling_sums = rs.droplevel(level=0).sort_index()
  last10 = rolling_sums
  last10["i"] = last10.index // 18

  last10 = last10.groupby(["Where", "i"]).last()
  last10 = last10.droplevel(level=0).sort_index().reset_index().drop(columns=["i", "index"])
  repeat_index = (features.index // 18)*2 + features.Where
  last10 = last10.loc[repeat_index].reset_index().drop(columns=["index"])
  print(last10.shape)
  print(features.shape)
  last10.columns = ["L10_" + c for c in list(last10.columns)]
  feature_column_names += list(last10.columns)
  features = pd.concat([features, last10], axis=1)

  # make sure that xG labels are the last ones in the list, so that tensorflow can apply MSE loss instead of Poisson deviance loss
  label_column_names.remove("T1_xG")
  label_column_names.remove("T2_xG")
  label_column_names.remove("T1_GFTa")
  label_column_names.remove("T2_GFTa")
  label_column_names.remove("T1_xsg")
  label_column_names.remove("T2_xsg")
  label_column_names.remove("T1_xnsg")
  label_column_names.remove("T2_xnsg")

  label_column_names = label_column_names+["T1_GFTa","T2_GFTa","T1_xsg","T2_xsg","T1_xnsg","T2_xnsg", "T1_xG", "T2_xG", "T1__PT", "T2__PT"]
  labels_df =  features[label_column_names].copy()
  #label_df.describe()
  features_df = features[feature_column_names].copy()
  #features_df .describe()
  features_df .mean()
  print(feature_column_names)
  print(features[label_column_names].mean())

  return features_df , labels_df, teamsoh, team_onehot_encoder, team_encoder



df_data = pd.read_csv(dir_path+"/full_data.csv")
results = build_features(df_data)
features_df , labels_df, teamsoh, team_onehot_encoder, team_encoder = results

features_df.to_csv(dir_path+"/all_features.csv", index=False)
labels_df.to_csv(dir_path+"/all_labels.csv", index=False)
pd.DataFrame(teamsoh).to_csv(dir_path+"/teams_onehot.csv")

team_mapping = pd.DataFrame({"Teamname":team_encoder.classes_, 
              "TeamID":team_encoder.transform(team_encoder.classes_)})
team_mapping.to_csv(dir_path+"/team_mapping.csv", index=False) 

#features_df["t1dayssince"].describe()
#features_df["t1promoted"].value_counts()
