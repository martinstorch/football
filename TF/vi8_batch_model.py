    # -*- coding: utf-8 -*-
    """
    Created on Mon Jan 27 17:23:02 2020
    
    @author: marti
    """
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, cohen_kappa_score, \
        balanced_accuracy_score, classification_report, log_loss
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # from ngboost.scores import CRPS, MLE
    # from ngboost import NGBClassifier, NGBRegressor
    # from ngboost.distns import k_categorical
    # from ngboost.learners import default_tree_learner
    from scipy.stats import poisson

    import seaborn as sns
    import re
    import argparse
    # import shutil
    import sys
    import tempfile
    import pickle
    import datetime

    import csv
    import pandas as pd

    pd.set_option('expand_frame_repr', False)
    pd.options.display.float_format = '{:,.4f}'.format
    pd.options.display.max_rows = 30
    pd.options.display.min_rows = 20
    import numpy as np

    np.set_printoptions(threshold=50)
    np.set_printoptions(linewidth=200)

    from datetime import datetime
    import os
    from threading import Event

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.lines as lines
    from matplotlib import collections  as mc

    # from tensorflow.python.training.session_run_hook import SessionRunHook
    # from tensorflow.contrib.layers import l2_regularizer

    from collections import Counter
    # from pathlib import Path
    from sklearn.metrics import confusion_matrix
    from sklearn import preprocessing

    import random
    import itertools
    import copy

    import tensorflow.compat.v2 as tf
    from tensorflow import keras
    from tensorflow.keras import regularizers

    tf.enable_v2_behavior()
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors
    tfk = tfp.math.psd_kernels

    Feature_COLUMNS = ["HomeTeam", "AwayTeam"]
    Label_COLUMNS = ["FTHG", "FTAG"]
    CSV_COLUMNS = Feature_COLUMNS + Label_COLUMNS
    Derived_COLUMNS = ["t1goals", "t2goals", "t1goals_where", "t2goals_where"]
    COLS = ["HGFT", "AGFT", "HGHT", "AGHT", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
    Meta_COLUMNS = ["t1games", "t2games", "t1games_where", "t2games_where"]
    COLS_Extended = COLS + ['HWin', 'AWin', 'HLoss', 'ALoss', 'HDraw', 'ADraw']

    point_scheme_tcs = [[4, 6, 7], [3, 2, 5], [2, 2, 4], [2.62, 3.77, 4.93], [285 / 9 / 27, 247 / 9 / 26, 196 / 9 / 27]]
    point_scheme_pistor = [[3, 3, 3], [2, 2, 2], [1, 1, 1], [1.65, 2.46, 1.73], [285 / 9 / 27, 247 / 9 / 26, 196 / 9 / 27]]
    point_scheme_sky = [[5, 5, 5], [2, 2, 2], [2, 2, 2], [2.60, 3.38, 2.66], [285 / 9 / 27, 247 / 9 / 26, 196 / 9 / 27]]
    point_scheme_goal_diff = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [7, 9, 7.5],
                              [285 / 9 / 27, 247 / 9 / 26, 196 / 9 / 27]]

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

        full_data = pd.read_csv(data_dir + "/full_data.csv")
        is_train = full_data.Season.astype(str).apply(lambda x: x.zfill(4)).isin(train_seasons).repeat(2)
        is_test = full_data.Season.astype(str).apply(lambda x: x.zfill(4)).isin(test_seasons).repeat(2)
        is_test.value_counts()

        all_data = pd.read_csv(data_dir + "/all_features.csv")
        all_labels = pd.read_csv(data_dir + "/all_labels.csv")
        team_mapping = pd.read_csv(data_dir + "/team_mapping.csv")
        # all_features = pd.read_csv(data_dir+"/lfda_data.csv")
        feature_names = pd.read_csv(data_dir + "/feature_candidates_long.csv")
        print(feature_names)
        feature_names = feature_names.x.tolist()
        if 'BW1' in feature_names:
            # place 'BW1', 'BW2', 'BW0' first is list, so that they have a fixed colunm position
            feature_names = pd.Series(['BW1', 'BW2', 'BW0'] + feature_names).drop_duplicates().tolist()

        feature_names = ['Time', 't1games', 't1dayssince', 't2dayssince', 't1dayssince_ema', 't2dayssince_ema',
                         'roundsleft', 't1promoted', 't2promoted', 't1points', 't2points', 't1rank', 't2rank',
                         't1rank6_attention', 't2rank6_attention', 't1rank16_attention', 't2rank16_attention',
                         't1cards_ema', 't2cards_ema', 'BW1', 'BW0', 'BW2', 'T1_CUM_T1_GFT', 'T2_CUM_T2_GFT',
                         'T1_CUM_T1_W_GFT', 'T2_CUM_T2_W_GFT', 'T1_CUM_T2_GFT', 'T2_CUM_T1_GFT', 'T1_CUM_T2_W_GFT',
                         'T2_CUM_T1_W_GFT', 'T12_CUM_T1_GFT', 'T12_CUM_T1_W_GFT', 'T21_CUM_T2_GFT', 'T21_CUM_T2_W_GFT',
                         'T12_CUM_T12_GFT', 'T12_CUM_T12_W_GFT', 'T1221_CUM_GFT', 'T1221_CUM_W_GFT', 'T1_CUM_T1_GHT',
                         'T2_CUM_T2_GHT', 'T1_CUM_T1_W_GHT', 'T2_CUM_T2_W_GHT', 'T1_CUM_T2_GHT', 'T2_CUM_T1_GHT',
                         'T1_CUM_T2_W_GHT', 'T2_CUM_T1_W_GHT', 'T12_CUM_T1_GHT', 'T12_CUM_T1_W_GHT', 'T21_CUM_T2_GHT',
                         'T21_CUM_T2_W_GHT', 'T12_CUM_T12_GHT', 'T12_CUM_T12_W_GHT', 'T1221_CUM_GHT', 'T1221_CUM_W_GHT',
                         'T1_CUM_T1_S', 'T2_CUM_T2_S', 'T1_CUM_T1_W_S', 'T2_CUM_T2_W_S', 'T1_CUM_T2_S', 'T2_CUM_T1_S',
                         'T1_CUM_T2_W_S', 'T2_CUM_T1_W_S', 'T12_CUM_T1_S', 'T12_CUM_T1_W_S', 'T21_CUM_T2_S',
                         'T21_CUM_T2_W_S', 'T12_CUM_T12_S', 'T12_CUM_T12_W_S', 'T1221_CUM_S', 'T1221_CUM_W_S',
                         'T1_CUM_T1_ST', 'T2_CUM_T2_ST', 'T1_CUM_T1_W_ST', 'T2_CUM_T2_W_ST', 'T1_CUM_T2_ST', 'T2_CUM_T1_ST',
                         'T1_CUM_T2_W_ST', 'T2_CUM_T1_W_ST', 'T12_CUM_T1_ST', 'T12_CUM_T1_W_ST', 'T21_CUM_T2_ST',
                         'T21_CUM_T2_W_ST', 'T12_CUM_T12_ST', 'T12_CUM_T12_W_ST', 'T1221_CUM_ST', 'T1221_CUM_W_ST',
                         'T1_CUM_T1_F', 'T2_CUM_T2_F', 'T1_CUM_T1_W_F', 'T2_CUM_T2_W_F', 'T1_CUM_T2_F', 'T2_CUM_T1_F',
                         'T1_CUM_T2_W_F', 'T2_CUM_T1_W_F', 'T12_CUM_T1_F', 'T12_CUM_T1_W_F', 'T21_CUM_T2_F',
                         'T21_CUM_T2_W_F', 'T12_CUM_T12_F', 'T12_CUM_T12_W_F', 'T1221_CUM_F', 'T1221_CUM_W_F',
                         'T1_CUM_T1_C', 'T2_CUM_T2_C', 'T1_CUM_T1_W_C', 'T2_CUM_T2_W_C', 'T1_CUM_T2_C', 'T2_CUM_T1_C',
                         'T1_CUM_T2_W_C', 'T2_CUM_T1_W_C', 'T12_CUM_T1_C', 'T12_CUM_T1_W_C', 'T21_CUM_T2_C',
                         'T21_CUM_T2_W_C', 'T12_CUM_T12_C', 'T12_CUM_T12_W_C', 'T1221_CUM_C', 'T1221_CUM_W_C',
                         'T1_CUM_T1_Y', 'T2_CUM_T2_Y', 'T1_CUM_T1_W_Y', 'T2_CUM_T2_W_Y', 'T1_CUM_T2_Y', 'T2_CUM_T1_Y',
                         'T1_CUM_T2_W_Y', 'T2_CUM_T1_W_Y', 'T12_CUM_T1_Y', 'T12_CUM_T1_W_Y', 'T21_CUM_T2_Y',
                         'T21_CUM_T2_W_Y', 'T12_CUM_T12_Y', 'T12_CUM_T12_W_Y', 'T1221_CUM_Y', 'T1221_CUM_W_Y',
                         'T1_CUM_T1_R', 'T2_CUM_T2_R', 'T1_CUM_T1_W_R', 'T2_CUM_T2_W_R', 'T1_CUM_T2_R', 'T2_CUM_T1_R',
                         'T1_CUM_T2_W_R', 'T2_CUM_T1_W_R', 'T12_CUM_T1_R', 'T12_CUM_T1_W_R', 'T21_CUM_T2_R',
                         'T21_CUM_T2_W_R', 'T12_CUM_T12_R', 'T12_CUM_T12_W_R', 'T1221_CUM_R', 'T1221_CUM_W_R',
                         'T1_CUM_T1_xG', 'T2_CUM_T2_xG', 'T1_CUM_T1_W_xG', 'T2_CUM_T2_W_xG', 'T1_CUM_T2_xG', 'T2_CUM_T1_xG',
                         'T1_CUM_T2_W_xG', 'T2_CUM_T1_W_xG', 'T12_CUM_T1_xG', 'T12_CUM_T1_W_xG', 'T21_CUM_T2_xG',
                         'T21_CUM_T2_W_xG', 'T12_CUM_T12_xG', 'T12_CUM_T12_W_xG', 'T1221_CUM_xG', 'T1221_CUM_W_xG',
                         'T1_CUM_T1_GH2', 'T2_CUM_T2_GH2', 'T1_CUM_T1_W_GH2', 'T2_CUM_T2_W_GH2', 'T1_CUM_T2_GH2',
                         'T2_CUM_T1_GH2', 'T1_CUM_T2_W_GH2', 'T2_CUM_T1_W_GH2', 'T12_CUM_T1_GH2', 'T12_CUM_T1_W_GH2',
                         'T21_CUM_T2_GH2', 'T21_CUM_T2_W_GH2', 'T12_CUM_T12_GH2', 'T12_CUM_T12_W_GH2', 'T1221_CUM_GH2',
                         'T1221_CUM_W_GH2', 'T1_CUM_T1_Win', 'T2_CUM_T2_Win', 'T1_CUM_T1_W_Win', 'T2_CUM_T2_W_Win',
                         'T1_CUM_T1_HTWin', 'T2_CUM_T2_HTWin', 'T1_CUM_T1_W_HTWin', 'T2_CUM_T2_W_HTWin', 'T1_CUM_T1_Loss',
                         'T2_CUM_T2_Loss', 'T1_CUM_T1_W_Loss', 'T2_CUM_T2_W_Loss', 'T1_CUM_T1_HTLoss', 'T2_CUM_T2_HTLoss',
                         'T1_CUM_T1_W_HTLoss', 'T2_CUM_T2_W_HTLoss', 'T1_CUM_T1_Draw', 'T2_CUM_T2_Draw', 'T1_CUM_T1_W_Draw',
                         'T2_CUM_T2_W_Draw', 'T1_CUM_T1_HTDraw', 'T2_CUM_T2_HTDraw', 'T1_CUM_T1_W_HTDraw',
                         'T2_CUM_T2_W_HTDraw']
        feature_names += ["T1_spi", "T2_spi", "T1_imp", "T2_imp", "T1_GFTe", "T2_GFTe", "pp1", "pp0", "pp2"]
        feature_names += ["T1_CUM_T1_GFTa", "T2_CUM_T2_GFTa", "T1_CUM_T1_W_GFTa", "T2_CUM_T2_W_GFTa", "T1_CUM_T2_GFTa",
                          "T2_CUM_T1_GFTa", "T1_CUM_T2_W_GFTa", "T2_CUM_T1_W_GFTa", "T12_CUM_T1_GFTa", "T12_CUM_T1_W_GFTa",
                          "T21_CUM_T2_GFTa", "T21_CUM_T2_W_GFTa", "T12_CUM_T12_GFTa", "T12_CUM_T12_W_GFTa",
                          "T1221_CUM_GFTa", "T1221_CUM_W_GFTa", "T1_CUM_T1_xsg", "T2_CUM_T2_xsg", "T1_CUM_T1_W_xsg",
                          "T2_CUM_T2_W_xsg", "T1_CUM_T2_xsg", "T2_CUM_T1_xsg", "T1_CUM_T2_W_xsg", "T2_CUM_T1_W_xsg",
                          "T12_CUM_T1_xsg", "T12_CUM_T1_W_xsg", "T21_CUM_T2_xsg", "T21_CUM_T2_W_xsg", "T12_CUM_T12_xsg",
                          "T12_CUM_T12_W_xsg", "T1221_CUM_xsg", "T1221_CUM_W_xsg", "T1_CUM_T1_xnsg", "T2_CUM_T2_xnsg",
                          "T1_CUM_T1_W_xnsg", "T2_CUM_T2_W_xnsg", "T1_CUM_T2_xnsg", "T2_CUM_T1_xnsg", "T1_CUM_T2_W_xnsg",
                          "T2_CUM_T1_W_xnsg", "T12_CUM_T1_xnsg", "T12_CUM_T1_W_xnsg", "T21_CUM_T2_xnsg",
                          "T21_CUM_T2_W_xnsg", "T12_CUM_T12_xnsg", "T12_CUM_T12_W_xnsg", "T1221_CUM_xnsg",
                          "T1221_CUM_W_xnsg", "T1_CUM_T1_spi", "T2_CUM_T2_spi", "T1_CUM_T1_W_spi", "T2_CUM_T2_W_spi",
                          "T1_CUM_T2_spi", "T2_CUM_T1_spi", "T1_CUM_T2_W_spi", "T2_CUM_T1_W_spi", "T12_CUM_T1_spi",
                          "T12_CUM_T1_W_spi", "T21_CUM_T2_spi", "T21_CUM_T2_W_spi", "T12_CUM_T12_spi", "T12_CUM_T12_W_spi",
                          "T1221_CUM_spi", "T1221_CUM_W_spi", "T1_CUM_T1_imp", "T2_CUM_T2_imp", "T1_CUM_T1_W_imp",
                          "T2_CUM_T2_W_imp", "T1_CUM_T2_imp", "T2_CUM_T1_imp", "T1_CUM_T2_W_imp", "T2_CUM_T1_W_imp",
                          "T12_CUM_T1_imp", "T12_CUM_T1_W_imp", "T21_CUM_T2_imp", "T21_CUM_T2_W_imp", "T12_CUM_T12_imp",
                          "T12_CUM_T12_W_imp", "T1221_CUM_imp", "T1221_CUM_W_imp", "T1_CUM_T1_GFTe", "T2_CUM_T2_GFTe",
                          "T1_CUM_T1_W_GFTe", "T2_CUM_T2_W_GFTe", "T1_CUM_T2_GFTe", "T2_CUM_T1_GFTe", "T1_CUM_T2_W_GFTe",
                          "T2_CUM_T1_W_GFTe", "T12_CUM_T1_GFTe", "T12_CUM_T1_W_GFTe", "T21_CUM_T2_GFTe",
                          "T21_CUM_T2_W_GFTe", "T12_CUM_T12_GFTe", "T12_CUM_T12_W_GFTe", "T1221_CUM_GFTe",
                          "T1221_CUM_W_GFTe"]
        feature_names += ['t1goals_ema', 't2goals_ema', 'T1_EMA_GFT', 'T2_EMA_GFT', 'T1_EMA_GHT', 'T2_EMA_GHT', 'T1_EMA_ST', 'T2_EMA_ST',
                         'T1_EMA_S', 'T2_EMA_S', 'T1_EMA_C', 'T2_EMA_C', 'T1_EMA_F', 'T2_EMA_F', 'T1_EMA_Y', 'T2_EMA_Y', 'T1_EMA_R', 'T2_EMA_R',
                         'T1_EMA_xG', 'T2_EMA_xG', 'T1_EMA_GFTa', 'T2_EMA_GFTa', 'T1_EMA_xsg', 'T2_EMA_xsg', 'T1_EMA_xnsg', 'T2_EMA_xnsg',
                         'T1_EMA_spi', 'T2_EMA_spi', 'T1_EMA_imp', 'T2_EMA_imp', 'T1_EMA_GFTe', 'T2_EMA_GFTe', 'T1_EMA_GH2', 'T2_EMA_GH2']



        # last 10 rounds across all teams - general distribution
        feature_names += ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
                          'L10_T1_GFT', 'L10_T2_GFT', 'L10_T1_GHT', 'L10_T2_GHT', 'L10_T1_S', 'L10_T2_S', 'L10_T1_ST', 'L10_T2_ST',
                          'L10_T1_F', 'L10_T2_F', 'L10_T1_C', 'L10_T2_C', 'L10_T1_Y', 'L10_T2_Y', 'L10_T1_R', 'L10_T2_R',
                          'L10_T1_xG', 'L10_T2_xG', 'L10_T1_GFTa', 'L10_T2_GFTa', 'L10_T1_xsg', 'L10_T2_xsg', 'L10_T1_xnsg', 'L10_T2_xnsg',
                          'L10_T1_GH2', 'L10_T2_GH2', 'L10_Loss', 'L10_Draw', 'L10_Win', 'L10_HTLoss', 'L10_HTDraw', 'L10_HTWin',
                          'L10_HT2Loss', 'L10_HT2Draw', 'L10_HT2Win',
                          'L10_zScore0:3', 'L10_zScore1:3', 'L10_zScore0:2', 'L10_zScore1:2', 'L10_zScore0:1', 'L10_zScore0:0', 'L10_zScore1:1',
                          'L10_zScore1:0', 'L10_zScore2:1', 'L10_zScore2:0', 'L10_zScore3:1', 'L10_zScore3:0', 'L10_zScore2:2', 'L10_zScore3:2',
                          'L10_zScore2:3', 'L10_zScore3:3', 'L10_FTG4', 'L10_FTG0', 'L10_HTG0']



        all_features = all_data[feature_names].copy()
        all_data["Train"] = is_train.values & (~all_data["Predict"])
        all_data["Test"] = is_test.values & (~all_data["Predict"])

        if not useBWIN and 'BW1' in feature_names:
            all_data["BW1"] = 0.0
            all_data["BW0"] = 0.0
            all_data["BW2"] = 0.0
            all_features.loc[:, "BW1"] = 0.0
            all_features.loc[:, "BW0"] = 0.0
            all_features.loc[:, "BW2"] = 0.0
            print("BWIN features set to zero")

        #  all_labels["Train"]=is_train.values
        #  all_labels["Test"]=is_test.values
        #  all_labels["Predict"] = all_data["Predict"]
        #
        #  all_features["Train"]=is_train.values
        #  all_features["Test"]=is_test.values
        #  all_features["Predict"] = all_data["Predict"]
        teamnames = team_mapping.Teamname.tolist()
        # print(all_data.iloc[1000])
        return all_data, all_labels, all_features, teamnames, team_mapping


    def build_features(all_data, all_labels, all_features, teamnames, team_mapping):
        all_data["gameindex"] = all_data.index

        mh1 = [all_data.groupby("Team1_index").gameindex.shift(i) for i in range(SEQ_LENGTH, 0, -1)]
        mh1 = np.stack(mh1, axis=1)
        all_data["mh1len"] = np.sum(~pd.isna(mh1), axis=1)
        mh1[np.isnan(mh1)] = -1
        mh1 = mh1.astype(np.int16)

        mh2 = [all_data.groupby("Team2_index").gameindex.shift(i) for i in range(SEQ_LENGTH, 0, -1)]
        mh2 = np.stack(mh2, axis=1)
        all_data["mh2len"] = np.sum(~pd.isna(mh2), axis=1)
        mh2[np.isnan(mh2)] = -1
        mh2 = mh2.astype(np.int16)

        mh12 = [all_data.groupby(["Team1_index", "Team2_index"]).gameindex.shift(i) for i in range(SEQ_LENGTH, 0, -1)]
        mh12 = np.stack(mh12, axis=1)
        all_data["mh12len"] = np.sum(~pd.isna(mh12), axis=1)
        mh12[np.isnan(mh12)] = -1
        mh12 = mh12.astype(np.int16)

        label_column_names = all_labels.columns
        #  feature_column_names = all_features.columns

        prefix0 = all_data[["mh1len", "Where", "mh2len", "mh12len"]].astype(np.float32)
        prefix = prefix0.values
        prefix[:, [0, 2, 3]] *= 0.1
        match_input_layer = np.concatenate([prefix, all_features.values], axis=1)
        print("prefix.columns.tolist()+all_features.columns.tolist()")
        print(prefix0.columns.tolist() + all_features.columns.tolist())

        print("match_input_layer [:,23:26]")
        print(match_input_layer[:, 23:26])
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
            # print("Normalized confusion matrix")
        # else:
        # print('Confusion matrix, without normalization')

        # print(cm)

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
        if prefix == "ens/":
            return

        print("-----------------------------------------------------")
        print("{} / {}:{}".format(title, gs, gc))
        default_color = np.array([[0.12, 0.47, 0.71, 0.5]])
        if prefix == "p1/" or prefix == "p3/":
            default_color = "darkolivegreen"
        if prefix == "p2/" or prefix == "p4/":
            default_color = "darkmagenta"

        sp = pred[prefix + "p_pred_12"]
        # print(sp)
        spt = pred[prefix + "ev_points"]
        gs = min(gs, 6)
        gc = min(gc, 6)
        margin_pred_prob1 = pred[prefix + "p_marg_1"]
        margin_poisson_prob1 = pred[prefix + "p_poisson_1"]
        margin_pred_prob2 = pred[prefix + "p_marg_2"]
        margin_poisson_prob2 = pred[prefix + "p_poisson_2"]
        margin_pred_expected1 = pred[prefix + "ev_goals_1"]
        margin_pred_expected2 = pred[prefix + "ev_goals_2"]
        g = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        g1 = [0.0] * 7 + [1.0] * 7 + [2.0] * 7 + [3.0] * 7 + [4.0] * 7 + [5.0] * 7 + [6.0] * 7
        g2 = g * 7
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(g1, g2, s=sp * 10000, alpha=0.4, color=default_color)
        ax[0].scatter(gs, gc, s=sp[gs * 7 + gc] * 10000, alpha=0.7, color=default_color)
        for i, txt in enumerate(sp):
            ax[0].annotate("{:4.2f}".format(txt * 100), (g1[i] - 0.3, g2[i] - 0.1))
        ax[1].scatter(g1, g2, s=spt * 500, alpha=0.4, color='red')
        ax[1].scatter(gs, gc, s=spt[gs * 7 + gc] * 500, alpha=0.7, color='red')
        for i, txt in enumerate(spt):
            ax[1].annotate("{:4.2f}".format(txt), (g1[i] - 0.3, g2[i] - 0.1))
        ax[0].set_title(prefix)
        ax[1].set_title(prefix)
        max_sp = max(sp)
        max_sp_index = np.argmax(sp)
        ax[0].scatter((max_sp_index // 7).astype(float), np.mod(max_sp_index, 7).astype(float), s=max_sp * 10000.0,
                      facecolors='none', edgecolors='black', linewidth=2)
        max_spt = max(spt)
        max_spt_index = np.argmax(spt)
        ax[1].scatter((max_spt_index // 7).astype(float), np.mod(max_spt_index, 7).astype(float), s=max_spt * 500.0,
                      facecolors='none', edgecolors='black', linewidth=2)

        p_loss = 0.0
        p_win = 0.0
        p_draw = 0.0
        for i in range(7):
            for j in range(7):
                if i > j:
                    p_win += sp[i * 7 + j]
                if i < j:
                    p_loss += sp[i * 7 + j]
                if i == j:
                    p_draw += sp[i * 7 + j]
        ax[2].axis('equal')
        explode = [0, 0, 0]
        explode[1 - np.sign(gs - gc)] = 0.1
        wedges, _, _ = ax[2].pie([p_win, p_draw, p_loss], labels=["Win", "Draw", "Loss"], colors=["blue", "green", "red"],
                                 startangle=90, autopct='%1.1f%%',
                                 radius=1.0, explode=explode, wedgeprops={"alpha": 0.5})
        wedges[1 - np.sign(gs - gc)].set_alpha(0.8)
        plt.show()

        w = 0.35
        fig, ax = plt.subplots(1, 3, figsize=(15, 1))
        ax[0].bar(g, margin_pred_prob1, alpha=0.6, width=w, color=default_color)
        ax[0].bar([x + w for x in g], margin_poisson_prob1, alpha=0.3, color="red", width=0.35)
        ax[0].bar(gs, margin_pred_prob1[gs], alpha=0.5, width=w, color=default_color)
        ax[0].bar(gs + w, margin_poisson_prob1[gs], alpha=0.7, color="red", width=0.35)
        ax[0].axvline(x=margin_pred_expected1, color='red')
        ax[1].bar(g, margin_pred_prob2, alpha=0.6, width=w, color=default_color)
        ax[1].bar([x + w for x in g], margin_poisson_prob2, alpha=0.3, color="red", width=0.35)
        ax[1].bar(gc, margin_pred_prob2[gc], alpha=0.5, width=w, color=default_color)
        ax[1].bar(gc + w, margin_poisson_prob2[gc], alpha=0.7, color="red", width=0.35)
        ax[1].axvline(x=margin_pred_expected2, color='red')
        ax[0].set_title(margin_pred_expected1)
        ax[1].set_title(margin_pred_expected2)
        bars = ax[2].bar([0, 1, 2], height=[p_win, p_draw, p_loss], tick_label=["Win", "Draw", "Loss"],
                         color=["blue", "green", "red"], alpha=0.5)
        bars[1 - np.sign(gs - gc)].set_alpha(0.8)
        bars[1 - np.sign(gs - gc)].set_linewidth(2)
        bars[1 - np.sign(gs - gc)].set_edgecolor("black")
        for i in [-1, 0, 1]:
            if i != np.sign(gs - gc):
                bars[1 - i].set_hatch("x")
        plt.show()


    def print_match_dates(X, team_onehot_encoder):
        features = X["newgame"]
        tn = len(team_onehot_encoder.classes_)
        match_dates = features[:, 4 + 2 * tn + 0] * 1000.0
        match_dates = [datetime.strftime(datetime.fromordinal(int(m) + 734138), "%Y/%m/%d") for m in match_dates]
        match_dates = list(sorted(set(match_dates)))
        print(len(features))
        print(match_dates)
        return match_dates

    #plot_predictions_3(df2, "poisson", "Train", silent=False)
    def plot_predictions_3(df, prefix, dataset, silent=False):
        df = df.loc[(df.Prefix == prefix) & (df.dataset == dataset)].copy()
        goal_cnt = Counter([str(gs) + ":" + str(gc) if w == "Home" else str(gc) + ":" + str(gs) for gs, gc, w in
                            zip(df["GS"], df["GC"], df["Where"])])
        gdiff_cnt = Counter([gs - gc if w == "Home" else gc - gs for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])])
        tend_cnt = Counter(
            [np.sign(gs - gc) if w == "Home" else np.sign(gc - gs) for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])])

        # goodness of fit:
        df["gof"] = ["3_full" if p1 == gs and p2 == gc else
                     "2_diff" if p1 - p2 == gs - gc else
                     "1_tendency" if np.sign(p1 - p2) == np.sign(gs - gc) else
                     "0_none"
                     for p1, p2, gs, gc in zip(df["pGS"], df["pGC"], df["GS"], df["GC"])]
        df["pFTHG"] = [p1 if w == "Home" else p2 for p1, p2, w in zip(df["pGS"], df["pGC"], df["Where"])]
        df["pFTAG"] = [p2 if w == "Home" else p1 for p1, p2, w in zip(df["pGS"], df["pGC"], df["Where"])]
        df["pGoals"] = [str(g1) + ":" + str(g2) for g1, g2 in zip(df["pFTHG"], df["pFTAG"])]
        df["pGDiff"] = df["pFTHG"] - df["pFTAG"]
        df["pTendency"] = np.sign(df["pFTHG"] - df["pFTAG"])
        df["total_points"] = 0

        if point_scheme[0][0] == -1:
            # GoalDiff calculation
            df["total_points"] = (np.sign(df["pGS"] - df["pGC"]) == np.sign(df["GS"] - df["GC"])) * (
                    10.0 - np.minimum(5.0, np.abs(df["pGS"] - df["GS"]) + np.abs(df["pGC"] - df["GC"])))
        else:
            df["total_points"] += [point_scheme[0][1] if t == 0 and gof == "3_full" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[1][1] if t == 0 and gof == "2_diff" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[2][1] if t == 0 and gof == "1_tendency" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[2][0] if t == 1 and gof == "1_tendency" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[1][0] if t == 1 and gof == "2_diff" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[0][0] if t == 1 and gof == "3_full" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[2][2] if t == -1 and gof == "1_tendency" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[1][2] if t == -1 and gof == "2_diff" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]
            df["total_points"] += [point_scheme[0][2] if t == -1 and gof == "3_full" else 0 for t, gof in
                                   zip(np.sign(df["pFTHG"] - df["pFTAG"]), df["gof"])]

        df = pd.concat([df, pd.get_dummies(df["gof"])], axis=1)  # one-hot encoding of gof
        if "3_full" not in df:
            df["3_full"] = 0
        if "2_diff" not in df:
            df["2_diff"] = 0
        if "1_tendency" not in df:
            df["1_tendency"] = 0
        if "0_none" not in df:
            df["0_none"] = 0

        df["hit_goal"] = 100 * df["3_full"]
        df["hit_diff"] = 100 * (df["3_full"] + df["2_diff"])
        df["hit_tend"] = 100 * (df["3_full"] + df["2_diff"] + df["1_tendency"])

        df["sort_idx"] = [(x[0] - x[1]) * (1 + 0.1 * x[0]) + 0.01 * x[0] for x in zip(df["pFTHG"], df["pFTAG"])]
        df = df.sort_values(by=('sort_idx'), ascending=True)

        def get_freq(cnt, lookup):
            return [cnt[x] * 100.0 / len(lookup) if x in cnt else 0 for x in lookup]

        df["goal_freq"] = get_freq(goal_cnt, df["pGoals"])
        df["gdiff_freq"] = get_freq(gdiff_cnt, df["pGDiff"])
        df["tend_freq"] = get_freq(tend_cnt, df["pTendency"])

        # t5 = df.pivot_table(index='Team1',
        #                     aggfunc={'0_none': np.sum, '1_tendency': np.sum, '2_diff': np.sum, '3_full': np.sum,
        #                              'total_points': [np.sum, np.mean],
        #                              "hit_tend": np.mean, "hit_diff": np.mean, "hit_goal": np.mean},
        #                     margins=False, fill_value=0)
        # t5.columns = ["None", "Tendency", "Diff", "Full", "Diff%", "Goal%", "Tendency%", "AvgPoints", "TotalPoints"]
        # t5 = t5.sort_values(by=('AvgPoints'), ascending=False)
        # print(t5)
        print(prefix)
        t2 = df.pivot_table(index=[df["sort_idx"], "pGoals"],
                            aggfunc={'0_none': np.sum, '1_tendency': np.sum, '2_diff': np.sum, '3_full': np.sum,
                                     'total_points': [len, np.sum, np.mean],
                                     "hit_goal": np.mean, "goal_freq": np.mean},
                            margins=False, fill_value=0)

        def append_metrics(pred, act, scores):
            df2 = pd.DataFrame({"act": act, "pred": [p for p in
                                                     pred]})  # convert pred to list in order to prevent unwanted index reordering
            df2["TP"] = df2.pred & df2.act
            df2["TN"] = ~df2.pred & ~df2.act
            df2["FP"] = df2.pred & ~df2.act
            df2["FN"] = ~df2.pred & df2.act
            df2 = df2.astype(int)
            df3 = df2.sum()
            df3["MCC"] = (df3.TP * df3.TN - df3.FP * df3.FN) / np.sqrt(
                (df3.TP + df3.FP) * (df3.TP + df3.FN) * (df3.TN + df3.FP) * (df3.TN + df3.FN))
            df3["F1"] = 2 * df3.TP / (2 * df3.TP + df3.FP + df3.FN)
            df3["ACC"] = (df3.TP + df3.TN) / (df3.TP + df3.TN + df3.FP + df3.FN)
            df3["Recall"] = df3.TP / (df3.TP + df3.FN)
            df3["Precision"] = df3.TP / (df3.TP + df3.FP)
            df3["name"] = r
            df3 = df3.T
            scores = scores.append(df3, ignore_index=True)
            return scores

        scores = pd.DataFrame()
        for r in t2.reset_index().pGoals:
            pred = r == df["pGoals"]
            act = [x if w == "Home" else x[::-1] for x, w in zip(df["act"], df["Where"])]
            act = pd.Series([r == x for x in act])
            ##### scores = append_metrics(pred, act, scores) #### may lead to division by zero
        # print(scores)

        t2.reset_index(inplace=True)
        t2.columns = ['_'.join(col).strip() for col in t2.columns.values]
        t2 = t2.drop(['sort_idx_'], axis=1)
        t2 = t2.rename(columns={"pGoals_": "Goals",
                                "0_none_sum": "None",
                                "1_tendency_sum": "Tendency",
                                "1_tendency_sum": "Tendency",
                                "2_diff_sum": "Diff",
                                "3_full_sum": "Full",
                                "total_points_len": "Total",
                                "hit_goal_mean": "ActualRate",
                                "goal_freq_mean": "TargetRate",
                                "total_points_mean": "AvgPoints",
                                "total_points_sum": "TotalPoints",
                                })
        t2 = t2.assign(EffRate=t2.ActualRate - t2.TargetRate, Contribution=t2.TotalPoints / len(df))
        t2 = t2[['Goals', 'None', 'Tendency', 'Diff', 'Full', 'Total',
                 'ActualRate', 'TargetRate', 'EffRate',
                 'AvgPoints', 'TotalPoints', 'Contribution']]

        print()
        print(t2)

        avg_points = np.sum(t2["Contribution"])

        pie_chart_values = t2[["None", "Tendency", "Diff", "Full"]].sum()

        t3 = df.pivot_table(index=["pGDiff"],
                            aggfunc={'0_none': np.sum, '1_tendency': np.sum, '2_diff': np.sum, '3_full': np.sum,
                                     'total_points': [len, np.sum, np.mean],
                                     "hit_diff": np.mean, "gdiff_freq": np.mean},
                            margins=False, fill_value=0)

        t3.reset_index(inplace=True)
        t3.columns = ['_'.join(col).strip() for col in t3.columns.values]
        t3 = t3.rename(columns={"pGDiff_": "GDiff",
                                "0_none_sum": "None",
                                "1_tendency_sum": "Tendency",
                                "1_tendency_sum": "Tendency",
                                "2_diff_sum": "Diff",
                                "3_full_sum": "Full",
                                "total_points_len": "Total",
                                "hit_diff_mean": "ActualRate",
                                "gdiff_freq_mean": "TargetRate",
                                "total_points_mean": "AvgPoints",
                                "total_points_sum": "TotalPoints",
                                })
        t3 = t3.assign(EffRate=t3.ActualRate - t3.TargetRate, Contribution=t3.TotalPoints / len(df))
        t3 = t3[['GDiff', 'None', 'Tendency', 'Diff', 'Full', 'Total',
                 'ActualRate', 'TargetRate', 'EffRate',
                 'AvgPoints', 'TotalPoints', 'Contribution']]
        if not silent:
            print()
            print(t3)

        for r in t3.reset_index().GDiff:
            pred = df["pGDiff"] == r
            act = [gs - gc if w == "Home" else gc - gs for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])]
            act = pd.Series([r == x for x in act])
            scores = append_metrics(pred, act, scores)
        # print(scores)

        tend_cnt = Counter(df["pTendency"])
        tendency_values = [tend_cnt.get(-1), tend_cnt.get(0), tend_cnt.get(1)]
        tendency_values = [0 if v is None else v for v in tendency_values]

        print()

        t4 = df.pivot_table(index=["pTendency"],
                            aggfunc={'0_none': np.sum, '1_tendency': np.sum, '2_diff': np.sum, '3_full': np.sum,
                                     'total_points': [len, np.sum, np.mean],
                                     "hit_tend": np.mean, "tend_freq": np.mean},
                            margins=False, fill_value=0)

        t4.reset_index(inplace=True)
        t4.columns = ['_'.join(col).strip() for col in t4.columns.values]
        t4 = t4.rename(columns={"pTendency_": "Prediction",
                                "0_none_sum": "None",
                                "1_tendency_sum": "Tendency",
                                "1_tendency_sum": "Tendency",
                                "2_diff_sum": "Diff",
                                "3_full_sum": "Full",
                                "total_points_len": "Total",
                                "hit_tend_mean": "ActualRate",
                                "tend_freq_mean": "TargetRate",
                                "total_points_mean": "AvgPoints",
                                "total_points_sum": "TotalPoints",
                                })
        t4 = t4.assign(EffRate=t4.ActualRate - t4.TargetRate, Contribution=t4.TotalPoints / len(df))
        t4 = t4[['Prediction', 'None', 'Tendency', 'Diff', 'Full', 'Total',
                 'ActualRate', 'TargetRate', 'EffRate',
                 'AvgPoints', 'TotalPoints', 'Contribution']]
        t4["Prediction"] = ["Draw" if p == 0 else "Homewin" if p == 1 else "Awaywin" for p in t4["Prediction"]]
        if not silent:
            print(t4)

        for r in t4.reset_index().Prediction:
            pred = ["Draw" if p == 0 else "Homewin" if p > 0 else "Awaywin" for p in df["pGDiff"]]
            pred = pd.Series([p == r for p in pred])
            act = [gs - gc if w == "Home" else gc - gs for gs, gc, w in zip(df["GS"], df["GC"], df["Where"])]
            act = ["Draw" if p == 0 else "Homewin" if p > 0 else "Awaywin" for p in act]
            act = pd.Series([r == x for x in act])
            scores = append_metrics(pred, act, scores)

        scores = pd.concat([scores.name,
                            scores[['act', 'pred', 'TN', 'FN', 'FP', 'TP']].astype(int),
                            scores[['Precision', 'Recall', 'F1', 'ACC', 'MCC']]], axis=1)
        if not silent:
            print()
            print(scores)

        t1 = df.pivot_table(index=[df["sort_idx"], 'pGoals'], columns=['gof'], values=["Team1"], aggfunc=len, margins=False,
                            fill_value=0)
        t1 = df.pivot_table(index=[df["sort_idx"], 'pGoals'], columns=['total_points'], values=["Team1"], aggfunc=len, margins=False,
                            fill_value=0, )
        bar_label = df.groupby(["sort_idx", 'pGoals']).agg({"total_points":["mean", "sum"], "Where":"count"})
        #t1.columns = ["None", "Tendency", "Diff", "Full"][:len(t1.columns)]
        t1.index = t1.index.droplevel(level=0)
        t1 = t1.droplevel(level=0, axis=1)
        t1.columns = t1.columns.astype(int)
        print(bar_label)
        print(bar_label.loc[:,("Where", "count")].sum())
        print(bar_label.loc[:,("Where", "count")].max())
        offset = bar_label.loc[:,("Where", "count")].max()

        if not silent:
            fig = plt.figure(figsize=(16, 4.5))
            ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=2, rowspan=1)
            ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=1, rowspan=1)
            ax2.axis('off')
            ax3 = plt.subplot2grid((1, 4), (0, 3), colspan=1, rowspan=1)
            ax3.axis('off')

        if not silent:
            barcolors = [plt.cm.RdYlBu_r((i + 0.5) / (len(t1.columns))) for i in range(len(t1.columns))]
            t1.plot(kind='bar', stacked=True, ax=ax1, color=barcolors, linewidth=1, edgecolor = 'darkblue') # cmap=plt.cm.RdYlBu_r
            plt.subplots_adjust(bottom=0.15, top=0.94, left=0.04, right=0.97)
            ax1.set_title("{}".format(prefix)).set_size(15)
            ax1.set_xlabel(None)
            for idx in range(len(bar_label.Where)):
                ax1.text(idx, bar_label.Where.iloc[idx], "{:.3f}".format(bar_label.loc[:,("total_points", "mean")].iloc[idx]), ha='center', va='bottom', rotation=0)
                ax1.text(idx, -0.12*offset, "{:.3f}".format(bar_label.loc[:,("total_points", "sum")].iloc[idx]/bar_label.loc[:,("Where", "count")].sum()),
                         ha='center', va='top', rotation=0, color="red")
            _, _, autotexts = ax2.pie(np.sum(t1, axis=0), colors=barcolors,  wedgeprops = {'linewidth': 1, 'edgecolor' : 'darkblue'},
                                      autopct='%1.1f%%', startangle=90)
            #autotexts[0].set_color("white")
            #autotexts[1].set_color("white")
            #autotexts[-1].set_color("white")
            autotexts[1].set_fontsize("large")
            autotexts[1].set_position([z*0.75 for z in autotexts[1].get_position()])
            autotexts[1].set_verticalalignment("center")
            autotexts[2].set_verticalalignment("top")
            ax2.set_title("{}: {:.04f} ({})".format(prefix, avg_points, dataset)).set_size(20)

            # percentages = [pie_chart_values[0],  # None
            #                pie_chart_values[1] + pie_chart_values[2] + pie_chart_values[3],  # Tendency
            #                pie_chart_values[2] + pie_chart_values[3],  # GDiff
            #                pie_chart_values[3],  # Full
            #                ]
            c = len(t1.columns)
            percentages = [np.sum(t1.values[:,0])]+[np.sum(t1.values[:,i:]) for i in range(1, c)]
            for t, p in zip(autotexts, percentages):
                t.set_text("{:.01f}%".format(100.0 * p / len(df)))

        y_pred = [np.sign(p1 - p2) if w == "Home" else np.sign(p2 - p1) for p1, p2, w in
                  zip(df["pGS"], df["pGC"], df["Where"])]
        y_pred = ["Draw" if i == 0 else "HomeWin" if i == 1 else "AwayWin" for i in y_pred]
        y_test = [np.sign(p1 - p2) if w == "Home" else np.sign(p2 - p1) for p1, p2, w in
                  zip(df["GS"], df["GC"], df["Where"])]
        y_test = ["Draw" if i == 0 else "HomeWin" if i == 1 else "AwayWin" for i in y_test]

        cnf_matrix = confusion_matrix(y_test, y_pred)

        if not silent:
            pAtA, pDtA, pHtA, pAtD, pDtD, pHtD, pAtH, pDtH, pHtH = cnf_matrix.reshape([9])
            ax3.axis('equal')
            wedges, _, autotexts = ax3.pie([pHtA, pHtH, pHtD, pDtH, pDtD, pDtA, pAtD, pAtA, pAtH],
                                           labels=["", "Home", "", "", "Draw", "", "", "Away", ""],
                                           # colors=["blue", "blue", "blue", "green", "green", "green", "red", "red", "red"],
                                           colors=["white"] * 9,
                                           startangle=90, autopct='%1.1f%%',
                                           radius=1.0, pctdistance=0.75,
                                           wedgeprops={"alpha": 1.0, "linewidth": 3})

            true_colors = ["red", "blue", "green", "blue", "green", "red", "green", "red", "blue"]
            pred_colors = ["blue", "blue", "blue", "green", "green", "green", "red", "red", "red"]

            for t, c in zip(autotexts, true_colors):
                t.set_color(c)

            for w, c in zip(wedges, pred_colors):
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

            # wedges[1].set_alpha(0.6)
            # wedges[4].set_alpha(0.6)
            # wedges[7].set_alpha(0.6)

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
                100.0 * tendency_values[2] / len(df),
                100.0 * tendency_values[1] / len(df),
                100.0 * tendency_values[0] / len(df)
            ))

            plt.show()
            #plt.close()

        print()
        print(
            "Points: {0:.4f}, Tendency: {1:.2f}, Diff: {2:.2f}, Full: {3:.2f},    Home: {4:.1f}, Draw: {5:.1f}, Away: {6:.1f}".format(
                avg_points,
                100.0 * (1 - pie_chart_values[0] / len(df)),
                100.0 * (pie_chart_values[2] + pie_chart_values[3]) / len(df),
                100.0 * pie_chart_values[3] / len(df),
                100.0 * tendency_values[2] / len(df),
                100.0 * tendency_values[1] / len(df),
                100.0 * tendency_values[0] / len(df)
            ))
        c_home = df["Where"] == "Home"
        c_win = df['pGS'] > df['pGC']
        c_loss = df['pGS'] < df['pGC']
        c_draw = df['pGS'] == df['pGC']
        c_tendency = np.sign(df['pGS'] - df['pGC']) == np.sign(df["GS"] - df["GC"])

        if not silent:
            default_color = "blue"
            default_cmap = plt.cm.Blues
            if prefix == "cp" or prefix == "cp2":
                default_color = "darkolivegreen"
                default_cmap = plt.cm.Greens
            if prefix == "pg2" or prefix == "pgpt":
                default_color = "darkmagenta"
                default_cmap = plt.cm.Purples  # RdPu # PuRd

            def createTitle(series1, series2):
                return "pearson: {:.4f}, spearman: {:.4f}".format(
                    series1.corr(series2, method="pearson"),
                    series1.corr(series2, method="spearman")
                )

            if prefix != "ens":
                df["offset"] = np.random.rand(len(df))
                df["offset"] = df["offset"] * 0.8 - 0.4
                fig, ax = plt.subplots(2, 2, figsize=(12, 9))
                plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
                ax[0][0].set_title(createTitle(df["est1"], df["GS"]))
                ax[0][0].scatter(df[~c_home]["est1"], df[~c_home]["GS"] + df[~c_home]["offset"], alpha=0.1, color="red")
                ax[0][0].scatter(df[c_home]["est1"], df[c_home]["GS"] + df[c_home]["offset"], alpha=0.1, color=default_color)
                ax[0][1].set_title(createTitle(df["est2"], df["GC"]))
                ax[0][1].scatter(df[~c_home]["est2"], df[~c_home]["GC"] + df[~c_home]["offset"], alpha=0.1, color="red")
                ax[0][1].scatter(df[c_home]["est2"], df[c_home]["GC"] + df[c_home]["offset"], alpha=0.1, color=default_color)
                ax[1][0].set_title(createTitle(df["est1"] - df["est2"], df["GS"] - df["GC"]))
                ax[1][0].scatter(df[~c_home]["est1"] - df[~c_home]["est2"],
                              df[~c_home]["GS"] - df[~c_home]["GC"] + df[~c_home]["offset"], alpha=0.1, color="red")
                ax[1][0].scatter(df[c_home]["est1"] - df[c_home]["est2"],
                              df[c_home]["GS"] - df[c_home]["GC"] + df[c_home]["offset"], alpha=0.1, color=default_color)

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
                    ax[1][1].scatter(est1[cond1 & c_tendency], est2[cond1 & c_tendency], alpha=alpha, color=color, marker='o')
                    ax[1][1].scatter(est1[cond1 & ~c_tendency], est2[cond1 & ~c_tendency], alpha=alpha, color=color, marker='x')
                    ax[1][1].scatter(est2[cond2 & c_tendency], est1[cond2 & c_tendency], alpha=alpha, color=color, marker='o')
                    ax[1][1].scatter(est2[cond2 & ~c_tendency], est1[cond2 & ~c_tendency], alpha=alpha, color=color, marker='x')

                plotEstimates(c_win & c_home, c_loss & ~c_home, "blue")
                plotEstimates(c_loss & c_home, c_win & ~c_home, "red")
                plotEstimates(c_draw & c_home, c_draw & ~c_home, "green", 0.3)
                ax[1][1].set_title("Points: {:.4f} ({:.2f}%), H: {:.1f}, D: {:.1f}, A: {:.1f}".format(
                    np.sum(t2["Contribution"]),
                    100.0 * (1 - pie_chart_values[0] / len(df)),
                    100.0 * tendency_values[2] / len(df),
                    100.0 * tendency_values[1] / len(df),
                    100.0 * tendency_values[0] / len(df)
                ))
                plt.show()
                #plt.close()

            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            fig, ax = plt.subplots(2, 2, figsize=(10, 9))
            plot_confusion_matrix(ax[0][0], cnf_matrix, classes=["AwayWin", "Draw", "HomeWin"],
                                  title='Tendency', cmap=default_cmap)

            # Plot normalized confusion matrix
            plot_confusion_matrix(ax[1][0], cnf_matrix, classes=["AwayWin", "Draw", "HomeWin"],
                                  normalize=True,
                                  title='Tendency', cmap=default_cmap)
            #plt.show()
            #plt.close()

            y_pred = [(p1 - p2) if w == "Home" else (p2 - p1) for p1, p2, w in zip(df["pGS"], df["pGC"], df["Where"])]
            y_test = [(p1 - p2) if w == "Home" else (p2 - p1) for p1, p2, w in zip(df["GS"], df["GC"], df["Where"])]
            y_test = [min(3, y) for y in y_test]
            y_test = [max(-3, y) for y in y_test]
            y_pred = [min(3, y) for y in y_pred]
            y_pred = [max(-3, y) for y in y_pred]
            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            #fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            plot_confusion_matrix(ax[0][1], cnf_matrix, classes=np.unique(y_test).tolist(),
                                  title='GoalDiff', cmap=default_cmap)

            # Plot normalized confusion matrix
            plot_confusion_matrix(ax[1][1], cnf_matrix, classes=np.unique(y_test).tolist(),
                                  normalize=True,
                                  title='GoalDiff', cmap=default_cmap)
            plt.show()
            #plt.close()

        if prefix == "ens/":
            t6 = df.pivot_table(index=["Strategy"],
                                aggfunc={'0_none': np.sum, '1_tendency': np.sum, '2_diff': np.sum, '3_full': np.sum,
                                         'total_points': [len, np.sum, np.mean],
                                         "hit_goal": np.mean, "goal_freq": np.mean},
                                margins=False, fill_value=0)

            t6.reset_index(inplace=True)
            t6.columns = ['_'.join(col).strip() for col in t6.columns.values]
            t6 = t6.rename(columns={"Strategy_": "Strategy",
                                    "0_none_sum": "None",
                                    "1_tendency_sum": "Tendency",
                                    "1_tendency_sum": "Tendency",
                                    "2_diff_sum": "Diff",
                                    "3_full_sum": "Full",
                                    "total_points_len": "Total",
                                    "hit_goal_mean": "ActualRate",
                                    "goal_freq_mean": "TargetRate",
                                    "total_points_mean": "AvgPoints",
                                    "total_points_sum": "TotalPoints",
                                    })
            t6 = t6.assign(EffRate=t6.ActualRate - t6.TargetRate, Contribution=t6.TotalPoints / len(df))
            t6 = t6[['Strategy', 'None', 'Tendency', 'Diff', 'Full', 'Total',
                     'ActualRate', 'TargetRate', 'EffRate',
                     'AvgPoints', 'TotalPoints', 'Contribution']]

            if not silent:
                print()
                print(t6)

                plt.axis('equal')
                plt.pie(t6["Total"], labels=t6['Strategy'], autopct='%1.1f%%')
                plt.show()
                #plt.close()



    def prepare_label_fit(predictions, features, labels, team_onehot_encoder, label_column_names, skip_plotting=False,
                          output_name="outputs_poisson"):
        features = features["match_input_layer"]
        features = features[:len(predictions)]  # cut off features if not enough predictions are present
        labels = labels[:len(predictions)]  # cut off labels if not enough predictions are present
        tn = len(team_onehot_encoder.classes_)
        df = pd.DataFrame()
        df['Team1'] = team_onehot_encoder.inverse_transform(features[:, 4:4 + tn])
        df['Team2'] = team_onehot_encoder.inverse_transform(features[:, 4 + tn:4 + 2 * tn])
        df['Where'] = ['Home' if h == 1 else 'Away' for h in features[:, 1]]

        #  df = pd.DataFrame().from_csv("file:///C:/tmp/Football/models_multi_2017/train_outputs_poisson.csv")
        #  tn=36
        #  label_column_names = df.columns[4::2]
        # print(label_column_names)
        fig = None
        if not skip_plotting:
            fig, ax = plt.subplots(1 + (len(label_column_names) // 4), 4, figsize=(20, 70))
        for i, col in enumerate(label_column_names):
            # print(i)
            df["p_" + col] = [np.exp(p[output_name][i]) for p in predictions]
            df[col] = labels[:, i]
            dfcorr = df[["p_" + col, col]]
            cor_p = dfcorr[col].corr(dfcorr["p_" + col], method='pearson')
            cor_s = dfcorr[col].corr(dfcorr["p_" + col], method='spearman')
            #    print(col)
            #    print(cor_p)
            #    print(cor_s)
            if not skip_plotting:
                ax0 = ax[i // 4, np.mod(i, 4)]
                df.boxplot(column="p_" + col,
                           by=col, ax=ax0, fontsize=10, grid=False)
                ax0.set_title('cor_p={:.4f}, cor_s={:.4f}'.format(cor_p, cor_s))

        # plt.show()
        # fig.savefig("C:/tmp/Football/models_multi_2017/train_outputs_poisson.pdf")
        return df, fig


    def print_prediction_summary(predictions, ens_predictions=None):
        for pt in ['Pt']:
            print("Begin {} ---------------------------".format(pt))
            p1 = predictions[['Where', pt, 'Prefix']].copy()
            p1["Season"] = p1.index // 612
            p2 = p1[['Season', 'Prefix', pt, 'Where']]
            print("Season Home/Away")
            print(p2.pivot_table(index=['Season', "Where"], values=pt, aggfunc=np.mean, columns="Prefix", margins=True))

            print("Season Complete")
            print(p2.pivot_table(index=['Season'], values=pt, aggfunc=np.mean, columns="Prefix", margins=True))

            if ens_predictions is None:
                return

            p4 = ens_predictions[[pt, 'Prefix', "Strategy"]].copy()
            p4["Season"] = p4.index // 306
            p4 = p4[['Season', 'Prefix', pt, 'Strategy']]
            print("Ensemble Strategy Mix")
            print(pd.crosstab(p4['Season'], columns=p4["Strategy"], normalize="index", margins=True))
            print("Ensemble Strategy Contribution")
            print(p4.pivot_table(index=['Season'], values=pt, aggfunc=[np.mean], columns=["Prefix", "Strategy"],
                                 margins=True))
            # print(p4.pivot_table(index=['Season'], values=pt, aggfunc=np.mean, columns="Prefix", margins=False))
            print("End {} ---------------------------".format(pt))


    def plot_point_summary(results):
        data = results.loc[results.dataset.isin(["train", "test"])]
        data = pd.pivot_table(data, index="Prefix", values="Pt", columns="dataset")
        fig, ax = plt.subplots(figsize=(10, 6))
        data.plot(kind='bar', stacked=False, ax=ax,
                  title='Points by strategy',
                  legend=True, table=False, use_index=True,
                  fontsize=12, grid=True,
                  ylim=(data.test.max() * 0.7, 0.06 + np.max(data[["train", "test"]].max())))
        ax.axhline(np.max(data.test), color="red")
        ax.annotate("{:.04f}".format(data.test.max()), [np.argmax(list(data.test)), 0.03 + data.test.max()], fontsize=15)
        ax.set_xlabel('Strategy')
        ax.set_ylabel("Points")
        fig.tight_layout()
        plt.show()


    def plot_checkpoints(df, predictions):
        for prefix in ["ngb"]:
            prefix_df = df.loc[df.Prefix == prefix[:-1]]
            s = random.sample(range(len(prefix_df)), 1)[0]
            print(prefix, s)
            # print({k:v.shape for k,v in predictions.items()})
            sample_preds = {k: v[s] for k, v in predictions.items()}
            plot_softprob(sample_preds, prefix_df["GS"][s], prefix_df["GC"][s],
                          prefix_df["Time"][s] + " : " + prefix_df["Team1"][s] + " - " + prefix_df["Team2"][s] + " (" +
                          prefix_df["Where"][s] + ")", prefix=prefix)
            s = random.sample(range(len(prefix_df)), 1)[0]
            sample_preds = {k: v[s] for k, v in predictions.items()}
            plot_softprob(sample_preds, prefix_df["GS"][s], prefix_df["GC"][s],
                          prefix_df["Time"][s] + " : " + prefix_df["Team1"][s] + " - " + prefix_df["Team2"][s] + " (" +
                          prefix_df["Where"][s] + ")", prefix=prefix)
            for dataset in ["test", "train"]:
                plot_predictions_3(df, prefix[:-1], dataset)

        print_prediction_summary(df.loc[df.dataset == "test"], None)  # test_ens_predictions)
        print_prediction_summary(df.loc[df.dataset == "train"], None)  # , train_ens_predictions)
        plot_point_summary(df)


    def get_input_data(model_dir, train_data, test_data, data_dir, useBWIN):
        all_data, all_labels, all_features, teamnames, team_mapping = get_train_test_data(model_dir, train_data, test_data,
                                                                                          data_dir, useBWIN)
        print(all_data.iloc[1000])

        features_arrays, labels_array, team_onehot_encoder, label_column_names = build_features(all_data.copy(), all_labels,
                                                                                                all_features, teamnames,
                                                                                                team_mapping)
        print(features_arrays["match_input_layer"][:, 23:26])
        data = (all_data, teamnames, features_arrays, labels_array, team_onehot_encoder, label_column_names)
        return data


    ind_win = [i_gs * 7 + i_gc for i_gs in range(7) for i_gc in range(7) if i_gs > i_gc]
    ind_loss = [i_gs * 7 + i_gc for i_gs in range(7) for i_gc in range(7) if i_gs < i_gc]
    ind_draw = [i_gs * 7 + i_gc for i_gs in range(7) for i_gc in range(7) if i_gs == i_gc]


    def plot_probs(probs, softpoints, gs, gc, title=""):
        print("-----------------------------------------------------")
        print("{} / {}:{}".format(title, gs, gc))
        default_color = np.array([[0.12, 0.47, 0.71, 0.5]])

        sp = probs
        # print(sp)
        spt = softpoints
        gs = min(gs, 6)
        gc = min(gc, 6)
        pp = probs.reshape((7, 7))
        margin_pred_prob1 = np.sum(pp, axis=1)
        margin_pred_prob2 = np.sum(pp, axis=0)
        margin_pred_expected1 = np.sum(np.arange(7) * margin_pred_prob1)
        margin_pred_expected2 = np.sum(np.arange(7) * margin_pred_prob2)
        margin_poisson_prob1 = [poisson.pmf(i, margin_pred_expected1) for i in range(7)]
        margin_poisson_prob2 = [poisson.pmf(i, margin_pred_expected2) for i in range(7)]
        # margin_pred_prob1 = pred[prefix+"p_marg_1"]
        # margin_poisson_prob1 = pred[prefix+"p_poisson_1"]
        # margin_pred_prob2 = pred[prefix+"p_marg_2"]
        # margin_poisson_prob2 = pred[prefix+"p_poisson_2"]
        # margin_pred_expected1 = pred[prefix+"ev_goals_1"]
        # margin_pred_expected2 = pred[prefix+"ev_goals_2"]
        g = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        g1 = [0.0] * 7 + [1.0] * 7 + [2.0] * 7 + [3.0] * 7 + [4.0] * 7 + [5.0] * 7 + [6.0] * 7
        g2 = g * 7
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(g1, g2, s=sp * 10000, alpha=0.4, color=default_color)
        ax[0].scatter(gs, gc, s=sp[gs * 7 + gc] * 10000, alpha=0.7, color=default_color)
        for i, txt in enumerate(sp):
            ax[0].annotate("{:4.2f}".format(txt * 100), (g1[i] - 0.3, g2[i] - 0.1))
        ax[1].scatter(g1, g2, s=spt * 500, alpha=0.4, color='red')
        ax[1].scatter(gs, gc, s=spt[gs * 7 + gc] * 500, alpha=0.7, color='red')
        for i, txt in enumerate(spt):
            ax[1].annotate("{:4.2f}".format(txt), (g1[i] - 0.3, g2[i] - 0.1))
        ax[0].set_title("p")
        ax[1].set_title("ev")
        ax[2].set_title(title)
        max_sp = max(sp)
        max_sp_index = np.argmax(sp)
        ax[0].scatter((max_sp_index // 7).astype(float), np.mod(max_sp_index, 7).astype(float), s=max_sp * 10000.0,
                      facecolors='none', edgecolors='black', linewidth=2)
        max_spt = max(spt)
        max_spt_index = np.argmax(spt)
        ax[1].scatter((max_spt_index // 7).astype(float), np.mod(max_spt_index, 7).astype(float), s=max_spt * 500.0,
                      facecolors='none', edgecolors='black', linewidth=2)

        p_loss = 0.0
        p_win = 0.0
        p_draw = 0.0
        for i in range(7):
            for j in range(7):
                if i > j:
                    p_win += sp[i * 7 + j]
                if i < j:
                    p_loss += sp[i * 7 + j]
                if i == j:
                    p_draw += sp[i * 7 + j]
        ax[2].axis('equal')
        explode = [0, 0, 0]
        explode[1 - np.sign(gs - gc)] = 0.1
        wedges, _, _ = ax[2].pie([p_win, p_draw, p_loss], labels=["Win", "Draw", "Loss"], colors=["blue", "green", "red"],
                                 startangle=90, autopct='%1.1f%%',
                                 radius=1.0, explode=explode, wedgeprops={"alpha": 0.5})
        wedges[1 - np.sign(gs - gc)].set_alpha(0.8)
        plt.show()

        w = 0.35
        fig, ax = plt.subplots(1, 3, figsize=(15, 1))
        ax[0].bar(g, margin_pred_prob1, alpha=0.6, width=w, color=default_color)
        ax[0].bar([x + w for x in g], margin_poisson_prob1, alpha=0.3, color="red", width=0.35)
        ax[0].bar(gs, margin_pred_prob1[gs], alpha=0.5, width=w, color=default_color)
        ax[0].bar(gs + w, margin_poisson_prob1[gs], alpha=0.7, color="red", width=0.35)
        ax[0].axvline(x=margin_pred_expected1, color='red')
        ax[1].bar(g, margin_pred_prob2, alpha=0.6, width=w, color=default_color)
        ax[1].bar([x + w for x in g], margin_poisson_prob2, alpha=0.3, color="red", width=0.35)
        ax[1].bar(gc, margin_pred_prob2[gc], alpha=0.5, width=w, color=default_color)
        ax[1].bar(gc + w, margin_poisson_prob2[gc], alpha=0.7, color="red", width=0.35)
        ax[1].axvline(x=margin_pred_expected2, color='red')
        ax[0].set_title(margin_pred_expected1)
        ax[1].set_title(margin_pred_expected2)
        bars = ax[2].bar([0, 1, 2], height=[p_win, p_draw, p_loss], tick_label=["Win", "Draw", "Loss"],
                         color=["blue", "green", "red"], alpha=0.5)
        bars[1 - np.sign(gs - gc)].set_alpha(0.8)
        bars[1 - np.sign(gs - gc)].set_linewidth(2)
        bars[1 - np.sign(gs - gc)].set_edgecolor("black")
        for i in [-1, 0, 1]:
            if i != np.sign(gs - gc):
                bars[1 - i].set_hatch("x")
        plt.show()


    def plot_softprob_simple(sp, title="", prefix=""):
        g = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        g1 = [0.0] * 7 + [1.0] * 7 + [2.0] * 7 + [3.0] * 7 + [4.0] * 7 + [5.0] * 7 + [6.0] * 7
        g2 = g * 7
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].scatter(g1, g2, s=sp * 10000, alpha=0.4)
        for i, txt in enumerate(sp):
            ax[0].annotate("{:4.2f}".format(txt * 100), (g1[i] - 0.3, g2[i] - 0.1))
        ax[0].set_title(prefix)
        max_sp = max(sp)
        max_sp_index = np.argmax(sp)
        ax[0].scatter((max_sp_index // 7).astype(float), np.mod(max_sp_index, 7).astype(float), s=max_sp * 10000.0,
                      facecolors='none', edgecolors='black', linewidth=2)

        p_loss = 0.0
        p_win = 0.0
        p_draw = 0.0
        for i in range(7):
            for j in range(7):
                if i > j:
                    p_win += sp[i * 7 + j]
                if i < j:
                    p_loss += sp[i * 7 + j]
                if i == j:
                    p_draw += sp[i * 7 + j]
        ax[1].axis('equal')
        wedges, _, _ = ax[1].pie([p_win, p_draw, p_loss], labels=["Win", "Draw", "Loss"],
                                 colors=["blue", "green", "red"],
                                 startangle=90, autopct='%1.1f%%',
                                 radius=1.0, wedgeprops={"alpha": 0.5})
        plt.show()

    def plot_softprob_grid(sp, ax0, ax1, prefix=""):
        g = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        g1 = [0.0] * 7 + [1.0] * 7 + [2.0] * 7 + [3.0] * 7 + [4.0] * 7 + [5.0] * 7 + [6.0] * 7
        g2 = g * 7
        ax0.scatter(g1, g2, s=sp * 5000, alpha=0.4)
        # for i, txt in enumerate(sp):
        #     ax0.annotate("{:4.2f}".format(txt * 100), (g1[i] - 0.3, g2[i] - 0.1))
        for i, txt in enumerate(sp):
            ax0.text(g1[i] - 0.3, g2[i] - 0.1, "{:4.2f}".format(txt * 100), size=8)
        ax0.axis('off')
        # Turn off tick labels
        #ax0.set_yticklabels([])
        #ax0.set_xticklabels([])

        max_sp = max(sp)
        max_sp_index = np.argmax(sp)
        title = ax1.get_title()
        if title is None or title == "":
            ax1.set_title(prefix + " " + str(max_sp_index // 7)+":"+ str(np.mod(max_sp_index, 7)))
        else:
            ax1.set_title(title + " " + str(max_sp_index // 7) + ":" + str(np.mod(max_sp_index, 7)))
        ax0.scatter((max_sp_index // 7).astype(float), np.mod(max_sp_index, 7).astype(float), s=max_sp * 5000.0,
                      facecolors='none', edgecolors='black', linewidth=1)

        p_loss = 0.0
        p_win = 0.0
        p_draw = 0.0
        for i in range(7):
            for j in range(7):
                if i > j:
                    p_win += sp[i * 7 + j]
                if i < j:
                    p_loss += sp[i * 7 + j]
                if i == j:
                    p_draw += sp[i * 7 + j]
        ax1.axis('equal')
        wedges, _, _ = ax1.pie([p_win, p_draw, p_loss], labels=["Win", "Draw", "Loss"],
                                 colors=["blue", "green", "red"],
                                 startangle=90, autopct='%1.1f%%',
                                 radius=1.0, wedgeprops={"alpha": 0.5})

    def dispatch_main(target_distr, model_dir, train_steps, train_data, test_data,
                      checkpoints, save_steps, data_dir, max_to_keep,
                      reset_variables, skip_plotting, target_system, modes, use_swa, histograms, useBWIN):
        """Train and evaluate the model."""
        # train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
        # Specify file path below if want to find the output easily
        model_dir = tempfile.mkdtemp() if not model_dir else model_dir
        print(model_dir)

        global point_scheme
        global point_scale
        if target_system == "TCS":
            point_scheme = point_scheme_tcs
            point_scale = 1
        elif target_system == "Pistor":
            point_scheme = point_scheme_pistor
            point_scale = 4
        elif target_system == "Sky":
            point_scheme = point_scheme_sky
            point_scale = 3
        elif target_system == "GoalDiff":
            point_scheme = point_scheme_goal_diff
            point_scale = 0.25
        else:
            raise Exception("Unknown point scheme")

        train_data = [s.strip() for s in train_data.strip('[]').split(',')]
        test_data = [s.strip() for s in test_data.strip('[]').split(',')]

        all_data, teamnames, features_arrays, labels_array, team_onehot_encoder, label_column_names = get_input_data(
            model_dir, train_data, test_data, data_dir, useBWIN)
        print(features_arrays["match_input_layer"][:, 23:26])

        #  train_idx = range(2*306*len(train_data))
        #  test_idx = range(2*306*len(train_data), 2*306*len(train_data)+2*306*len(test_data))
        #  print(train_idx)
        #  print(test_idx)
        print(target_system)
        print(all_data.shape)
        print(labels_array.shape)
        print(labels_array.dtype)
        print([(k, v.shape, v.dtype) for k, v in features_arrays.items()])

        #  print(feature_columns)
        #  print(teamnames)
        train_idx = all_data.index[all_data['Train']].tolist()
        test_idx = all_data.index[all_data['Test']].tolist()
        pred_idx = all_data.index[all_data['Predict']].tolist()
        team1 = all_data.Team1[pred_idx].tolist()
        team2 = all_data.Team2[pred_idx].tolist()

        # skip first rounds if test data is placed in front
        if test_idx and np.min(test_idx) == 0 and np.min(train_idx) > 45:
            test_idx = [t for t in test_idx if t > 45]

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

        return model_data, point_scheme, label_column_names, team1, team2


    FLAGS = None


    def main(_):
        #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/Mapping_Layer/WM", target_file_name="mapping.csv", all_tensor_names=False, all_tensors=False)
        #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel", target_file_name="rnn_candidate_kernel.csv", all_tensor_names=False, all_tensors=False)
        #  utils.print_tensors_in_checkpoint_file(FLAGS.model_dir, tensor_name="Model/RNN_1/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel", target_file_name="rnn_gates_kernel.csv", all_tensor_names=False, all_tensors=False)
        target_distr = [(5, 20, 35), 10, (15, 8, 2),
                        (20, 20, 80)]  # [(3:0, 3:1, 2:1), 1:1, (1:2, 1:3, 0:3), (0:0, 0:1/1:0, 0:2/2:0)]

        if FLAGS.target_system == "Pistor":
            # Pistor
            target_distr = {"cp": [(2, 10, 48), 15, (16, 8, 1), (2, 2, 50)],
                            "sp": [(2, 20, 43), 15, (14, 5, 1), (20, 20, 80)],
                            #                "pgpt":[(5, 20, 35), 25, (8, 5, 2), (20, 20, 80)],
                            "pg2": [(2, 12, 48), 5, (22, 5, 1), (20, 20, 80)],
                            "av": [(2, 10, 43), 15, (21, 8, 1), (2, 2, 50)],
                            }
        elif FLAGS.target_system == "Sky" or FLAGS.target_system == "GoalDiff":
            # Sky
            target_distr = {"cp": [(5, 20, 45), 1, (22, 5, 2), (1, 1, 85)],
                            "sp": [(6, 10, 58), 1, (19, 5, 2), (10, 10, 90)],
                            #                "pgpt":[(5, 20, 35), 25, (8, 5, 2), (20, 20, 80)],
                            "pg2": [(2, 13, 55), 1, (18, 9, 2), (30, 30, 70)],
                            "av": [(7, 15, 48), 0, (23, 5, 2), (3, 3, 70)],
                            }
        else:
            raise ("Wrong system")

        return dispatch_main(target_distr, FLAGS.model_dir, FLAGS.train_steps,
                             FLAGS.train_data, FLAGS.test_data, FLAGS.checkpoints,
                             FLAGS.save_steps, FLAGS.data_dir, FLAGS.warmup,
                             FLAGS.reset_variables, FLAGS.skip_plotting, FLAGS.target_system,
                             FLAGS.action, FLAGS.prefix, FLAGS.histograms,
                             FLAGS.useBWIN)


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.register("type", "bool", lambda v: v.lower() == "true")
        parser.add_argument(
            "--data_dir",
            type=str,
            default="c:/git/football/TF/data",
            # default="d:/gitrepository/Football/football/TF/data",
            help="input data"
        )
        parser.add_argument(
            "--model_dir",
            type=str,
            default="d:/Models/model_tfp",
            help="Base directory for output models."
        )
        parser.add_argument(
            "--skip_plotting", type=bool,
            default=True,
            # default=False,
            help="Print plots of predicted data"
        )
        parser.add_argument(
            "--reset_variables", type=str,  # nargs='+',
            default="cbsp",
            # default=300,
            help="List of variable names to be re-initialized during upgrade"
        )
        parser.add_argument(
            "--histograms", type=bool,
            # default=True,
            default=False,
            help="create histogram data in summary files"
        )
        parser.add_argument(
            "--warmup", type=int,
            default=5,
            help="Number of initial rounds without Gaussian Process meta-parameter search."
        )
        parser.add_argument(
            "--train_steps", type=int,
            default=5,
            help="Number of training steps."
        )
        parser.add_argument(
            "--save_steps", type=int,
            # default=200,
            #default=300,
            default=0,
            help="Number of training steps between checkpoint files."
        )
        parser.add_argument(
            "--train_data", type=str,
            # default="0910,1112,1314,1516,1718,1920", #
            #default="0910,1011,1112,1213,1314,1415", #
            #default="0910,1011,1112,1213,1314",
            #default="0910,1011,1112,1213,1314,1415,1516,1617,1718", #
            #default="0910,1011,1112,1213,1314,1415,1516,1617,1718,1819,1920", #
            #default="1617,1718",  #
            #default="1819,1920",
            #default="1920",
            default="1415,1516,1617,1718",
            help="Path to the training data."
        )
        parser.add_argument(
            "--test_data", type=str,
            # default="1011,1213,1415,1617,1819", #
            #default="0910,1011,1112,1213,1314,1415", #
            #default="1516,1617,1718,1819,1920,2021",  #
            #default="2021", #
            #default="1819,1920,2021",
            default="1819,1920,2021,2122", #
            help="Path to the test data."
        )
        parser.add_argument(
            "--target_system",
            type=str,
            #default="Pistor",
            #default="Sky",
            # default="TCS",
            default="GoalDiff",
            help="Point system to optimize for"
        )
        parser.add_argument(
            "--prefix", type=str,
            default="vi8_2122_batch",
            #default="vi8relubatch",
            help="The prefix to be used for model files"
        )
        parser.add_argument(
            "--checkpoints", type=str,
            default="best",
            #default="-6",  # slice(-2, None)
            #default="",
             #default="20201201_081025", # Sky 2
            #default="20201130_141446", # Pistor 2
            # default="60000:92000",
            help="Range of checkpoints for evaluation / prediction. Format: "
        )
        parser.add_argument(
            "--action",
            type=str,
            # default="static",
            #default="train",
            # default="eval_stop",
            default="eval",
            # default="predict",
            # default="upgrade",
            # default="train_eval",
            # default="upgrade,train,eval,predict",
            help="What to do"
        )
        parser.add_argument(
            "--useBWIN", type=bool,
            default=True,
            #default=False,
            help="Run in Stochastic Weight Averaging mode."
        )
        FLAGS, unparsed = parser.parse_known_args()
        print([sys.argv[0]] + unparsed)
        print(FLAGS)
        model_data, point_scheme, label_column_names, team1, team2 = main([sys.argv[0]] + unparsed)

        # Path('C:/tmp/Football/models/reset.txt').touch()

        features_arrays, labels_array, train_idx, test_idx, pred_idx = model_data

        # print(label_column_names)
        print(features_arrays["match_input_layer"].shape)
        print(labels_array.shape)

        if point_scheme == point_scheme_pistor:
            point_matrix = np.array([[3 if (i // 7 == j // 7) and (np.mod(i, 7) == np.mod(j, 7)) else
                                      2 if (i // 7 - np.mod(i, 7) == j // 7 - np.mod(j, 7)) else
                                      1 if (np.sign(i // 7 - np.mod(i, 7)) == np.sign(j // 7 - np.mod(j, 7))) else
                                      0 for i in range(49)] for j in range(49)])
        elif point_scheme == point_scheme_sky:
            point_matrix = np.array([[5 if (i // 7 == j // 7) and (np.mod(i, 7) == np.mod(j, 7)) else
                                      2 if (np.sign(i // 7 - np.mod(i, 7)) == np.sign(j // 7 - np.mod(j, 7))) else
                                      0 for i in range(49)] for j in range(49)])
        if point_scheme == point_scheme_goal_diff:
            point_matrix = np.array([[np.maximum(5, 10 - np.abs(i // 7 - j // 7) - np.abs(np.mod(i, 7) - np.mod(j, 7))) if (
                    np.sign(i // 7 - np.mod(i, 7)) == np.sign(j // 7 - np.mod(j, 7))) else
                                      0 for i in range(49)] for j in range(49)])
        home_away_inversion_matrix = np.array([[1 if (i // 7) == np.mod(j, 7) and (j // 7) == np.mod(i, 7) else 0 for i in range(49)] for j in range(49)])

        X = features_arrays["match_input_layer"]
        label_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 46, 47, 48, 49, 50, 51, 52, 53]
        # X = np.concatenate([X,
        #                     np.mean(labels_array[features_arrays["match_history_t1"][:, -5:]][:, :, label_filter], axis=1),
        #                     np.mean(labels_array[features_arrays["match_history_t2"][:, -5:]][:, :, label_filter], axis=1),
        #                     np.mean(labels_array[features_arrays["match_history_t12"][:, -2:]][:, :, label_filter], axis=1)
        #                     # labels_array[features_arrays["match_history_t2"][:,-5:]][:,:,label_filter].reshape((-1,5*len(label_filter))),
        #                     # labels_array[features_arrays["match_history_t12"][:,-2:]][:,:,label_filter].reshape((-1,2*len(label_filter)))
        #                     ], axis=1)

        # label_column_sublist = label_column_names[18:21]
        label_column_sublist = label_column_names[label_filter]
        # label_column_sublist = label_column_names
        label_column_names_extended = \
            ["T1_Hi5_" + x for x in label_column_sublist.tolist()] + \
            ["T2_Hi5_" + x for x in label_column_sublist.tolist()] + \
            ["T12_Hi2_" + x for x in label_column_sublist.tolist()]

        feature_names = ['t1histlen', 'where', 't2histlen', 't12histlen']
        feature_names += ['Time', 't1games', 't1dayssince', 't2dayssince', 't1dayssince_ema', 't2dayssince_ema',
                          'roundsleft', 't1promoted', 't2promoted', 't1points', 't2points', 't1rank', 't2rank',
                          't1rank6_attention', 't2rank6_attention', 't1rank16_attention', 't2rank16_attention',
                          't1cards_ema', 't2cards_ema', 'BW1', 'BW0', 'BW2', 'T1_CUM_T1_GFT', 'T2_CUM_T2_GFT',
                          'T1_CUM_T1_W_GFT', 'T2_CUM_T2_W_GFT', 'T1_CUM_T2_GFT', 'T2_CUM_T1_GFT', 'T1_CUM_T2_W_GFT',
                          'T2_CUM_T1_W_GFT', 'T12_CUM_T1_GFT', 'T12_CUM_T1_W_GFT', 'T21_CUM_T2_GFT', 'T21_CUM_T2_W_GFT',
                          'T12_CUM_T12_GFT', 'T12_CUM_T12_W_GFT', 'T1221_CUM_GFT', 'T1221_CUM_W_GFT', 'T1_CUM_T1_GHT',
                          'T2_CUM_T2_GHT', 'T1_CUM_T1_W_GHT', 'T2_CUM_T2_W_GHT', 'T1_CUM_T2_GHT', 'T2_CUM_T1_GHT',
                          'T1_CUM_T2_W_GHT', 'T2_CUM_T1_W_GHT', 'T12_CUM_T1_GHT', 'T12_CUM_T1_W_GHT', 'T21_CUM_T2_GHT',
                          'T21_CUM_T2_W_GHT', 'T12_CUM_T12_GHT', 'T12_CUM_T12_W_GHT', 'T1221_CUM_GHT', 'T1221_CUM_W_GHT',
                          'T1_CUM_T1_S', 'T2_CUM_T2_S', 'T1_CUM_T1_W_S', 'T2_CUM_T2_W_S', 'T1_CUM_T2_S', 'T2_CUM_T1_S',
                          'T1_CUM_T2_W_S', 'T2_CUM_T1_W_S', 'T12_CUM_T1_S', 'T12_CUM_T1_W_S', 'T21_CUM_T2_S',
                          'T21_CUM_T2_W_S', 'T12_CUM_T12_S', 'T12_CUM_T12_W_S', 'T1221_CUM_S', 'T1221_CUM_W_S',
                          'T1_CUM_T1_ST', 'T2_CUM_T2_ST', 'T1_CUM_T1_W_ST', 'T2_CUM_T2_W_ST', 'T1_CUM_T2_ST',
                          'T2_CUM_T1_ST', 'T1_CUM_T2_W_ST', 'T2_CUM_T1_W_ST', 'T12_CUM_T1_ST', 'T12_CUM_T1_W_ST',
                          'T21_CUM_T2_ST', 'T21_CUM_T2_W_ST', 'T12_CUM_T12_ST', 'T12_CUM_T12_W_ST', 'T1221_CUM_ST',
                          'T1221_CUM_W_ST', 'T1_CUM_T1_F', 'T2_CUM_T2_F', 'T1_CUM_T1_W_F', 'T2_CUM_T2_W_F', 'T1_CUM_T2_F',
                          'T2_CUM_T1_F', 'T1_CUM_T2_W_F', 'T2_CUM_T1_W_F', 'T12_CUM_T1_F', 'T12_CUM_T1_W_F', 'T21_CUM_T2_F',
                          'T21_CUM_T2_W_F', 'T12_CUM_T12_F', 'T12_CUM_T12_W_F', 'T1221_CUM_F', 'T1221_CUM_W_F',
                          'T1_CUM_T1_C', 'T2_CUM_T2_C', 'T1_CUM_T1_W_C', 'T2_CUM_T2_W_C', 'T1_CUM_T2_C', 'T2_CUM_T1_C',
                          'T1_CUM_T2_W_C', 'T2_CUM_T1_W_C', 'T12_CUM_T1_C', 'T12_CUM_T1_W_C', 'T21_CUM_T2_C',
                          'T21_CUM_T2_W_C', 'T12_CUM_T12_C', 'T12_CUM_T12_W_C', 'T1221_CUM_C', 'T1221_CUM_W_C',
                          'T1_CUM_T1_Y', 'T2_CUM_T2_Y', 'T1_CUM_T1_W_Y', 'T2_CUM_T2_W_Y', 'T1_CUM_T2_Y', 'T2_CUM_T1_Y',
                          'T1_CUM_T2_W_Y', 'T2_CUM_T1_W_Y', 'T12_CUM_T1_Y', 'T12_CUM_T1_W_Y', 'T21_CUM_T2_Y',
                          'T21_CUM_T2_W_Y', 'T12_CUM_T12_Y', 'T12_CUM_T12_W_Y', 'T1221_CUM_Y', 'T1221_CUM_W_Y',
                          'T1_CUM_T1_R', 'T2_CUM_T2_R', 'T1_CUM_T1_W_R', 'T2_CUM_T2_W_R', 'T1_CUM_T2_R', 'T2_CUM_T1_R',
                          'T1_CUM_T2_W_R', 'T2_CUM_T1_W_R', 'T12_CUM_T1_R', 'T12_CUM_T1_W_R', 'T21_CUM_T2_R',
                          'T21_CUM_T2_W_R', 'T12_CUM_T12_R', 'T12_CUM_T12_W_R', 'T1221_CUM_R', 'T1221_CUM_W_R',
                          'T1_CUM_T1_xG', 'T2_CUM_T2_xG', 'T1_CUM_T1_W_xG', 'T2_CUM_T2_W_xG', 'T1_CUM_T2_xG',
                          'T2_CUM_T1_xG', 'T1_CUM_T2_W_xG', 'T2_CUM_T1_W_xG', 'T12_CUM_T1_xG', 'T12_CUM_T1_W_xG',
                          'T21_CUM_T2_xG', 'T21_CUM_T2_W_xG', 'T12_CUM_T12_xG', 'T12_CUM_T12_W_xG', 'T1221_CUM_xG',
                          'T1221_CUM_W_xG', 'T1_CUM_T1_GH2', 'T2_CUM_T2_GH2', 'T1_CUM_T1_W_GH2', 'T2_CUM_T2_W_GH2',
                          'T1_CUM_T2_GH2', 'T2_CUM_T1_GH2', 'T1_CUM_T2_W_GH2', 'T2_CUM_T1_W_GH2', 'T12_CUM_T1_GH2',
                          'T12_CUM_T1_W_GH2', 'T21_CUM_T2_GH2', 'T21_CUM_T2_W_GH2', 'T12_CUM_T12_GH2', 'T12_CUM_T12_W_GH2',
                          'T1221_CUM_GH2', 'T1221_CUM_W_GH2', 'T1_CUM_T1_Win', 'T2_CUM_T2_Win', 'T1_CUM_T1_W_Win',
                          'T2_CUM_T2_W_Win', 'T1_CUM_T1_HTWin', 'T2_CUM_T2_HTWin', 'T1_CUM_T1_W_HTWin', 'T2_CUM_T2_W_HTWin',
                          'T1_CUM_T1_Loss', 'T2_CUM_T2_Loss', 'T1_CUM_T1_W_Loss', 'T2_CUM_T2_W_Loss', 'T1_CUM_T1_HTLoss',
                          'T2_CUM_T2_HTLoss', 'T1_CUM_T1_W_HTLoss', 'T2_CUM_T2_W_HTLoss', 'T1_CUM_T1_Draw',
                          'T2_CUM_T2_Draw', 'T1_CUM_T1_W_Draw', 'T2_CUM_T2_W_Draw', 'T1_CUM_T1_HTDraw', 'T2_CUM_T2_HTDraw',
                          'T1_CUM_T1_W_HTDraw', 'T2_CUM_T2_W_HTDraw']
        feature_names += ["T1_spi", "T2_spi", "T1_imp", "T2_imp", "T1_GFTe", "T2_GFTe", "pp1", "pp0", "pp2"]
        feature_names += ["T1_CUM_T1_GFTa", "T2_CUM_T2_GFTa", "T1_CUM_T1_W_GFTa", "T2_CUM_T2_W_GFTa", "T1_CUM_T2_GFTa",
                          "T2_CUM_T1_GFTa", "T1_CUM_T2_W_GFTa", "T2_CUM_T1_W_GFTa", "T12_CUM_T1_GFTa", "T12_CUM_T1_W_GFTa",
                          "T21_CUM_T2_GFTa", "T21_CUM_T2_W_GFTa", "T12_CUM_T12_GFTa", "T12_CUM_T12_W_GFTa",
                          "T1221_CUM_GFTa", "T1221_CUM_W_GFTa", "T1_CUM_T1_xsg", "T2_CUM_T2_xsg", "T1_CUM_T1_W_xsg",
                          "T2_CUM_T2_W_xsg", "T1_CUM_T2_xsg", "T2_CUM_T1_xsg", "T1_CUM_T2_W_xsg", "T2_CUM_T1_W_xsg",
                          "T12_CUM_T1_xsg", "T12_CUM_T1_W_xsg", "T21_CUM_T2_xsg", "T21_CUM_T2_W_xsg", "T12_CUM_T12_xsg",
                          "T12_CUM_T12_W_xsg", "T1221_CUM_xsg", "T1221_CUM_W_xsg", "T1_CUM_T1_xnsg", "T2_CUM_T2_xnsg",
                          "T1_CUM_T1_W_xnsg", "T2_CUM_T2_W_xnsg", "T1_CUM_T2_xnsg", "T2_CUM_T1_xnsg", "T1_CUM_T2_W_xnsg",
                          "T2_CUM_T1_W_xnsg", "T12_CUM_T1_xnsg", "T12_CUM_T1_W_xnsg", "T21_CUM_T2_xnsg",
                          "T21_CUM_T2_W_xnsg", "T12_CUM_T12_xnsg", "T12_CUM_T12_W_  xnsg", "T1221_CUM_xnsg",
                          "T1221_CUM_W_xnsg", "T1_CUM_T1_spi", "T2_CUM_T2_spi", "T1_CUM_T1_W_spi", "T2_CUM_T2_W_spi",
                          "T1_CUM_T2_spi", "T2_CUM_T1_spi", "T1_CUM_T2_W_spi", "T2_CUM_T1_W_spi", "T12_CUM_T1_spi",
                          "T12_CUM_T1_W_spi", "T21_CUM_T2_spi", "T21_CUM_T2_W_spi", "T12_CUM_T12_spi", "T12_CUM_T12_W_spi",
                          "T1221_CUM_spi", "T1221_CUM_W_spi", "T1_CUM_T1_imp", "T2_CUM_T2_imp", "T1_CUM_T1_W_imp",
                          "T2_CUM_T2_W_imp", "T1_CUM_T2_imp", "T2_CUM_T1_imp", "T1_CUM_T2_W_imp", "T2_CUM_T1_W_imp",
                          "T12_CUM_T1_imp", "T12_CUM_T1_W_imp", "T21_CUM_T2_imp", "T21_CUM_T2_W_imp", "T12_CUM_T12_imp",
                          "T12_CUM_T12_W_imp", "T1221_CUM_imp", "T1221_CUM_W_imp", "T1_CUM_T1_GFTe", "T2_CUM_T2_GFTe",
                          "T1_CUM_T1_W_GFTe", "T2_CUM_T2_W_GFTe", "T1_CUM_T2_GFTe", "T2_CUM_T1_GFTe", "T1_CUM_T2_W_GFTe",
                          "T2_CUM_T1_W_GFTe", "T12_CUM_T1_GFTe", "T12_CUM_T1_W_GFTe", "T21_CUM_T2_GFTe",
                          "T21_CUM_T2_W_GFTe", "T12_CUM_T12_GFTe", "T12_CUM_T12_W_GFTe", "T1221_CUM_GFTe",
                          "T1221_CUM_W_GFTe"]




        # reduced features
        feature_names = ['t1histlen', 'where', 't2histlen', 't12histlen']
        feature_names += ['Time', 't1games', 't1dayssince', 't2dayssince', 't1dayssince_ema', 't2dayssince_ema',
                          'roundsleft', 't1promoted', 't2promoted', 't1points', 't2points', 't1rank', 't2rank',
                          't1rank6_attention', 't2rank6_attention', 't1rank16_attention', 't2rank16_attention',
                          't1cards_ema', 't2cards_ema', 'BW1', 'BW0', 'BW2']
                          # exponential moving average
        feature_names += ['t1goals_ema', 't2goals_ema', 'T1_EMA_GFT', 'T2_EMA_GFT', 'T1_EMA_GHT', 'T2_EMA_GHT', 'T1_EMA_ST', 'T2_EMA_ST',
                         'T1_EMA_S', 'T2_EMA_S', 'T1_EMA_C', 'T2_EMA_C', 'T1_EMA_F', 'T2_EMA_F', 'T1_EMA_Y', 'T2_EMA_Y', 'T1_EMA_R', 'T2_EMA_R',
                         'T1_EMA_xG', 'T2_EMA_xG', 'T1_EMA_GFTa', 'T2_EMA_GFTa', 'T1_EMA_xsg', 'T2_EMA_xsg', 'T1_EMA_xnsg', 'T2_EMA_xnsg',
                         'T1_EMA_spi', 'T2_EMA_spi', 'T1_EMA_imp', 'T2_EMA_imp', 'T1_EMA_GFTe', 'T2_EMA_GFTe', 'T1_EMA_GH2', 'T2_EMA_GH2']
        # last 10 rounds across all teams - general distribution
        feature_names += ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
                          'L10_T1_GFT', 'L10_T2_GFT', 'L10_T1_GHT', 'L10_T2_GHT', 'L10_T1_S', 'L10_T2_S', 'L10_T1_ST', 'L10_T2_ST',
                          'L10_T1_F', 'L10_T2_F', 'L10_T1_C', 'L10_T2_C', 'L10_T1_Y', 'L10_T2_Y', 'L10_T1_R', 'L10_T2_R',
                          'L10_T1_xG', 'L10_T2_xG', 'L10_T1_GFTa', 'L10_T2_GFTa', 'L10_T1_xsg', 'L10_T2_xsg', 'L10_T1_xnsg', 'L10_T2_xnsg',
                          'L10_T1_GH2', 'L10_T2_GH2', 'L10_Loss', 'L10_Draw', 'L10_Win', 'L10_HTLoss', 'L10_HTDraw', 'L10_HTWin',
                          'L10_HT2Loss', 'L10_HT2Draw', 'L10_HT2Win',
                          'L10_zScore0:3', 'L10_zScore1:3', 'L10_zScore0:2', 'L10_zScore1:2', 'L10_zScore0:1', 'L10_zScore0:0', 'L10_zScore1:1',
                          'L10_zScore1:0', 'L10_zScore2:1', 'L10_zScore2:0', 'L10_zScore3:1', 'L10_zScore3:0', 'L10_zScore2:2', 'L10_zScore3:2',
                          'L10_zScore2:3', 'L10_zScore3:3', 'L10_FTG4', 'L10_FTG0', 'L10_HTG0']

        #feature_names += label_column_names_extended

        print(X.shape)

        #  train_idx = train_idx[::2]
        #  test_idx = test_idx[::2]
        #  pred_idx = pred_idx[::2]
        #train_idx = train_idx[::306]

        # reset "Where"
        #X[:,1]=0.5

        X_train = X[train_idx]
        X_test = X[test_idx]
        X_pred = X[pred_idx]
        bwin_index = [1, 23, 24, 25]
        X_train_bwin = np.take(X[train_idx], bwin_index, axis=1)
        X_test_bwin = np.take(X[test_idx], bwin_index, axis=1)
        X_pred_bwin = np.take(X[pred_idx], bwin_index, axis=1)

        # spi_index = [210, 211, 212, 213, 214, 215, 216, 217, 218]
        # spi_index = [1, 214, 215, 216, 217, 218]
        # X_train_spi = np.take(X[train_idx], spi_index, axis=1)
        # X_test_spi = np.take(X[test_idx], spi_index, axis=1)
        # X_pred_spi = np.take(X[pred_idx], spi_index, axis=1)

        #tf.config.set_visible_devices([], 'GPU')
        #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))


        if tf.test.gpu_device_name() != '/device:GPU:0':
            print('WARNING: GPU device not found.')
        else:
            print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

        # For numeric stability, set the default floating-point dtype to float64
        tf.keras.backend.set_floatx('float64')


        #feature_idx = list(range(0,30))
        #feature_idx = bwin_index
        feature_idx = range(X_train.shape[1])
        label_columns_poisson = [i for i in range(46)]
        label_columns_gaussian = [i for i in range(46, 54)]
        #Y_train = labels_array[train_idx, 0:16]
        Y_train = labels_array[train_idx]
        Y_test = labels_array[test_idx]
        Y_pred = labels_array[pred_idx]

        x_train = X_train[:, feature_idx].astype(np.float64)
        x_test = X_test[:, feature_idx].astype(np.float64)
        x_pred = X_pred[:, feature_idx].astype(np.float64)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        x_pred_scaled = scaler.transform(x_pred)
        #scaler.inverse_transform(small_x_train)

        def sample_df(yhat, Y, dataset):
            sample = yhat.sample().numpy()
            l = sample.shape[0]//Y.shape[0]
            df = pd.DataFrame({"Where":np.tile(np.tile(["Home", "Away"], reps=Y[:,0].shape[0]//2), reps=l),
                               "GS":np.tile(Y[:,0].astype(int), reps=l),
                               "GC":np.tile(Y[:,1].astype(int), reps=l),
                               "pGS":sample[:,0].astype(int),
                               "pGC":sample[:,1].astype(int),
                               "est1":yhat.mean()[:,0],
                               "est2":yhat.mean()[:,1],
                                   # "pGS":np.argmax(sample, axis=1)//7,
                               # "pGC":np.mod(np.argmax(sample, axis=1), 7),
                               # "est1":np.sum(yhat.mean() * arraygs, axis=1),
                               # "est2":np.sum(yhat.mean() * arraygc, axis=1),
                               "Prefix": "poisson",
                               "dataset": dataset,
                               "act":"Y_test[:,0:2]",
                               "Team1":"Team1",
                               "Team2": "Team2",
                               "match":np.tile(range(Y[:,0].shape[0]), reps=l)
                               })
            return df

        def sample_categorical_df(yhat, Y, dataset):
            sample = yhat.sample().numpy()
            l = sample.shape[0]//Y.shape[0]
            probs = yhat.mean()
            probs = probs / np.sum(probs, axis=1,  keepdims=True)
            df = pd.DataFrame({"Where":np.tile(np.tile(["Home", "Away"], reps=Y[:,0].shape[0]//2), reps=l),
                               "GS":np.tile(Y[:,0].astype(int), reps=l),
                               "GC":np.tile(Y[:,1].astype(int), reps=l),
                               # "pGS":sample//7,
                               # "pGC":np.mod(sample, 7),
                               # "est1":yhat.mean()[:,0],
                               # "est2":yhat.mean()[:,1],
                               "pGS":np.argmax(sample, axis=1)//7,
                               "pGC":np.mod(np.argmax(sample, axis=1), 7),
                               "est1":np.sum(probs * arraygs, axis=1),
                               "est2":np.sum(probs * arraygc, axis=1),
                               "Prefix": "poisson",
                               "dataset": dataset,
                               "act":"Y_test[:,0:2]",
                               "Team1":"Team1",
                               "Team2": "Team2",
                               "match":np.tile(range(Y[:,0].shape[0]), reps=l)
                               })
            return df

        def create_result_index(GS, GC):
            return (np.maximum(0, np.minimum(np.round(GS), 6)) * 7 + np.maximum(0, np.minimum(np.round(GC), 6))).astype(np.int32)

        def laplacian_matrix():
            m = [[-1 if abs(i - i2 + j - j2) == 1 and abs(i - i2 - j + j2) == 1 else \
                      2 if (i, j) == (i2, j2) and (i, j) in [(0, 0), (0, 6), (6, 0), (6, 6)] else \
                          3 if (i, j) == (i2, j2) and (i in [0, 6] or j in [0, 6]) else \
                              4 if (i, j) == (i2, j2) else \
                                  0
                  for i in range(7) for j in range(7)] for i2 in range(7) for j2 in range(7)]
            t_laplacian = tf.constant(m, dtype=tf.float64)
            return t_laplacian

        def create_laplacian_loss(p_pred_12, alpha=1.0):
            laplm = laplacian_matrix()
            lp = tf.matmul(p_pred_12, laplm)
            laplacian_loss = (lp ** 2) / 2
            laplacian_loss = tf.reduce_sum(laplacian_loss, axis=1)
            laplacian_loss = alpha * tf.reduce_mean(laplacian_loss, name="laplacian")
            return laplacian_loss


        arraygs = np.array([i // 7 for i in range(49)])[np.newaxis, ...]
        arraygc = np.array([np.mod(i, 7) for i in range(49)])[np.newaxis, ...]
        outputs_ = tf.one_hot(create_result_index(Y_train[:,0], Y_train[:,1]), 49) #.astype(np.float64)
        test_outputs_ = tf.one_hot(create_result_index(Y_test[:, 0], Y_test[:, 1]), 49) #.astype(np.float64)
        empHomeDist = (1.0 + np.sum(outputs_[1::2], axis=0, dtype=np.float64))#*0.1
        empAwayDist = (1.0 + np.sum(outputs_[0::2], axis=0, dtype=np.float64))#*0.1
        #outputs_ = create_result_index(Y_train[:,0], Y_train[:,1]) #.astype(np.float64)
        #test_outputs_ = create_result_index(Y_test[:, 0], Y_test[:, 1]) #.astype(np.float64)

        #outputs_ = tf.matmul(outputs_, tf.cast(point_matrix, tf.float32))
        #test_outputs_ = tf.matmul(test_outputs_, tf.cast(point_matrix, tf.float32))

        print(empHomeDist)
        print(empAwayDist)
        print(np.sum(empHomeDist))

        def create_maxpoint_prediction(df):
            dfpoints = np.stack([point_matrix[create_result_index(df.pGS, df.pGC), i] for i in range(49)],
                                axis=1)
            dfpoints = pd.DataFrame(dfpoints, index=df.match)
            dfpoints = dfpoints.groupby(dfpoints.index).mean()
            maxpoints = pd.DataFrame({"pGS": np.argmax(dfpoints.to_numpy(), axis=1) // 7,
                                      "pGC": np.mod(np.argmax(dfpoints.to_numpy(), axis=1), 7),
                                      "points": np.amax(dfpoints.to_numpy(), axis=1)})
            df2 = df. \
                groupby(['match', 'Where', 'GS', 'GC', 'Prefix', 'dataset', 'act', 'Team1', 'Team2']).mean(). \
                rename(columns={"pGS": "pGS2", "pGC": "pGC2"}). \
                reset_index().set_index("match")
            df3 = pd.concat([df2, maxpoints], axis=1)
            return df3

        def create_maxpoint_prediction_from_mean(probs, Ylen):
            l = probs.shape[0]//Ylen
            expPoints = np.mean(np.matmul(probs, point_matrix).reshape((-1, Ylen,49)), axis=0)
            maxpoints = pd.DataFrame({"pGS": np.argmax(expPoints, axis=1) // 7,
                                      "pGC": np.mod(np.argmax(expPoints, axis=1), 7),
                                      "points": np.amax(expPoints, axis=1),
                                      "est1": np.mean(np.sum(probs * arraygs, axis=1).reshape((-1, Ylen)), axis=0),
                                      "est2": np.mean(np.sum(probs * arraygc, axis=1).reshape((-1, Ylen)), axis=0)
                                      })
            return maxpoints

        def create_argmax_prediction_from_mean(probs, Ylen):
            probs = probs.reshape((-1, Ylen,49))
            expPoints = np.mean(np.matmul(probs, point_matrix), axis=-3) # shape: w*batch, 49
            maxpoints = pd.DataFrame({"pGS": np.argmax(np.mean(probs, axis=-3), axis=-1) // 7,
                                      "pGC": np.mod(np.argmax(np.mean(probs, axis=-3), axis=-1), 7),
                                      "points": np.amax(expPoints, axis=-1),
                                      "est1": np.mean(np.sum(probs * arraygs, axis=-1).reshape((-1, Ylen)), axis=0), #
                                      "est2": np.mean(np.sum(probs * arraygc, axis=-1).reshape((-1, Ylen)), axis=0)
                                      })
            return maxpoints

        def create_df(maxpoints, Y, dataset, team1=None, team2=None):
            l = maxpoints.shape[0]//Y.shape[0]
            df = pd.DataFrame({"Where":np.tile(np.tile(["Home", "Away"], reps=Y[:,0].shape[0]//2), reps=l),
                               "GS":np.tile(Y[:,0].astype(int), reps=l),
                               "GC":np.tile(Y[:,1].astype(int), reps=l),
                               "pGS":maxpoints.pGS,
                               "pGC":maxpoints.pGC,
                               "est1":maxpoints.est1,
                               "est2":maxpoints.est2,
                               "points":maxpoints.points,
                               "Prefix": "poisson",
                               "dataset": dataset,
                               "act":"act",
                               "Team1":team1,
                               "Team2": team2,
                               "match":np.tile(range(Y[:,0].shape[0]), reps=l)
                               })
            return df


        #plot_gaussian_process(searchlog, 0, 6)
        def plot_gaussian_process(searchlog, xlabel, ylabel):
            #observations_ = searchlog.pts
            observations_ = -searchlog.score

            # remove lower end outliers (ususally bad scores)
            lower_bound = np.quantile(observations_, 0.9) - 5*(np.quantile(observations_, 0.9)-np.quantile(observations_, 0.35))
            searchlog = searchlog[observations_.gt(lower_bound)]
            observations_ = observations_[observations_.gt(lower_bound)]

            # center the observations so that the best point is positive
            med = np.median(observations_)
            print("Median:")
            print(med)
            observations_ = observations_ - med
            print(observations_)
            print(searchlog)
            observation_index_points_ = searchlog.loc[:,
                                        ["reg1", "reg2", "reg2g", "reg3", "reg4", "wkl", "wlpt", "wpois", "wgaus", "wsmx"]].to_numpy()
            tfk = tfp.math.psd_kernels
            k = observation_index_points_.shape[1]
            observation_index_points_ = np.log(observation_index_points_)

            scaler = preprocessing.StandardScaler().fit(observation_index_points_)
            observation_index_points_ = scaler.transform(observation_index_points_)

            last_score = observations_[searchlog.accept.astype(bool)].iloc[-1]
            current_point = observation_index_points_[searchlog.accept.astype(bool)][-1]

            #plotdata = searchlog.loc[:, [xlabel, ylabel]]

            def build_gp(amplitude, length_scale, scale_diag, observation_noise_variance):
                """Defines the conditional dist. of GP outputs, given kernel parameters."""

                # Create the covariance kernel, which will be shared between the prior (which we
                # use for maximum likelihood training) and the posterior (which we use for
                # posterior predictive sampling)
                #kernel = tfk.MaternOneHalf(amplitude, length_scale, feature_ndims=1)
                #kernel = tfk.MaternThreeHalves(amplitude, length_scale, feature_ndims=1)

                #kernel = tfk.MaternFiveHalves(amplitude, length_scale, feature_ndims=1)
                kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale, feature_ndims=1)
                kernel = tfk.FeatureScaled(kernel, scale_diag)
                # Create the GP prior distribution, which we will use to train the model
                # parameters.
                gp = tfd.GaussianProcess(
                    kernel=kernel,
                    index_points=observation_index_points_,
                    observation_noise_variance=observation_noise_variance)
                return gp

            gp_joint_model = tfd.JointDistributionNamed({
                'amplitude':  tfd.LogNormal(loc=0., scale=np.float64(1.)),
                'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
                'scale_diag': tfd.Independent(tfd.LogNormal(loc=[0.] * k, scale=np.float64([1.] * k)),
                                              reinterpreted_batch_ndims=1),
                'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
                'observations': build_gp,
            })
            # Create the trainable model parameters, which we'll subsequently optimize.
            # Note that we constrain them to be strictly positive.
            constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

            amplitude_var = tfp.util.TransformedVariable(
                initial_value=1.,
                bijector=constrain_positive,
                name='amplitude',
                dtype=np.float64)

            length_scale_var = tfp.util.TransformedVariable(
                initial_value=1.,
                bijector=constrain_positive,
                name='length_scale',
                dtype=np.float64)

            scale_diag_var = tfp.util.TransformedVariable(
                initial_value=[1.] * k,
                bijector=constrain_positive,
                name='amplitude',
                dtype=np.float64)

            observation_noise_variance_var = tfp.util.TransformedVariable(
                initial_value=1.,
                bijector=constrain_positive,
                name='observation_noise_variance_var',
                dtype=np.float64)

            trainable_variables = [v.trainable_variables[0] for v in
                                   [amplitude_var,
                                    length_scale_var,
                                    scale_diag_var,
                                    observation_noise_variance_var]]
            # Use `tf.function` to trace the loss for more efficient evaluation.
            @tf.function(autograph=False, experimental_compile=False)
            def target_log_prob(amplitude, length_scale, scale_diag, observation_noise_variance):
                return gp_joint_model.log_prob({
                    'amplitude': amplitude,
                    'length_scale': length_scale,
                    'scale_diag': scale_diag,
                    'observation_noise_variance': observation_noise_variance,
                    'observations': observations_
                })

            # Now we optimize the model parameters.
            num_iters = 100
            optimizer2 = tf.optimizers.Adam(learning_rate=.1)

            # Store the likelihood values during training, so we can plot the progress
            lls_ = np.zeros(num_iters, np.float64)
            for i in range(num_iters):
                with tf.GradientTape() as tape:
                    loss = -target_log_prob(amplitude_var, length_scale_var, scale_diag_var,
                                            observation_noise_variance_var)
                grads = tape.gradient(loss, trainable_variables)
                optimizer2.apply_gradients(zip(grads, trainable_variables))
                lls_[i] = loss

            print('Trained parameters:')
            print('amplitude: {}'.format(amplitude_var._value().numpy()))
            print('length_scale: {}'.format(length_scale_var._value().numpy()))
            print('scale_diag: {}'.format(scale_diag_var._value().numpy()))
            print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

            #optimized_kernel = tfk.MaternFiveHalves(amplitude_var, length_scale_var, feature_ndims=1)
            #optimized_kernel = tfk.MaternOneHalf(amplitude_var, length_scale_var, feature_ndims=1)
            #optimized_kernel = tfk.MaternThreeHalves(amplitude_var, length_scale_var, feature_ndims=1)
            optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var, feature_ndims=1)
            optimized_kernel = tfk.FeatureScaled(optimized_kernel, scale_diag_var)

            @tf.function(autograph=False, experimental_compile=False)
            def expected_improvement(index_var):
                newgprm = tfd.GaussianProcessRegressionModel(
                    kernel=optimized_kernel,
                    index_points=index_var, #[tf.newaxis, ...],
                    observation_index_points=observation_index_points_,
                    observations=observations_,
                    observation_noise_variance=observation_noise_variance_var,
                    predictive_noise_variance=0.)
                Z = (newgprm.mean() - last_score) / newgprm.stddev()
                EI = (newgprm.mean() - last_score) * nn.cdf(Z) + newgprm.stddev() * nn.prob(Z)
                return EI,newgprm.mean() , newgprm.stddev()


            # determine the index_point with the highest expected improvement using gradient descent
            #best = np.max(observations_)
            nn = tfd.Normal(tf.zeros(1, dtype=tf.float64), tf.ones(1, dtype=tf.float64))

            current_point = observation_index_points_[searchlog.accept.astype(bool)][-1]
            print("current_point:")
            print(current_point)

            print("norms:")
            print(np.linalg.norm(observation_index_points_-current_point[np.newaxis, ...], ord=2, axis=1))

            xlabel = 0
            ylabel = 6
            #plotdata = searchlog.loc[:, [xlabel, ylabel]]

            xdata = observation_index_points_[:,xlabel]
            ydata = observation_index_points_[:, ylabel]
            k = len(observation_index_points_[:,xlabel])
            pointdata = np.concatenate([xdata[...,np.newaxis],
                                       current_point[1:6]*np.ones((k,5)),
                                       ydata[...,np.newaxis],
                                       current_point[7:]*np.ones((k,3))], axis=-1)

            obsei = expected_improvement(observation_index_points_)
            obsei2 = expected_improvement(pointdata)
            print(pd.DataFrame({"actual":observations_, "pred":obsei[1], "stddev":obsei[2], "ei":obsei[0], "distnorm":np.linalg.norm(observation_index_points_-current_point[np.newaxis, ...], ord=2, axis=1),
                               "x":xdata, "y":ydata, "p_pred":obsei2[1], "p_stddev":obsei2[2], "p_ei":obsei2[0], "p_distnorm":np.linalg.norm(pointdata-current_point[np.newaxis, ...], ord=2, axis=1)}))

            y = np.arange(ydata.min(), ydata.max(), 0.01*(ydata.max()-ydata.min()))
            x = np.arange(xdata.min(), xdata.max(), 0.01*(xdata.max()-xdata.min()))
            X, Y = np.meshgrid(x, y)
            #print(X)
            #print(Y)
            griddata = np.concatenate([X[...,np.newaxis],
                                       current_point[1:6]*np.ones((100,100,5)),
                                       Y[...,np.newaxis],
                                       current_point[7:]*np.ones((100,100,3))], axis=-1)
            Z0, Z, Zstd = expected_improvement(griddata)

            fig, ax = plt.subplots()
            #plt.imshow(Z)
            CS = ax.contour(X, Y, Z0, cmap="RdYlGn", levels=12)
            ax.clabel(CS, inline=1, fontsize=10)
            #cset1 = ax.contourf(X, Y, Z, cmap="RdYlGn_r", levels=6)
            #ax.scatter(xdata[~searchlog.accept.astype(bool)], ydata[~searchlog.accept.astype(bool)], color="blue")
            #ax.scatter(xdata[searchlog.accept.astype(bool)], ydata[searchlog.accept.astype(bool)], color="red")
            ax.plot(xdata, ydata)
            sns.scatterplot(xdata, ydata, hue=observations_, palette="RdYlGn")
            #plt.colorbar()
            for i, obs in enumerate(observations_):
                ax.annotate("{:.3f}".format(obs), (xdata[i]+0.01, ydata[i]+0.01))
            plt.show()

            fig, ax = plt.subplots()
            plt.title("Expected Improvement")
            im = plt.imshow(Z0)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            plt.title("Means")
            im = plt.imshow(Z)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            plt.title("Standard deviations")
            im = plt.imshow(Zstd)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()

            xlabel = 1
            ylabel = 7
            #plotdata = searchlog.loc[:, [xlabel, ylabel]]
            xdata = observation_index_points_[:,xlabel]
            ydata = observation_index_points_[:, ylabel]

            obsei = expected_improvement(observation_index_points_)
            print(pd.DataFrame({"actual":observations_, "pred":obsei[1], "stddev":obsei[2], "ei":obsei[0], "x":xdata, "y":ydata}))

            y = np.arange(ydata.min(), ydata.max(), 0.01*(ydata.max()-ydata.min()))
            x = np.arange(xdata.min(), xdata.max(), 0.01*(xdata.max()-xdata.min()))
            x=x[0:100]
            y=y[0:100]
            X, Y = np.meshgrid(x, y)
            #print(X)
            #print(Y)
            griddata = np.concatenate([current_point[0:1]*np.ones((100,100,1)),
                                       X[...,np.newaxis],
                                       current_point[2:7]*np.ones((100,100,5)),
                                       Y[...,np.newaxis],
                                       current_point[8:]*np.ones((100,100,2))], axis=-1)
            Z0, Z, Zstd = expected_improvement(griddata)

            fig, ax = plt.subplots()
            #plt.imshow(Z)
            CS = ax.contour(X, Y, Z0, cmap="RdYlGn", levels=12)
            ax.clabel(CS, inline=1, fontsize=10)
            #cset1 = ax.contourf(X, Y, Z, cmap="RdYlGn_r", levels=6)
            #ax.scatter(xdata[~searchlog.accept.astype(bool)], ydata[~searchlog.accept.astype(bool)], color="blue")
            #ax.scatter(xdata[searchlog.accept.astype(bool)], ydata[searchlog.accept.astype(bool)], color="red")
            ax.plot(xdata, ydata)
            sns.scatterplot(xdata, ydata, hue=observations_, palette="RdYlGn")
            #plt.colorbar()
            for i, obs in enumerate(observations_):
                ax.annotate("{:.3f}".format(obs), (xdata[i]+0.01, ydata[i]+0.01))
            plt.show()

            fig, ax = plt.subplots()
            plt.title("Expected Improvement")
            im = plt.imshow(Z0)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            plt.title("Means")
            im = plt.imshow(Z)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            plt.title("Standard deviations")
            im = plt.imshow(Zstd)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()


        #find_next_params_from_gaussian_process(searchlog, num_iters3=1000)
        def find_next_params_from_gaussian_process(searchlog, num_iters3=1000):
            #observations_ = searchlog.pts
            observations_ = -searchlog.score

            # remove lower end outliers (ususally bad scores)
            lower_bound = np.quantile(observations_, 0.9) - 5*(np.quantile(observations_, 0.9)-np.quantile(observations_, 0.35))
            searchlog = searchlog[observations_.gt(lower_bound)]
            observations_ = observations_[observations_.gt(lower_bound)]

            # center the observations so that the best point is positive
            med = np.median(observations_)
            print("Median:")
            print(med)
            observations_ = observations_ - med

            observation_index_points_ = searchlog.loc[:,
                                        ["reg1", "reg2", "reg2g", "reg3", "reg4", "wkl", "wlpt", "wpois", "wgaus", "wsmx"]].to_numpy()
            #print(current_point)
            tfk = tfp.math.psd_kernels
            k = observation_index_points_.shape[1]
            observation_index_points_ = np.log(observation_index_points_)
            last_score = observations_[searchlog.accept.astype(bool)].iloc[-1]

            scaler = preprocessing.StandardScaler().fit(observation_index_points_)
            observation_index_points_ = scaler.transform(observation_index_points_)

            current_point = observation_index_points_[searchlog.accept.astype(bool)][-1]
            print("Last score: "+str(last_score))
            print("Current point: "+str(current_point))
            # if np.random.random() < 0.15:
            #     current_point = [m * np.random.lognormal(0, 0.15) for m in current_point] # introduce some randomness into the search

            #mm_scaler = preprocessing.MinMaxScaler()
            #observation_index_points_ = mm_scaler.fit_transform(np.log(observation_index_points_))

            def build_gp(amplitude, length_scale, scale_diag, observation_noise_variance):
                """Defines the conditional dist. of GP outputs, given kernel parameters."""

                # Create the covariance kernel, which will be shared between the prior (which we
                # use for maximum likelihood training) and the posterior (which we use for
                # posterior predictive sampling)
                #kernel = tfk.MaternOneHalf(amplitude, length_scale, feature_ndims=1)
                kernel = tfk.MaternThreeHalves(amplitude, length_scale, feature_ndims=1)
                #kernel = tfk.MaternFiveHalves(amplitude, length_scale, feature_ndims=1)
                #kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale, feature_ndims=1)
                kernel = tfk.FeatureScaled(kernel, scale_diag)
                # Create the GP prior distribution, which we will use to train the model
                # parameters.
                gp = tfd.GaussianProcess(
                    kernel=kernel,
                    index_points=observation_index_points_,
                    observation_noise_variance=observation_noise_variance)
                return gp

            gp_joint_model = tfd.JointDistributionNamed({
                'amplitude':  tfd.LogNormal(loc=0., scale=np.float64(1.)),
                'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
                'scale_diag': tfd.Independent(tfd.LogNormal(loc=[0.] * k, scale=np.float64([1.] * k)),
                                              reinterpreted_batch_ndims=1),
                'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
                'observations': build_gp,
            })
            # Create the trainable model parameters, which we'll subsequently optimize.
            # Note that we constrain them to be strictly positive.
            constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

            amplitude_var = tfp.util.TransformedVariable(
                initial_value=1.,
                bijector=constrain_positive,
                name='amplitude',
                dtype=np.float64)

            length_scale_var = tfp.util.TransformedVariable(
                initial_value=1.,
                bijector=constrain_positive,
                name='length_scale',
                dtype=np.float64)

            scale_diag_var = tfp.util.TransformedVariable(
                initial_value=[1.] * k,
                bijector=constrain_positive,
                name='amplitude',
                dtype=np.float64)

            observation_noise_variance_var = tfp.util.TransformedVariable(
                initial_value=1.,
                bijector=constrain_positive,
                name='observation_noise_variance_var',
                dtype=np.float64)

            trainable_variables = [v.trainable_variables[0] for v in
                                   [amplitude_var,
                                    length_scale_var,
                                    scale_diag_var,
                                    observation_noise_variance_var]]
            # Use `tf.function` to trace the loss for more efficient evaluation.
            @tf.function(autograph=False, experimental_compile=False)
            def target_log_prob(amplitude, length_scale, scale_diag, observation_noise_variance):
                return gp_joint_model.log_prob({
                    'amplitude': amplitude,
                    'length_scale': length_scale,
                    'scale_diag': scale_diag,
                    'observation_noise_variance': observation_noise_variance,
                    'observations': observations_
                })

            # Now we optimize the model parameters.
            num_iters = 100
            optimizer2 = tf.optimizers.Adam(learning_rate=.1)

            # Store the likelihood values during training, so we can plot the progress
            lls_ = np.zeros(num_iters, np.float64)
            for i in range(num_iters):
                with tf.GradientTape() as tape:
                    loss = -target_log_prob(amplitude_var, length_scale_var, scale_diag_var,
                                            observation_noise_variance_var)
                grads = tape.gradient(loss, trainable_variables)
                optimizer2.apply_gradients(zip(grads, trainable_variables))
                lls_[i] = loss

            print('Trained parameters:')
            print('amplitude: {}'.format(amplitude_var._value().numpy()))
            print('length_scale: {}'.format(length_scale_var._value().numpy()))
            print('scale_diag: {}'.format(scale_diag_var._value().numpy()))
            print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

            #optimized_kernel = tfk.MaternFiveHalves(amplitude_var, length_scale_var, feature_ndims=1)
            #optimized_kernel = tfk.MaternOneHalf(amplitude_var, length_scale_var, feature_ndims=1)
            optimized_kernel = tfk.MaternThreeHalves(amplitude_var, length_scale_var, feature_ndims=1)
            #optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var, feature_ndims=1)
            optimized_kernel = tfk.FeatureScaled(optimized_kernel, scale_diag_var)

            # determine the index_point with the highest expected improvement using gradient descent
            #best = np.max(observations_)
            nn = tfd.Normal(tf.zeros(1, dtype=tf.float64), tf.ones(1, dtype=tf.float64))

            reg_min=0.0001
            reg_max=1000.0
            wgt_min=0.1
            wgt_max=1000000
            lower_limit=np.log(np.array([reg_min]*5+[wgt_min]*5))
            upper_limit=np.log(np.array([reg_max]*5+[wgt_max]*5))
            limit_range=upper_limit-lower_limit
            b = 5 # batch_size
            best_index_var = tfp.util.TransformedVariable(
                initial_value=tf.random_uniform_initializer(minval=lower_limit+0.2*limit_range, maxval=upper_limit-0.2*limit_range)(shape=[b,k], dtype=tf.float64),
                name="best_index_var", dtype=np.float64,
                bijector=tfb.SoftClip(low=tf.constant(lower_limit, dtype=tf.float64), high=tf.constant(upper_limit, dtype=tf.float64)))

            #prior_dist = tfd.Independent(tfd.Normal(loc=current_point, scale=1.5), reinterpreted_batch_ndims=1) # prevent movements far away from origin
            @tf.function(autograph=False, experimental_compile=False)
            def prior_log_prob(index_var):
                prior = tfd.Normal(loc=tf.constant(0.0, dtype=tf.float64), scale=tf.constant(1.0, dtype=tf.float64))
                distance = tf.norm(index_var - current_point[tf.newaxis, ...], axis=-1)
                return prior.log_prob(tf.math.log(distance))

            # best_index_var = tfp.util.TransformedVariable(
            #     initial_value=tf.random_uniform_initializer(minval=0.2, maxval=0.8)(shape=[k], dtype=tf.float64),
            #     name="best_index_var", dtype=np.float64,
            #     bijector=tfb.SoftClip(low=tf.constant(-0.1, dtype=tf.float64), high=tf.constant(1.1, dtype=tf.float64)))

            @tf.function(autograph=False, experimental_compile=False)
            def expected_improvement(index_var):
                newgprm = tfd.GaussianProcessRegressionModel(
                    kernel=optimized_kernel,
                    index_points=index_var, #[tf.newaxis, ...],
                    observation_index_points=observation_index_points_,
                    observations=observations_,
                    observation_noise_variance=observation_noise_variance_var,
                    predictive_noise_variance=0.)
                Z = (newgprm.mean() - last_score) / newgprm.stddev()
                EI = (newgprm.mean() - last_score) * nn.cdf(Z) + newgprm.stddev() * nn.prob(Z)
                return EI, newgprm.mean(), newgprm.stddev(), Z


            obsei = expected_improvement(observation_index_points_)
            obsdist = prior_log_prob(observation_index_points_).numpy()

            #print(obsei)
            #obsdist = prior_dist.log_prob(observation_index_points_)
            print(pd.DataFrame({"actual":observations_, "pred":obsei[1], "ei":obsei[0], "dist":obsdist}))
            # Now we search the best index point.
            #num_iters3 = 100

            optimizer3 = tf.optimizers.Adam(learning_rate=.02)

            # Store the likelihood values during training, so we can plot the progress
            #print(searchlog)
            # print(last_score)
            print("New search point base:")
            print(current_point)
            print({"expected_improvement":[v.numpy() for v in expected_improvement(np.array(current_point)[tf.newaxis, ...])],
                   "distance log prob": prior_log_prob(np.array(current_point)[tf.newaxis, ...]).numpy()})
            #stop()
            lls_ = np.zeros((num_iters3, b), np.float64)
            early_stop_rounds = 20
            minloss = 1e20
            rounds_without_improvement = 0
            for i in range(num_iters3):
                with tf.GradientTape() as tape:
                    ei = expected_improvement(best_index_var)[0]
                    pp = prior_log_prob(best_index_var)
                    loss = -ei - pp #-ei#+0.0001*tf.reduce_sum(tf.math.reduce_variance(tf.concat([observation_index_points_, best_index_var], axis=0), axis=0))
                    # if np.mod(i,50) == 0:
                    #     print((i, ei, pp))
                    #     print(best_index_var.trainable_variables[0])
                grads = tape.gradient(loss, [best_index_var.trainable_variables[0]])
                optimizer3.apply_gradients(zip(grads, [best_index_var.trainable_variables[0]]))
                lls_[i] = loss
                if np.min(loss) < minloss:
                    minloss = np.min(loss)
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement > early_stop_rounds:
                        break
            print("Optimization Rounds:")
            print(i)
            print(best_index_var)
            print("Expected Improvement:")
            print([v.numpy() for v in expected_improvement(best_index_var)])
            print("Prior Log Prob:")
            print(prior_log_prob(best_index_var).numpy())
            #print("Expected Improvement: "+str(expected_improvement(best_index_var).numpy()[0]))

            best_ei_idx = np.argmax(expected_improvement(best_index_var)[0])

            print("Selected:")
            print(pd.DataFrame({"old":current_point, "new":best_index_var[best_ei_idx].numpy()}))

            best_index_var = scaler.inverse_transform(best_index_var.numpy())

            best_index_var = np.exp(best_index_var)

            print("Selected:")
            print(best_index_var[best_ei_idx].tolist())
            #print(result)
            return best_index_var[best_ei_idx].tolist()

        n_train_samples = x_train.shape[0]

        mixcom = 15 # mixture components
        L1size = 32
        L2size = Y_train.shape[1]
        x = x_train_scaled

        batch_size = 3
        def create_weights():
            return (
                tf.Variable(0.001*tf.random.normal(shape=[batch_size, x.shape[1] + 1, L1size], dtype=tf.float64))
                 , tf.Variable(0.01*tf.random.normal(shape=[batch_size, L1size + 1, len(label_columns_poisson)], dtype=tf.float64))
                 , tf.Variable(0.01*tf.random.normal(shape=[batch_size, L1size + 1, len(label_columns_gaussian)], dtype=tf.float64))
                 , tf.Variable(0.01*tf.random.normal(shape=[batch_size, L1size + 1, 49], dtype=tf.float64))
                 , tf.Variable(0.01*tf.random.normal(shape=[batch_size, L1size + 1, 49], dtype=tf.float64))
            )
        l1weights, l2weights, lgweights, weights, smweights  = create_weights()

        regularizer = tf.keras.regularizers.L1L2(l1=1.0)
        regularizer2 = tf.keras.regularizers.L1L2(l2=1.0)

        def make_joint_mixture_model(x):
            def make_outputs(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, ): #tfd.Sample(
               l1w = l1weights * l1w_in[...,tf.newaxis]  * l1w_out[...,tf.newaxis, :]
               l2w = l2weights * l2w_out[...,tf.newaxis, :]
               lgw = lgweights *  lgw_out[...,tf.newaxis, :]
               wgt = weights *  w_out[...,tf.newaxis, :]
               smw = smweights  * sm_out[...,tf.newaxis, :]
               x1 = tf.nn.relu(tf.matmul(x, l1w[..., :-1, :]) + l1w[..., -1:, :]) # tf.math.tanh
               x2 = tf.matmul(x1, l2w[..., :-1, :]) + l2w[..., -1:, :]
               x2g = tf.matmul(x1, lgw[..., :-1, :]) + lgw[..., -1:, :]
               l = tf.matmul(x1, wgt[..., :-1, :]) + wgt[..., -1:, :]
               sml = tf.matmul(x1, smw[..., :-1, :]) + smw[..., -1:, :]
               return tfd.JointDistributionNamed(dict(
                   mainresult = tfd.Independent(tfd.OneHotCategorical(logits=l), reinterpreted_batch_ndims=1),
                   sidebets = tfd.Independent(tfd.Poisson(log_rate=x2), reinterpreted_batch_ndims=2),
                   sidebets_gaussian = tfd.Independent(tfd.Normal(loc=x2g, scale=1.0), reinterpreted_batch_ndims=2),
                   softmax = tfd.Independent(tfd.OneHotCategorical(logits=sml), reinterpreted_batch_ndims=1)
               ))
            sc = 2.0
            return make_outputs, tfd.JointDistributionNamed(dict(
                l1w_in=tfd.Independent(
                    tfd.Normal(loc=tf.zeros(shape=[batch_size, x.shape[1] + 1], dtype=tf.float64),
                               scale=sc * tf.ones(shape=[batch_size, x.shape[1] + 1], dtype=tf.float64)), reinterpreted_batch_ndims=2),
                l1w_out=tfd.Independent(
                    tfd.Normal(loc=tf.zeros(shape=[batch_size, L1size], dtype=tf.float64),
                               scale=sc * tf.ones(shape=[batch_size, L1size], dtype=tf.float64)), reinterpreted_batch_ndims=2),
                l2w_out=tfd.Independent(
                    tfd.Normal(loc=tf.zeros(shape=[batch_size, len(label_columns_poisson)], dtype=tf.float64),
                               scale=sc * tf.ones(shape=[batch_size, len(label_columns_poisson)], dtype=tf.float64)), reinterpreted_batch_ndims=2),
                lgw_out=tfd.Independent(
                    tfd.Normal(loc=tf.zeros(shape=[batch_size, len(label_columns_gaussian)], dtype=tf.float64),
                               scale=sc * tf.ones(shape=[batch_size, len(label_columns_gaussian)], dtype=tf.float64)), reinterpreted_batch_ndims=2),
                w_out=tfd.Independent(
                    tfd.Normal(loc=tf.zeros(shape=[batch_size, 49], dtype=tf.float64),
                               scale=sc * tf.ones(shape=[batch_size, 49], dtype=tf.float64)), reinterpreted_batch_ndims=2),
                sm_out=tfd.Independent(
                    tfd.Normal(loc=tf.zeros(shape=[batch_size, 49], dtype=tf.float64),
                               scale=sc * tf.ones(shape=[batch_size, 49], dtype=tf.float64)), reinterpreted_batch_ndims=2),
                outputs=make_outputs
             )
            )


        make_mixture_probs_train, joint_model = make_joint_mixture_model(x_train_scaled)
        make_mixture_probs_test, joint_model_test = make_joint_mixture_model(x_test_scaled)
        make_mixture_probs_pred, joint_model_pred = make_joint_mixture_model(x_pred_scaled)

        print(joint_model)
        #joint_model.sample()["w_in"]
        # joint_model.sample()["weight_mean"]
        # joint_model.sample()["weight_scale"]
        #joint_model.sample()["component_distribution"]
        joint_model.sample()["sm_out"]

        # cd = joint_model.parameters["model"]["smweights"]
        # cd.sample()
        # cd.log_prob(cd.sample())
        # cd.prob(cd.sample())

        #np.sum(cd.sample(), axis=1)


        joint_model
        joint_model.parameters
        joint_model.event_shape
        joint_model.batch_shape
        # joint_model.parameters["model"]["outputs"](
        #     joint_model.parameters["model"]["component_distribution"].sample(),
        #     joint_model.parameters["model"]["weights"](
        #         joint_model.parameters["model"]["weight_scale"].sample(),
        #         joint_model.parameters["model"]["weight_mean"].sample()
        #     ).sample()
        # )

        joint_model.sample()
        joint_model.log_prob(joint_model.sample())
        # joint_model.mean()
        # {s:v.shape for s,v in joint_model.sample().items()}
        #empiricalDist = tfd.Empirical(empirical_results, event_ndims=0)  # [x.shape[0], d]

        reg1 = 0.3
        reg2 = 0.3
        reg2g = 0.3
        reg3 = 0.3
        reg4 = 0.3
        kl = 10.0
        lpt = 250.0
        pois = 40.0
        gaus = 100.0
        smx = 250.0

        reg1 = 3.3
        reg2 = 3.3
        reg2g = 3.3
        reg3 = 3.3
        reg4 = 3.3
        kl = 10.0
        lpt = 125.0
        pois = 40.0
        gaus = 100.0
        smx = 200.0
        reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = [
            4.944416316777906,
         10.082641897054293,
         6.360554538652137,
         6.256629927766232,
         1.4681506627824792,
         3.415245441326043,
         299.5762196324349,
         75.10385968042918,
         228.4353119286165,
         778.1051397543414] # GoalDiff

        # reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = [
        #     7.790581026476395,
        #  54.118634654859505,
        #  6.45872733699362,
        #  3.5960901202501345,
        #  3.5185993570104217,
        #  1.9763906465042098,
        #  1463.2568888902092,
        #  184.14446596325726,
        #  498.2907336214255,
        #  836.6541099515921] # Pistor

        # Sky
        #reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = [30.06458190875656,	40.1918119676015,	4.321314865899112,	2.524883871214151,	7.122375159069683,	7.5613837836437146,	3390.5851157718234,	433.29620880964984,	826.7903719059955,	921.7516541247792]

        # Sky 64
        # reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = [
        # 65.95459628358374,
        # 16.401733089714277,
        # 1.3146886990891065,
        # 1.0183184492433304,
        # 3.520335892462611,
        # 4.882879118372497,
        # 5780.341648714803,
        # 796.3056309968048,
        # 861.3450940373662,
        # 758.1406346008536
        # ]
        metaparams = (reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx )

        #metaparams = (0.0051081069220212175, 0.059952482246497935, 0.19331630222128357, 0.035694381954010014, 0.8370222414757852, 116.83426393946185, 183.8459838528271, 207.63877420562545)
        #metaparams = (0.004488,  0.010834,  0.025476,  0.075807,  36.465435,   983.788435,  1995.498555,  471.912004)
        draw_mask = tf.constant([1 if np.mod(z,7)==(z//7) else 0 for z in range(49)], dtype=tf.float32)
        wdl_mask = tf.transpose(tf.constant([
            [1 if np.mod(z, 7) > (z // 7) else 0 for z in range(49)],
            [1 if np.mod(z, 7) == (z // 7) else 0 for z in range(49)],
            [1 if np.mod(z, 7) < (z // 7) else 0 for z in range(49)]], dtype=tf.float32))


        def loss_function(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, X, Y, Y2, reduce_axis=None): # weight_mean, weight_scale,
            Y_pois = Y2[:,label_columns_poisson]
            Y_gaus = Y2[:,label_columns_gaussian]

            mk, joint_model = make_joint_mixture_model(X)
            #Y = tf.cond(tf.rank(weights)==3, lambda: Y[..., tf.newaxis, :, :], lambda: Y)
            lpp = joint_model.log_prob_parts({
                'l1w_in': l1w_in,
                'l1w_out': l1w_out,
                'l2w_out': l2w_out,
                'lgw_out': lgw_out,
                'w_out': w_out,
                'sm_out': sm_out,
                'outputs': {"mainresult":Y, "sidebets":Y_pois, "sidebets_gaussian":Y_gaus, "softmax":Y}
            })
            lp = tf.reduce_sum(lpp['l1w_in']+lpp['l1w_out']+lpp['l2w_out']+lpp['lgw_out']+lpp['w_out']+lpp['sm_out'])
            #reg_loss = create_laplacian_loss(smweights, alpha=0.3)
            rl_l1 = reg1*regularizer(l1weights)
            rl_l2 = reg2*regularizer2(l2weights)
            rl_lg = reg2g*regularizer2(lgweights)
            rl_w = reg3*regularizer2(weights)
            rl_sm = reg4*regularizer2(smweights)
            reg_loss2 = rl_l1+rl_l2+rl_lg+rl_w+rl_sm
            achievable_points = tf.matmul(Y, tf.cast(point_matrix, tf.float32))
            pred = mk(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out)

            predprob = tf.nn.softmax(pred.parameters["model"]["mainresult"].distribution.logits, axis=-1)
            logpoints = tf.math.log(1+predprob)  * tf.cast(achievable_points, tf.float64) # * (0.0001+tf.stop_gradient(pred.parameters["model"]["softmax"].mean()))
            reg_loss3 = create_laplacian_loss(predprob, alpha=10.0)
            actual_points = predprob * tf.cast(achievable_points, tf.float64)
            actual_points = tf.reduce_sum(actual_points, axis=-1)
            logpoints = tf.reduce_sum(logpoints, axis=-1)
            prediction = tf.one_hot(tf.argmax(pred.parameters["model"]["mainresult"].mean(), axis=-1), 49)
            numdraws = tf.reduce_sum(prediction * draw_mask, axis=-1)
            real_points = tf.reduce_sum(prediction * achievable_points, axis=-1)
            accuracy = tf.reduce_sum(tf.cast(tf.matmul(prediction, wdl_mask) * tf.matmul(Y, wdl_mask), tf.float32), axis=-1)
            draw_points = tf.reduce_sum(prediction * draw_mask * achievable_points, axis=-1)/tf.reduce_mean(numdraws)
            softmax_lp = pred.parameters["model"]["softmax"].log_prob(Y)/Y.shape[-2]
            poisson_lp = pred.parameters["model"]["sidebets"].log_prob(Y_pois)/Y_pois.shape[-2]
            gaussian_lp = pred.parameters["model"]["sidebets_gaussian"].log_prob(Y_gaus)/Y_gaus.shape[-2]
            kl_div = pred.parameters["model"]["mainresult"].kl_divergence(pred.parameters["model"]["softmax"])/Y2.shape[-2]

            softmax_acc = tf.reduce_sum(tf.one_hot(tf.argmax(pred.parameters["model"]["softmax"].mean(), axis=-1), 49) * Y, axis=-1)
            lpp.update({"rl_l1":rl_l1, "rl_l2":rl_l2, "rl_lg":rl_lg, "rl_w":rl_w, "rl_sm":rl_sm, "laplacian_prob_reg_loss":reg_loss3,
                        "joint_model_log_prob":lp, "numdraws":tf.reduce_mean(numdraws, axis=reduce_axis),  "draw_points":tf.reduce_mean(draw_points, axis=reduce_axis), "accuracy":tf.reduce_mean(accuracy, axis=reduce_axis),
                        "kl_div":tf.reduce_mean(kl_div, axis=reduce_axis), "softmax_lp_mean":tf.reduce_mean(softmax_lp, axis=reduce_axis), "softmax_mean_acc":tf.reduce_mean(softmax_acc, axis=reduce_axis),
                        "poisson_lp_mean":tf.reduce_mean(poisson_lp, axis=reduce_axis), "gaussian_lp_mean":tf.reduce_mean(gaussian_lp, axis=reduce_axis),
                        "logpoints":tf.reduce_mean(logpoints, axis=reduce_axis), "mean_points":tf.reduce_mean(actual_points, axis=reduce_axis), "real_points":tf.reduce_mean(real_points, axis=reduce_axis)
                       })
            # Pistor lpp["total"] = tf.reduce_sum(lp) - 0.03*tf.reduce_sum(kl_div) + 0.2*tf.reduce_sum(logpoints)  +0.202*tf.reduce_sum(poisson_lp) + 0.5*tf.reduce_sum(softmax_lp) - tf.reduce_sum(reg_loss2)#- lpp["outputs"] + lpp["outputs"]*0.3 #
            # lpp["total"] = tf.reduce_sum(lp) - 0.06 * tf.reduce_sum(kl_div) + 0.3 * tf.reduce_sum(
            #      logpoints) + 0.202 * tf.reduce_sum(poisson_lp) + 0.5 * tf.reduce_sum(softmax_lp) - tf.reduce_sum(
            #      reg_loss2)  # Pistor
            lpp["total"] = tf.reduce_mean(lp) - kl * tf.reduce_mean(kl_div) + lpt * tf.reduce_mean(logpoints) + pois * tf.reduce_mean(poisson_lp) + gaus * tf.reduce_mean(gaussian_lp) + smx * tf.reduce_mean(softmax_lp) - reg_loss2  # Pistor
            # Sky lpp["total"] = tf.reduce_sum(lp) - 0.01*tf.reduce_sum(kl_div) + 0.3*tf.reduce_sum(logpoints)  +0.202*tf.reduce_sum(poisson_lp) + 0.5*tf.reduce_sum(softmax_lp) - tf.reduce_sum(reg_loss2)#- lpp["outputs"] + lpp["outputs"]*0.3 #
            # Sky lpp["total"] = tf.reduce_sum(lp) - 0.25*tf.reduce_sum(kl_div) + 0.6*tf.reduce_sum(logpoints)  +0.202*tf.reduce_sum(poisson_lp) + 0.5*tf.reduce_sum(softmax_lp) - tf.reduce_sum(reg_loss2)#- lpp["outputs"] + lpp["outputs"]*0.3 #
            #lpp["total"] = tf.reduce_mean(lp) - 100.0*tf.reduce_mean(kl_div) + 500.0*tf.reduce_mean(logpoints)  +400.0*tf.reduce_mean(poisson_lp) + 200.0*tf.reduce_mean(softmax_lp) - tf.reduce_mean(reg_loss2)#- lpp["outputs"] + lpp["outputs"]*0.3 #
            #lpp["total"] = tf.reduce_sum(lp) - 0.95*tf.reduce_sum(kl_div) + 0.5*tf.reduce_sum(logpoints)  +0.202*tf.reduce_sum(poisson_lp) + 0.5*tf.reduce_sum(softmax_lp) - tf.reduce_sum(reg_loss2)#- lpp["outputs"] + lpp["outputs"]*0.3 # Sky
            # GD
            #lpp["total"] = tf.reduce_sum(lp) - 0.001*tf.reduce_sum(kl_div) + 0.25*tf.reduce_sum(logpoints)  +0.202*tf.reduce_sum(poisson_lp) + 0.5*tf.reduce_sum(softmax_lp) - tf.reduce_sum(reg_loss2)# GoalDiff
            #
            #+0.3 * tf.reduce_sum(actual_points)
            # - tf.reduce_sum(reg_loss3)+0.0*tf.reduce_sum(lpp["outputs"])
            return lpp

        def analyse_losses(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, X, Y, Y2, reduce_axis=None): # weight_mean, weight_scale,
            return loss_function(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, X, Y, Y2, reduce_axis)

        def target_log_prob_cat(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out):
            return loss_function(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, x_train_scaled, outputs_, Y_train)["total"]


        def target_log_prob_cat_test(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out):
            return loss_function(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, x_test_scaled, test_outputs_, Y_test)["total"]


        def analyse_target_log_prob_cat(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, reduce_axis=None):
            #print("Train")
            al = analyse_losses(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, x_train_scaled, outputs_, Y_train, reduce_axis)
            return {k:v.numpy() for k,v in al.items()}

        def analyse_test_log_prob_cat(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, reduce_axis=None):
            #print("Test")
            return {k:v.numpy() for k,v in analyse_losses(l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out, x_test_scaled, test_outputs_, Y_test, reduce_axis).items()}

        print(joint_model)
        jm_components = joint_model.event_shape_tensor().keys()
        def make_surrogate_posterior():
            return tfp.experimental.vi.build_factored_surrogate_posterior(
                event_shape=[joint_model.event_shape_tensor()[c] for c in [ 'l1w_in', 'l1w_out','l2w_out', 'lgw_out', 'w_out', 'sm_out']],
                #constraining_bijectors=[None, None, None, None],
                trainable_distribution_fn=lambda initial_loc, initial_scale, event_ndims,
                                             validate_args: tfp.experimental.vi.build_trainable_location_scale_distribution(
                    tf.cast(initial_loc, tf.float64),
                    tf.cast(initial_scale, tf.float64),
                    event_ndims,
                    distribution_fn=tfd.Normal),
        )


        surrogate_posterior = make_surrogate_posterior()
        optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        searchlog = pd.DataFrame(columns=["dtn", "accept", "accprob", "score", "tpts", "pts", "logp", "pois", "gaus", "smx", "reg1", "reg2", "reg2g", "reg3", "reg4", "wkl", "wlpt", "wpois", "wgaus", "wsmx", "reset"])
        searchlog_filename = sorted([f for f in os.listdir("models") if re.search("mcmc_deterministic_"+FLAGS.prefix+"_"+FLAGS.target_system+"_"+".*csv", f)])
        if FLAGS.checkpoints != "" and len(searchlog_filename) > 0:
            searchlog_filename = searchlog_filename[-1]
            print(searchlog_filename)
            searchlog = pd.read_csv("models/"+searchlog_filename)
            if "Unnamed: 0" in searchlog.columns:
                searchlog.drop(columns="Unnamed: 0", inplace=True)
            if "reset" not in searchlog.columns:
                searchlog["reset"]=None

        #searchlog = pd.concat([pd.read_csv("models/mcmc_deterministic_vi5d2_GoalDiff_20201125_083211.csv"), pd.read_csv("models/mcmc_deterministic_vi5d2_GoalDiff_20201125_130602.csv")], axis=0, ignore_index=True)
        #searchlog.drop(columns="Unnamed: 0", inplace=True)
        inputfilename = ["mcmc_deterministic_vi5c_Sky_20201122_221608.pickle"]
        inputfilename = ["mcmc_deterministic_vi5c_Pistor_20201122_204812.pickle"]
        inputfilename = ["mcmc_deterministic_vi5d_Pistor_20201124_073714.pickle"] # 0.793859
        inputfilename = ["mcmc_deterministic_vi5c_Sky_20201123_065849.pickle"]
        inputfilename = ["mcmc_deterministic_vi5c_GoalDiff_20201123_085215.pickle"]
        inputfilename = ["mcmc_deterministic_vi5d2_Sky_20201124_162042.pickle"]
        inputfilename = ["mcmc_deterministic_vi5d2_Sky_20201129_161219.pickle"]
        inputfilename = ["mcmc_deterministic_vi5d2_Pistor_20201128_202100.pickle"] # 0.84
        inputfilename = sorted([f for f in os.listdir("models") if re.search("mcmc_deterministic_"+FLAGS.prefix+"_"+FLAGS.target_system+"_"+".*pickle", f)])
        if FLAGS.checkpoints != "" and len(inputfilename) > 0:
            if str.isnumeric(FLAGS.checkpoints[1:]):
                inputfilename = inputfilename[int(FLAGS.checkpoints)]
            elif FLAGS.checkpoints=="best":
                inputfilename = "mcmc_deterministic_"+FLAGS.prefix+"_"+FLAGS.target_system+"_"+searchlog.dtn[searchlog.pts==searchlog.pts.max()].iloc[-1]+".pickle" # row with max points
            else:
                inputfilename = "mcmc_deterministic_"+FLAGS.prefix+"_"+FLAGS.target_system+"_"+FLAGS.checkpoints+".pickle"
            print(inputfilename)
            filehandler = open("models/"+inputfilename, 'rb')
            all_states, all_weights, metaparams = pickle.load(filehandler)
            for value, variable in zip(all_states, surrogate_posterior.trainable_variables):
                variable.assign(value)
            if len(all_weights) == 4: # bugfix
                for value, variable in zip(all_weights, [l1weights, l2weights, weights, smweights]):
                    variable.assign(value)
            else:
                for value, variable in zip(all_weights, [l1weights, l2weights, lgweights, weights, smweights]):
                    variable.assign(value)

            reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = metaparams


        if FLAGS.action=="train":
            print(searchlog)
            best_score = 1e10
            last_accepted_score = 1e10
            rounds_without_score_improvement = 0
            for round_number in range(FLAGS.train_steps):
                print({"Round":round_number})
                reset_performed = False
                oldoldmetaparams = metaparams
                if (np.mod(round_number, 5)==0): # jump away every five rounds
                    metaparams = [m * np.random.lognormal(0, 0.1) for m in metaparams]

                oldmetaparams = metaparams

                if (round_number<FLAGS.warmup):
                    metaparams = [m * np.random.lognormal(0, 0.3) for m in metaparams]
                else:
                    metaparams = find_next_params_from_gaussian_process(searchlog)

                # metaparams = [30.06458190875656, 40.1918119676015, 4.321314865899112, 2.524883871214151,
                #               7.122375159069683,
                #               7.5613837836437146, 3390.5851157718234, 433.29620880964984, 826.7903719059955,
                #               921.7516541247792]

                # Perform a model reset if no improvement is showing
                if  False and round_number>0 and es2<best_score:
                    best_score = es2
                    rounds_without_score_improvement = 0
                else:
                    rounds_without_score_improvement += 1
                    print("rounds_without_score_improvement: "+str(rounds_without_score_improvement))

                    if rounds_without_score_improvement>=0 and round_number != FLAGS.train_steps-1:
                        print("MODEL RESET")
                        reset_performed = True
                        #metaparams = [m * np.random.lognormal(0, 0.3) for m in metaparams] # jump elsewhere
                        l = len(surrogate_posterior.trainable_variables)
                        for initvalue, variable in zip(make_surrogate_posterior().trainable_variables[:l], surrogate_posterior.trainable_variables):
                            variable.assign(initvalue)
                        for initvalue, variable in zip(create_weights()[l:], [l1weights, l2weights, lgweights, weights, smweights]):
                            variable.assign(initvalue)
                        best_score = 1e10
                        rounds_without_score_improvement = 0


                comparedf = pd.DataFrame({"previous":oldoldmetaparams, "candidate":metaparams},
                                         index = ["reg1", "reg2", "reg2g", "reg3", "reg4", "kl", "lpt", "pois", "gaus", "smx"])
                print(comparedf)
                reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = metaparams
                print(metaparams)
                # trace_fn = lambda loss, grads, vars: (loss, [], analyse_losses(*[v.mean() for v in surrogate_posterior.sample_distributions()[0]], x_test_scaled, test_outputs_, Y_test)["mean_points"]),
                surrpost_vars, samples = surrogate_posterior.sample_distributions(100)
                l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out = surrpost_vars

                def dictdiff(x,y):
                    return {k:y[k]-x[k] for k in x.keys()}

                def eval_score(loss0a, loss0b):
                    sm0 = 0.1*loss0a["softmax_lp_mean"]-loss0b["softmax_lp_mean"]
                    lp0 = 0.1*loss0a["logpoints"] - loss0b["logpoints"]
                    ps0 = 0.1*loss0a["poisson_lp_mean"] - loss0b["poisson_lp_mean"]
                    gs0 = 0.1*loss0a["gaussian_lp_mean"] - loss0b["gaussian_lp_mean"]
                    pt0 = 0.1*loss0a["real_points"] - loss0b["real_points"]
                    #score0=sm0+lp0+ps0+pt0-10*loss0b["real_points"]
                    score0 = sm0 + 0.3*ps0 + gs0 + point_scale*(10*lp0 + 0.5*pt0)
                    return(score0)

                def train_surrogate_posterior(learning_rate=1e-2, steps=FLAGS.save_steps):
                    tfp.vi.fit_surrogate_posterior(
                        target_log_prob_cat,
                        surrogate_posterior,
                        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        num_steps=steps,
                        # trainable_variables=surrogate_posterior.trainable_variables,
                        seed=42,
                        sample_size=8)
                    surrpost_vars, samples = surrogate_posterior.sample_distributions(100)
                    loss1a = analyse_target_log_prob_cat(*samples)
                    loss1b = analyse_test_log_prob_cat(*samples)
                    score = eval_score(loss1a, loss1b)
                    return (score, loss1a, loss1b)


                def optimize_surrogate_posterior(learning_rate, stepsize = 50):
                    oldscore = 1e10
                    r = 0
                    while(True):
                        score, loss_train, loss_test = train_surrogate_posterior(learning_rate, steps=stepsize)
                        r += stepsize
                        print((r, score, loss_train["real_points"], loss_test["real_points"]))
                        if score > oldscore:
                            return score, loss_train, loss_test
                        else:
                            oldscore = score


                # save previous values
                oldvalues = [v.numpy() for v in list(surrogate_posterior.trainable_variables) + [l1weights, l2weights, lgweights, weights, smweights]]
                # loss0a = analyse_target_log_prob_cat(*[v.mean() for v in surrpost_vars])
                # loss0b = analyse_test_log_prob_cat(*[v.mean() for v in surrpost_vars])
                loss0a = analyse_target_log_prob_cat(*samples)
                loss0b = analyse_test_log_prob_cat(*samples)
                es0 = eval_score(loss0a, loss0b)
                loss0b["score"] = es0
                print(pd.DataFrame([loss0a, loss0b]).T.iloc[11:])

                if FLAGS.save_steps==0:
                    es1, loss1a, loss1b  = optimize_surrogate_posterior(1e-2)
                    loss1b["score"] = es1
                    print(pd.DataFrame([loss0a, loss1a, dictdiff(loss0a, loss1a), loss0b, loss1b, dictdiff(loss0b, loss1b)]).T.iloc[11:])

                    es3, loss3a, loss3b  = optimize_surrogate_posterior(1e-3)
                    loss3b["score"] = es3
                    print(pd.DataFrame([loss1a, loss3a, dictdiff(loss1a, loss3a), loss1b, loss3b, dictdiff(loss1b, loss3b)]).T.iloc[11:])

                    es2, loss2a, loss2b  = optimize_surrogate_posterior(1e-4)
                    loss2b["score"] = es2
                    print(pd.DataFrame([loss3a, loss2a, dictdiff(loss3a, loss2a), loss3b, loss2b, dictdiff(loss3b, loss2b)]).T.iloc[11:])

                else:
                    es1, loss1a, loss1b  = train_surrogate_posterior(learning_rate=1e-2, steps=FLAGS.save_steps)
                    loss1b["score"] = es1
                    print(pd.DataFrame([loss0a, loss1a, dictdiff(loss0a, loss1a), loss0b, loss1b, dictdiff(loss0b, loss1b)]).T.iloc[11:])

                    es2, loss2a, loss2b  = train_surrogate_posterior(learning_rate=1e-3, steps=FLAGS.save_steps)
                    loss2b["score"] = es2
                    print(pd.DataFrame([loss0a, loss2a, dictdiff(loss0a, loss2a), loss0b, loss2b, dictdiff(loss0b, loss2b)]).T.iloc[11:])

                #accept_prob = np.exp((es0-es2)/0.03)
                accept_prob = np.exp((last_accepted_score-es2)/0.12)
                #accept_prob = np.exp((loss2b["real_points"] - loss0b["real_points"]) / 0.02)

                best_points = searchlog.pts.max()
                min_points = searchlog.pts.quantile(0.7)
                accept = es2 < last_accepted_score or random.random() < accept_prob or loss2b["real_points"]>best_points or (loss2b["real_points"]>loss0b["real_points"] and loss2b["real_points"]>min_points)
                if es2 < last_accepted_score:
                    accept_prob = np.nan
                # accept = loss2b["real_points"]>loss0b["real_points"] or random.random() < accept_prob
                # if loss2b["real_points"]>loss0b["real_points"]:
                #     accept_prob = np.nan
                print((last_accepted_score, es0, es1, es2, accept, accept_prob, loss0b["real_points"], loss1b["real_points"], loss2b["real_points"]))

                dtn = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_row = [dtn, accept, accept_prob, es2, loss2a["real_points"], loss2b["real_points"], loss2b["logpoints"], loss2b["poisson_lp_mean"], loss2b["gaussian_lp_mean"], loss2b["softmax_lp_mean"]]+metaparams+[reset_performed]
                searchlog = searchlog.append({k:v for k,v in zip(searchlog.columns, new_row)}, ignore_index=True)
                print(searchlog)
                if accept:
                    last_accepted_score = es2
                    filename = "models/mcmc_deterministic_"+FLAGS.prefix+"_"+FLAGS.target_system+"_"+dtn+".pickle"
                    filehandler = open(filename, 'wb')
                    pickle.dump((surrogate_posterior.trainable_variables, (l1weights, l2weights, lgweights, weights, smweights), metaparams), filehandler)

                else:
                    # restore previous values
                    metaparams = oldmetaparams
                    reg1, reg2, reg2g, reg3, reg4, kl, lpt, pois, gaus, smx = metaparams
                    l = len(surrogate_posterior.trainable_variables)
                    for value, variable in zip(oldvalues[:l], surrogate_posterior.trainable_variables):
                        variable.assign(value)
                    for value, variable in zip(oldvalues[l:], [l1weights, l2weights, lgweights, weights, smweights]):
                        variable.assign(value)

                if round_number == FLAGS.warmup-1:
                    filename = "models/mcmc_deterministic_" + FLAGS.prefix + "_" + FLAGS.target_system + "_" + dtn + ".csv"
                    searchlog.to_csv(filename, index=False)

            filename = "models/mcmc_deterministic_" + FLAGS.prefix + "_" + FLAGS.target_system + "_" + dtn + ".csv"
            searchlog.to_csv(filename, index=False)

        plt.figure()
        plt.plot(searchlog.score[1:])
        plt.plot(searchlog.score[1:].loc[searchlog[1:].accept])
        #plt.ylim(top=15)
        plt.axhline(searchlog.score.min(), color="gray")
        plt.show()

        plt.figure()
        plt.plot(searchlog.pts)
        plt.plot(searchlog.pts.loc[searchlog.accept])
        plt.plot(searchlog.tpts)
        plt.plot(searchlog.tpts.loc[searchlog.accept])
        plt.axhline(searchlog.pts.max(), color="gray")
        plt.ylim(np.quantile(searchlog.pts, 0.1),np.quantile(searchlog.tpts, 1.0))
        plt.show()

        def plot_searchlog(xlabel, ylabel, ylabel2=None, x=None, y=None, y2=None, logy=False, lim=5, lim_x=None, lim_y=None):
            if lim_x is None:
                lim_x = lim
            if lim_y is None:
                lim_y = lim
            fig = plt.figure()
            if x is None:
                x = searchlog.loc[:,xlabel]
            if y is None:
                y = searchlog.loc[:,ylabel]
            if y2 is None and ylabel2 is not None:
                y2 = searchlog.loc[:,ylabel2]
            minx = np.min(x)
            maxx = np.max(x)
            miny = np.min(y)
            maxy = np.max(y)
            if y2 is not None:
                miny = min(miny, np.min(y2))
                maxy = max(maxy, np.max(y2))

            plt.plot(x[1:-10], y[1:-10], color="lightgray")
            plt.plot(x[-11:], y[-11:])
            plt.scatter(x[1:], y[1:])
            plt.scatter(x[searchlog.accept][1:], y[searchlog.accept][1:])
            if y2 is not None:
                plt.plot(x[1:-10], y2[1:-10], color="gray")
                plt.plot(x[-11:], y2[-11:])
                plt.scatter(x[1:], y2[1:])
                plt.scatter(x[searchlog.accept][1:], y2[searchlog.accept][1:])
            fig.axes[0].set_xlabel(xlabel)
            fig.axes[0].set_ylabel(ylabel)
            if xlabel!="logp" and xlabel!="score":
                fig.axes[0].set_xscale('log')
                iqrx = np.log(x).quantile(.75) - np.log(x).quantile(.25)
                if lim is not None:
                    plt.xlim(max(minx, np.exp(np.log(x).median() - lim_x * iqrx)), min(maxx, np.exp(np.log(x).median() + lim_x * iqrx)))
            else:
                iqrx = x.quantile(.75) - x.quantile(.25)
                if lim is not None:
                    plt.xlim(max(minx, x.median() - lim_x * iqrx), min(maxx, x.median() + lim_x * iqrx))
            y3 = y if y2 is None else pd.concat([y, y2])

            if logy:
                fig.axes[0].set_yscale('log')
                iqry = np.log(y3).quantile(.75) - np.log(y3).quantile(.25)
                if lim is not None:
                    plt.ylim(max(miny, np.exp(np.log(y3).median()-lim_y*iqry)), min(maxy, np.exp(np.log(y3).median()+lim_y*iqry)))
            else:
                iqry = y3.quantile(.75) - y3.quantile(.25)
                if lim is not None:
                    plt.ylim((max(miny, y3.median()-lim_y*iqry), min(maxy, y3.median()+lim_y*iqry)))
            if ylabel=="pts":
                plt.axhline(y.max(), color="red")
            plt.show()
        if False:
            plot_searchlog("reg1", "pts", "tpts", lim_y=2)
            plot_searchlog("reg2", "pois", y=np.max(searchlog.pois)-searchlog.pois+0.01, logy=True)
            plot_searchlog("reg3", "logp")
            plot_searchlog("reg2g", "gaus")
            plot_searchlog("reg4", "smx", y=np.max(searchlog.smx)-searchlog.smx+0.01, logy=True)
            plot_searchlog("wlpt", "logp")
            plot_searchlog("wlpt", "pts", lim=None)
            plot_searchlog("wpois", "pts")
            plot_searchlog("wsmx", "pts")
            plot_searchlog("wgaus", "pts")
            plot_searchlog("logp", "pts")
            plot_searchlog("wkl", "pts")
            plot_searchlog("wlpt", "pts", "tpts", lim=None)
            plot_searchlog("wpois", "pts", "tpts")
            plot_searchlog("wsmx", "pts", "tpts")
            plot_searchlog("wgaus", "pts", "tpts", lim_y=2)
            plot_searchlog("logp", "pts", "tpts")
            plot_searchlog("wkl", "pts", "tpts")
            plot_searchlog("wlpt", "wkl", logy=True)
            plot_searchlog("score", "pts", lim=4)
            plot_searchlog("wlpt", "score", lim=None)
            plot_searchlog("wpois", "score")
            plot_searchlog("wsmx", "score")
            plot_searchlog("wgaus", "score")
            plot_searchlog("reg1", "score")
            plot_searchlog("reg2", "score")
            plot_searchlog("reg2g", "score")
            plot_searchlog("reg3", "score")
            plot_searchlog("reg4", "score")
            plot_searchlog("wkl", "score")

            #sns.kdeplot(data=searchlog.loc[:,["reg1", "wlpt", "pts"]], hue="pts")

        surrpost_vars, _ = surrogate_posterior.sample_distributions()
        l1w_in, l1w_out, l2w_out, lgw_out, w_out, sm_out = surrpost_vars

        fig, ax = plt.subplots(6, 3)
        plt.subplots_adjust(left=0.04, bottom=0.03, right=0.96, top=0.97, wspace=0.08, hspace=0.18)

        sns.distplot(l1w_in.mean().numpy().flatten(), ax=ax[0][0], rug=True, hist=False)
        sns.distplot(l1w_in.stddev().numpy().flatten(), ax=ax[0][1], rug=True, hist=False)
        sns.distplot(l1w_out.mean().numpy().flatten(), ax=ax[0][0], rug=True, hist=False)
        sns.distplot(l1w_out.stddev().numpy().flatten(), ax=ax[0][1], rug=True, hist=False)
        #sns.distplot(l2w_in.mean().numpy().flatten(), ax=ax[1][0], rug=True, hist=False)
        #sns.distplot(l2w_in.stddev().numpy().flatten(), ax=ax[1][1], rug=True, hist=False)
        sns.distplot(l2w_out.mean().numpy().flatten(), ax=ax[1][0], rug=True, hist=False)
        sns.distplot(l2w_out.stddev().numpy().flatten(), ax=ax[1][1], rug=True, hist=False)
        #sns.distplot(w_in.mean().numpy().flatten(), ax=ax[2][0], rug=True, hist=False)
        #sns.distplot(w_in.stddev().numpy().flatten(), ax=ax[2][1], rug=True, hist=False)
        sns.distplot(w_out.mean().numpy().flatten(), ax=ax[2][0], rug=True, hist=False)
        sns.distplot(w_out.stddev().numpy().flatten(), ax=ax[2][1], rug=True, hist=False)
        #sns.distplot(sm_in.mean().numpy().flatten(), ax=ax[3][0], rug=True, hist=False)
        #sns.distplot(sm_in.stddev().numpy().flatten(), ax=ax[3][1], rug=True, hist=False)
        sns.distplot(sm_out.mean().numpy().flatten(), ax=ax[3][0], rug=True, hist=False)
        sns.distplot(sm_out.stddev().numpy().flatten(), ax=ax[3][1], rug=True, hist=False)

        sns.distplot(lgw_out.mean().numpy().flatten(), ax=ax[4][0], rug=True, hist=False)
        sns.distplot(lgw_out.stddev().numpy().flatten(), ax=ax[4][1], rug=True, hist=False)


        sns.kdeplot(l1weights.numpy().flatten(), ax=ax[0][2], clip=(np.mean(l1weights.numpy())-3*np.std(l1weights.numpy()), np.mean(l1weights.numpy())+3*np.std(l1weights.numpy())))
        sns.kdeplot(l2weights.numpy().flatten(), ax=ax[1][2], clip=(np.mean(l2weights.numpy())-3*np.std(l2weights.numpy()), np.mean(l2weights.numpy())+3*np.std(l2weights.numpy())))
        sns.kdeplot(weights.numpy().flatten(), ax=ax[2][2], clip=(np.mean(weights.numpy())-3*np.std(weights.numpy()), np.mean(weights.numpy())+3*np.std(weights.numpy())))
        sns.kdeplot(smweights.numpy().flatten(), ax=ax[3][2], clip=(np.mean(smweights.numpy())-3*np.std(smweights.numpy()), np.mean(smweights.numpy())+3*np.std(smweights.numpy())))
        sns.kdeplot(lgweights.numpy().flatten(), ax=ax[4][2], clip=(np.mean(lgweights.numpy())-3*np.std(lgweights.numpy()), np.mean(lgweights.numpy())+3*np.std(lgweights.numpy())))
        # try:
        #     ax[5][0].plot(losses.numpy())
        #     ax[5][1].plot(losses.numpy()[750:])
        # except NameError:
        #     pass
        plt.show()

        if False:
            fig, ax = plt.subplots(4, 2)
            sns.distplot(((l1weights - l1weights0) * 2 / (l1weights + l1weights0)).numpy(), rug=True, ax=ax[0][0])
            sns.distplot(((l2weights - l2weights0) * 2 / (l2weights + l2weights0)).numpy(), rug=True, ax=ax[1][0])
            sns.distplot(((smweights - smweights0) * 2 / (smweights + smweights0)).numpy(), rug=True, ax=ax[2][0])
            sns.distplot(((weights - weights0) * 2 / (weights + weights0)).numpy(), rug=True, ax=ax[3][0])
            sns.distplot(((l1weights - l1weights0) * 2 * l1weights / (l1weights + l1weights0)).numpy(), rug=True, ax=ax[0][1])
            sns.distplot(((l2weights - l2weights0) * 2 * l2weights/ (l2weights + l2weights0)).numpy(), rug=True, ax=ax[1][1])
            sns.distplot(((smweights - smweights0) * 2 * weights/ (smweights + smweights0)).numpy(), rug=True, ax=ax[2][1])
            sns.distplot(((weights - weights0) * 2 * smweights / (weights + weights0)).numpy(), rug=True, ax=ax[3][1])
            plt.show()


        target_log_prob_cat_test(*[v.sample() for v in surrpost_vars])

        sample_train = analyse_losses(*[v.sample(100) for v in surrpost_vars], x_train_scaled, outputs_, Y_train, reduce_axis=-1)
        train_points = sample_train["real_points"].numpy()
        train_softpoints = sample_train["mean_points"].numpy()
        sample_test = analyse_losses(*[v.sample(100) for v in surrpost_vars], x_test_scaled, test_outputs_, Y_test, reduce_axis=-1)
        test_points = sample_test["real_points"].numpy()
        test_softpoints = sample_test["mean_points"].numpy()

        #train_points = [analyse_losses(*[v.sample() for v in surrpost_vars], x_train_scaled, outputs_, Y_train)["real_points"].numpy() for i in range(100)]
        train_mean_points = analyse_losses(*[v.mean() for v in surrpost_vars], x_train_scaled, outputs_, Y_train)["real_points"].numpy()

        #test_points = [analyse_losses(*[v.sample() for v in surrpost_vars], x_test_scaled, test_outputs_, Y_test)["real_points"] .numpy() for i in range(100)]
        test_mean_points = analyse_losses(*[v.mean() for v in surrpost_vars], x_test_scaled, test_outputs_, Y_test)["real_points"].numpy()

        #train_softpoints = [analyse_losses(*[v.sample() for v in surrpost_vars], x_train_scaled, outputs_, Y_train)["mean_points"].numpy() for i in range(100)]
        train_mean_softpoints = analyse_losses(*[v.mean() for v in surrpost_vars],  x_train_scaled, outputs_, Y_train)["mean_points"].numpy()
        train_mean_logpoints = analyse_losses(*[v.mean() for v in surrpost_vars],  x_train_scaled, outputs_, Y_train)["logpoints"].numpy()

        #test_softpoints = [analyse_losses(*[v.sample() for v in surrpost_vars], x_test_scaled, test_outputs_, Y_test)["mean_points"] .numpy() for i in range(100)]
        test_mean_softpoints = analyse_losses(*[v.mean() for v in surrpost_vars], x_test_scaled, test_outputs_, Y_test)["mean_points"].numpy()
        test_mean_logpoints = analyse_losses(*[v.mean() for v in surrpost_vars], x_test_scaled, test_outputs_, Y_test)["logpoints"].numpy()

        fig, ax = plt.subplots(3, 2)
        plt.subplots_adjust(left=0.04, bottom=0.08, right=0.96, top=0.92, wspace=0.08, hspace=0.18)
        for i in range(batch_size):
            sns.kdeplot(train_points[:,i], ax=ax[0][0])
            sns.kdeplot(test_points[:,i], ax=ax[0][1])
            sns.kdeplot(train_softpoints[:,i], ax=ax[1][0])
            sns.kdeplot(test_softpoints[:,i], ax=ax[1][1])
            sns.kdeplot(sample_train["logpoints"].numpy()[:,i], ax=ax[2][0])
            sns.kdeplot(sample_test["logpoints"].numpy()[:,i], ax=ax[2][1])
        ax[0][0].axvline(x=train_mean_points)
        ax[0][1].axvline(x=test_mean_points)
        ax[1][0].axvline(x=train_mean_softpoints)
        ax[1][1].axvline(x=test_mean_softpoints)
        ax[2][0].axvline(x=train_mean_logpoints)
        ax[2][1].axvline(x=test_mean_logpoints)
        plt.show()

        t2 = make_mixture_probs_train(*[v.mean() for v in surrpost_vars]).parameters["model"]["mainresult"].mean().numpy()
        df2 = pd.concat([create_df(create_argmax_prediction_from_mean(t2[i], Y_train.shape[0]), Y_train, "Train") for i in range(batch_size)], axis=0)
        plot_predictions_3(df2, "poisson", "Train", silent=False)

        t2 = make_mixture_probs_test(*[v.mean() for v in surrpost_vars]).parameters["model"]["mainresult"].mean().numpy()
        df2 = pd.concat([create_df(create_argmax_prediction_from_mean(t2[i], Y_test.shape[0]), Y_test, "Test") for i in range(batch_size)], axis=0)
        plot_predictions_3(df2, "poisson", "Test", silent=False)

        pred_probs = make_mixture_probs_pred(*[v.mean() for v in surrpost_vars]).parameters["model"]["mainresult"].mean().numpy()
        df_pred = pd.concat([create_df(create_argmax_prediction_from_mean(pred_probs[i], Y_pred.shape[0]), Y_pred, "Pred", team1, team2) for i in range(batch_size)], axis=0)
        df_pred["pred"] = df_pred.pGS.astype(str) + ":" + df_pred.pGC.astype(str)

        if False:
            t2 = make_mixture_probs_test(*[v.mean() for v in surrpost_vars]).parameters["model"]["softmax"].mean().numpy()
            df2 = create_df(create_argmax_prediction_from_mean(t2, Y_test.shape[0]), Y_test, "SoftmaxTest")
            plot_predictions_3(df2, "poisson", "SoftmaxTest", silent=False)

        smpred_probs = make_mixture_probs_pred(*[v.mean() for v in surrpost_vars]).parameters["model"]["softmax"].mean().numpy()
        df_smpred = pd.concat([create_df(create_argmax_prediction_from_mean(smpred_probs, Y_pred.shape[0]), Y_pred, "SoftmaxPred", team1, team2) for i in range(batch_size)], axis=0)
        df_smpred["pred"] = df_smpred.pGS.astype(str) + ":" + df_smpred.pGC.astype(str)

        if False:
            fig, ax = plt.subplots((mixcom+2)//3, 6, figsize=(20, 10))
            plt.subplots_adjust(left=0.02, bottom=0.03, right=0.98, top=0.97, wspace=0.18, hspace=0.08)
            for i in range(mixcom):
                pr = make_mixture_probs_train(*[v.mean() for v in surrpost_vars]).parameters["model"]["mainresult"].components_distribution[0].mean()[i]
                plot_softprob_grid(pr, ax[i // 3][2 * np.mod(i, 3)], ax[i // 3][2 * np.mod(i, 3) + 1],
                                   prefix="C" + str(i))
            plt.show()
        numMatches = min(9, pred_probs.shape[1]//2)
        if True:
            fig, ax = plt.subplots(3, 6, figsize=(20, 10))
            plt.subplots_adjust(left=0.02, bottom=0.03, right=0.98, top=0.97, wspace=0.18, hspace=0.08)
            for j in range(batch_size):
                for i in range(numMatches):
                    pr = (pred_probs[j, 2 * i] + np.dot(pred_probs[j, 2 * i + 1], home_away_inversion_matrix)) / 2
                    plot_softprob_grid(pr, ax[i // 3][2 * np.mod(i, 3)], ax[i // 3][2 * np.mod(i, 3) + 1],
                                       prefix=df_pred.Team1.iloc[j*pred_probs.shape[1] + i * 2] + " - " + df_pred.Team2.iloc[j*pred_probs.shape[1] + i * 2])
            plt.show()

        if True:
            fig, ax = plt.subplots(3, 6, figsize=(20, 10))
            plt.subplots_adjust(left=0.02, bottom=0.03, right=0.98, top=0.97, wspace=0.18, hspace=0.08)
            for j in range(batch_size):
                for i in range(numMatches):
                    pr = (smpred_probs[j, 2 * i] + np.dot(smpred_probs[j, 2 * i + 1], home_away_inversion_matrix)) / 2
                    plot_softprob_grid(pr, ax[i // 3][2 * np.mod(i, 3)], ax[i // 3][2 * np.mod(i, 3) + 1],
                                       prefix=df_smpred.Team1.iloc[j*pred_probs.shape[1] + i * 2] + " - " + df_smpred.Team2.iloc[j*pred_probs.shape[1] + i * 2])
            plt.show()

        #
        # num_results = 1000
        # num_burnin_steps = 1000
        #
        # #sampler = tfp.mcmc.TransformedTransitionKernel(
        # sampler = tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=target_log_prob_cat,
        #         step_size=tf.cast(0.1, tf.float64),
        #         num_leapfrog_steps=16)
        #         #, bijector=[tfb.Identity(), tfb.Identity(), tfb.Identity()])
        #
        # sampler_bij = tfp.mcmc.TransformedTransitionKernel(
        #     inner_kernel=sampler,
        #     bijector=[tfb.Identity(), tfb.Softplus(), tfb.Identity(), tfb.Identity()]) # tfb.Affine(shift=0.135, scale_identity_multiplier=0.01, dtype=tf.float64)
        #
        # adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        #     inner_kernel=sampler_bij,
        #     #num_adaptation_steps=int(0.8 * num_burnin_steps),
        #     num_adaptation_steps=tf.cast(0.8 * num_burnin_steps, tf.int32),
        #     target_accept_prob=tf.cast(0.75, tf.float64))
        #
        # #initial_state = [tf.ones_like(s) for s in list(joint_model.sample().values())[:3]]
        # initial_state = [#np.random.randn(x_train.shape[1]+1, mixcom), # np.zeros # weight_mean
        #                  tf.zeros(shape=[x_train.shape[1]+1, mixcom], dtype=tf.float64), # weight_mean
        #                  tf.ones(shape=[x_train.shape[1]+1, mixcom], dtype=tf.float64), # weight_scale
        #                  tf.ones(shape=[x_train.shape[1]+1, mixcom], dtype=tf.float64), # weights
        #     #np.ones(shape=[mixcom, 49]) / 49.
        #     tf.zeros(shape=[mixcom, 49], dtype=tf.float64),  # smweight
        #     #np.log(joint_model.sample()["component_distribution"]),
        #     #joint_model.sample()["component_distribution"]
        #     ]  # compdist
        #
        # s = joint_model.sample()
        # #initial_state = [s['weight_mean'].numpy(), s['weight_scale'].numpy(), s['weights'].numpy(), s['smweights'].numpy()] #, s['component_distribution'].numpy()]
        # #initial_state = [s['weight_mean'], s['weight_scale'], s['weights'], s['smweights']] #, s['component_distribution'].numpy()]
        #
        # # inputfilename = "mcmc_allstates_20200804_102700.pickle"
        # # inputfilename = "mcmc_allstates_20200805_142138.pickle"
        # # inputfilename = "mcmc_allstates_20200805_185024.pickle"
        # # inputfilename = "mcmc_allstates_20200915_110729.pickle"
        # inputfilename = sorted([f for f in os.listdir() if re.search("mcmc_opt12a_"+FLAGS.target_system+".*pickle", f)])
        # if len(inputfilename) > 0:
        #     inputfilename = inputfilename[-1]
        #     print(inputfilename)
        #     filehandler = open(inputfilename, 'rb')
        #     all_states = pickle.load(filehandler)
        #     weight_mean, weight_scale, weights, smweights = all_states
        #     initial_state = [weight_mean[-1], weight_scale[-1], weights[-1], smweights[-1]]
        #
        # [v.shape for v in initial_state]
        #
        # # v = {"component_distribution": initial_state[0],
        # # 'weight_mean': initial_state[1],
        # # 'weight_scale': initial_state[2],
        # # 'weights': initial_state[3],
        # # 'outputs': outputs_  # Y_train[:,0:2]
        # #  }
        # # print(joint_model.log_prob_parts(v))
        # # #joint_model.log_prob(*initial_state, outputs_)
        # # print(target_log_prob_cat(*initial_state))
        #
        # @tf.function(autograph=False)
        # def sample():
        #     return tfp.mcmc.sample_chain(
        #         num_results=num_results,
        #         num_burnin_steps=num_burnin_steps,
        #         current_state=initial_state,
        #         kernel=adaptive_sampler,
        #         trace_fn=lambda states, kernel_results: states
        #         )
        #
        # print(target_log_prob_cat(*initial_state))
        # print(analyse_target_log_prob_cat(*initial_state))
        #
        # # states = sample()
        # # filename = "mcmc_opt12a_"+FLAGS.target_system+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".pickle"
        # # filehandler = open(filename, 'wb')
        # # pickle.dump(states.all_states, filehandler)
        # # weight_mean, weight_scale, weights, smweights = states.all_states # , cdist
        #
        # current_state = [weight_mean[-1], weight_scale[-1], weights[-1], smweights[-1]]
        # current_state2 = [weight_mean[-10], weight_scale[-10], weights[-10], smweights[-10]]
        # print(target_log_prob_cat(*current_state))
        # print(analyse_target_log_prob_cat(*current_state))
        # tf.reduce_mean(analyse_target_log_prob_cat(*initial_state)["outputs"]-analyse_target_log_prob_cat(*current_state)["outputs"])
        # tf.reduce_mean(analyse_target_log_prob_cat(*current_state2)["outputs"]-analyse_target_log_prob_cat(*current_state)["outputs"])
        #
        def avg_points(weights, smweights, X, Y):
            mk, _ = make_joint_mixture_model(X)
            pred = mk(weights, smweights)
            achievable_points = tf.matmul(Y, tf.cast(point_matrix, tf.float32))
            actual_points = pred.mean() * tf.cast(achievable_points, tf.float64)
            actual_points = tf.reduce_sum(actual_points, axis=-1)
            return actual_points

        def real_points(weights, smweights, X, Y):
            mk, _ = make_joint_mixture_model(X)
            pred = mk(weights, smweights)
            achievable_points = tf.matmul(Y, tf.cast(point_matrix, tf.float32))
            actual_points = tf.one_hot(tf.argmax(pred.mean(), axis=-1), 49, dtype=tf.float64) * tf.cast(achievable_points, tf.float64)
            actual_points = tf.reduce_sum(actual_points, axis=-1)
            return actual_points


        if False:
            #loss_curve = [target_log_prob_cat(weight_mean[i], weight_scale[i], weights[i], smweights[i]).numpy() for i in range(weights.shape[0])]
            #loss_curve_test = [target_log_prob_cat_test(weight_mean[i], weight_scale[i], weights[i], smweights[i]).numpy() for i in range(weights.shape[0])]
            stepsize = 25
            loss_curve = [target_log_prob_cat(weight_mean[i:(i+stepsize)], weight_scale[i:(i+stepsize)], weights[i:(i+stepsize)], smweights[i:(i+stepsize)]).numpy() for i
                          in range(0, weights.shape[0], stepsize)]
            loss_curve_test = [target_log_prob_cat_test(weight_mean[i:(i+stepsize)], weight_scale[i:(i+stepsize)], weights[i:(i+stepsize)], smweights[i:(i+stepsize)]).numpy() for i in range(0, weights.shape[0], stepsize)]
            avg_points_train = np.concatenate([tf.reduce_mean(avg_points(weights[i:(i+stepsize)],
                                                                         smweights[i:(i+stepsize)], x_train_scaled, outputs_), axis=-1) for i in range(0, weights.shape[0], stepsize)], axis=0)
            avg_points_test = np.concatenate([tf.reduce_mean(avg_points(weights[i:(i+stepsize)],
                                                                        smweights[i:(i+stepsize)], x_test_scaled, test_outputs_), axis=-1) for i in range(0, weights.shape[0], stepsize)], axis=0)
            real_points_train = np.concatenate([tf.reduce_mean(real_points(weights[i:(i+stepsize)],
                                                                         smweights[i:(i+stepsize)], x_train_scaled, outputs_), axis=-1) for i in range(0, weights.shape[0], stepsize)], axis=0)
            real_points_test = np.concatenate([tf.reduce_mean(real_points(weights[i:(i+stepsize)],
                                                                        smweights[i:(i+stepsize)], x_test_scaled, test_outputs_), axis=-1) for i in range(0, weights.shape[0], stepsize)], axis=0)
            # avg_points_train = [tf.reduce_mean(avg_points(weights[i], smweights[i], x_train_scaled, outputs_)) for i in
            #  range(weights.shape[0])]
            # avg_points_test = [tf.reduce_mean(avg_points(weights[i], smweights[i], x_test_scaled, test_outputs_)) for i in
            #  range(weights.shape[0])]
            #stepsize = 10
            #loss_curve = np.concatenate([target_log_prob_cat(weight_mean[i:(i+stepsize)], weight_scale[i:(i+stepsize)], weights[i:(i+stepsize)], smweights[i:(i+stepsize)]).numpy() for i in range(0, weights.shape[0], stepsize)], axis=0)
            #loss_curve_test = np.concatenate([target_log_prob_cat_test(weight_mean[i:(i+stepsize)], weight_scale[i:(i+stepsize)], weights[i:(i+stepsize)], smweights[i:(i+stepsize)]).numpy() for i in range(0, weights.shape[0], stepsize)], axis=0)

            fig, ax = plt.subplots(6,2)
            plt.subplots_adjust(left=0.04, bottom=0.03, right=0.96, top=0.97, wspace=0.08, hspace=0.18)
            sns.kdeplot(weight_mean[-1].numpy().flatten(), ax=ax[0][0])
            sns.kdeplot(weight_scale[-1].numpy().flatten(), ax=ax[0][1])
            sns.kdeplot(weights[-1].numpy().flatten(), ax=ax[1][0])
            sns.kdeplot(smweights[-1].numpy().flatten(), ax=ax[1][1])
            ax[2][0].plot(loss_curve)
            ax[2][1].plot(loss_curve_test)
            sns.kdeplot(weight_mean[0].numpy().flatten(), ax=ax[0][0])
            sns.kdeplot(weight_scale[0].numpy().flatten(), ax=ax[0][1])
            sns.kdeplot(weights[0].numpy().flatten(), ax=ax[1][0])
            sns.kdeplot(smweights[0].numpy().flatten(), ax=ax[1][1])
            ax[3][0].plot(avg_points_train)
            ax[3][1].plot(avg_points_test)
            ax[4][0].plot(real_points_train)
            ax[4][1].plot(real_points_test)
            sns.kdeplot(avg_points_train, ax=ax[5][0])
            sns.kdeplot(avg_points_test, ax=ax[5][0])
            sns.kdeplot(real_points_train, ax=ax[5][1])
            sns.kdeplot(real_points_test, ax=ax[5][1])
            plt.show()
            #plt.close()
            fig, ax = plt.subplots(2,2)
            ax[0,0].plot(np.stack([joint_model.parameters["model"]["weight_mean"].log_prob(weight_mean[q]).numpy() for q in range(weight_scale.shape[0])]))
            ax[0,1].plot(np.stack([joint_model.parameters["model"]["weight_scale"].log_prob(weight_scale[q]).numpy() for q in range(weight_scale.shape[0])]))
            ax[1,0].plot(np.stack([joint_model.parameters["model"]["weights"](weight_scale[q], weight_mean[q]).log_prob(weights[q]).numpy() for q in range(weights.shape[0])]))
            ax[1,1].plot(np.stack([joint_model.parameters["model"]["smweights"].log_prob(smweights[q]).numpy() for q in range(weight_scale.shape[0])]))
            plt.show()

            initial_state = current_state


        #yhat = make_poisson(weights[-10])
        #df = sample_train_df(yhat)
        #plot_predictions_3(df, "poisson", "Train")
        #yhat.mean()
        #make_poisson(weights[-1]).sample().numpy()

        #sample = yhat.sample().numpy()
        if False:
            #df2 = pd.concat([create_df(create_argmax_prediction_from_mean(np.concatenate([make_mixture_probs_train(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], 5)], axis=0), Y_train.shape[0]), Y_train, "Train") for i in range(10)], axis=0)
            #df2 = pd.concat([create_df(create_argmax_prediction_from_mean(np.concatenate([make_mixture_probs_train(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], 5)], axis=0), Y_train.shape[0]), Y_train, "Train") for i in range(10)], axis=0)
            t2=np.mean(np.mean(np.array([make_mixture_probs_train(weights[i:(i+stepsize)], smweights[i:(i+stepsize)]).mean().numpy() for i in range(0, weights.shape[0], stepsize)]), axis=0), axis=0)
            df2 = create_df(create_argmax_prediction_from_mean(t2, Y_train.shape[0]), Y_train, "Train")
            #df2 = pd.concat([create_df(create_maxpoint_prediction_from_mean(np.concatenate([make_mixture_probs_train(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], 5)], axis=0), Y_train.shape[0]), Y_train, "Train") for i in range(10)], axis=0)
            #df = pd.concat([sample_categorical_df(make_mixture_probs_train(weights[w], smweights[w]), Y_train, "Train") for w in range(0, weights.shape[0])], axis=0)
            #df2 = create_maxpoint_prediction(df)
            print(df2)
            plot_predictions_3( df2, "poisson", "Train", silent=False)
            #
            # df = pd.concat([sample_categorical_df(make_mixture_probs_test(weights[w], smweights[w]), Y_test, "Test") for w in range(0, weights.shape[0])], axis=0)
            # df2 = create_maxpoint_prediction(df)
            #df2 = pd.concat([create_df(create_argmax_prediction_from_mean(np.concatenate([make_mixture_probs_test(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], 5)], axis=0), Y_test.shape[0]), Y_test, "Test") for i in range(10)], axis=0)
            t2=np.mean(np.mean(np.array([make_mixture_probs_test(weights[i:(i+stepsize)], smweights[i:(i+stepsize)]).mean().numpy() for i in range(0, weights.shape[0], stepsize)]), axis=0), axis=0)
            df2 = create_df(create_argmax_prediction_from_mean(t2, Y_test.shape[0]), Y_test, "Test")
            #df2 = pd.concat([create_df(create_maxpoint_prediction_from_mean(np.concatenate([make_mixture_probs_test(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], 5)], axis=0), Y_test.shape[0]), Y_test, "Test") for i in range(10)], axis=0)
            plot_predictions_3( df2, "poisson", "Test", silent=False)

        if False:
            make_mixture_probs_pred, joint_model_pred = make_joint_mixture_model(x_pred_scaled)
            df_pred = pd.concat([create_df(create_argmax_prediction_from_mean(np.concatenate(
                [make_mixture_probs_pred(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], weights.shape[0])],
                axis=0), Y_pred.shape[0]), Y_pred, "Pred", team1, team2) for i in range(weights.shape[0])], axis=0)
            df_pred["pred"] = df_pred.pGS.astype(str) + ":" + df_pred.pGC.astype(str)
            print(df_pred)

            m0 = df_pred.loc[df_pred.match == 0].copy()
            colour = ['gray', 'blue', 'green', 'darkorange', "yellow", "red"]
            # plt.figure()
            # for (group2Name, df2), c in zip(m0.groupby("pred"), colour):
            #     sns.kdeplot(df2["points"], shade=True, label=group2Name, color=c)
            # plt.plot()
            m0.groupby(["Team1", "Team2", "pred"]).agg({"points":"mean", "pred":"count"}).sort_values("points")
            # sns.pairplot(m0[["est1", "est2"]])
            # plt.plot()
            #for i in range(df_pred.match.nunique()):
            for i in range(18):
                print(df_pred.loc[df_pred.match == i].groupby(["Team1", "Team2", "pred"]).agg({"points":"mean", "pred":"count"}).sort_values("points"))
        #joint_model.resolve_graph()

        # df_pred = pd.concat([create_df(create_maxpoint_prediction_from_mean(np.concatenate(
        #     [make_mixture_probs_pred(weights[w], smweights[w]).mean().numpy() for w in range(i, weights.shape[0], 300)],
        #     axis=0), Y_pred.shape[0]), Y_pred, "Pred", team1, team2) for i in range(300)], axis=0)
        # df_pred["pred"] = df_pred.pGS.astype(str) + ":" + df_pred.pGC.astype(str)
        #
        # for i in range(18):
        #     print(df_pred.loc[df_pred.match == i].groupby(["Team1", "Team2", "pred"]).agg({"points":"mean", "pred":"count"}).sort_values("points"))


        if False:
            w=-1
            print(make_mixture_probs_train(weights[w], smweights[w]).mean())
            print(make_mixture_probs_train(weights[w], smweights[w]).mixture_distribution.prob(0))
            print(make_mixture_probs_train(weights[w], smweights[w]).mixture_distribution.prob(1))
            #print(make_mixture_probs_train(cdist[w], weights[w], smweights[w]).mixture_distribution.prob(2))
            #print(make_mixture_probs_train(cdist[w], weights[w], smweights[w]).mixture_distribution.prob(3))
            print(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean())
            print(np.sort(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean()))
            print(np.sum(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean(), axis=1))
            print(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean() / (np.sum(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean(), axis=1, keepdims=True)))
            w=-2
            print(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean() / (np.sum(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean(), axis=1, keepdims=True)))

            print(make_mixture_probs_train(weights[w], smweights[w]).components_distribution.sample())

        if False:
            w = tf.transpose(weights, (1, 2, 0))
            w = tf.reshape(w, (x_train.shape[1] + 1, -1))
            b = tfb.Reshape(event_shape_out=(-1, d, weights.shape[0]), event_shape_in=(-1, d * weights.shape[0]))
            b2 = tfb.Transpose((2, 0, 1))
            b3 = tfb.Reshape(event_shape_out=(-1, d), event_shape_in=[weights.shape[0], -1, d])
            b4 = b3(b2(b))

            #a = make_poisson_test(w)
            df = sample_categorical_df(b4(make_poisson_test(w)), Y_test[:,0:2], "Test")
            df2 = create_maxpoint_prediction(df)
            plot_predictions_3( df2, "poisson", "Test", silent=True)
            del df
            del df2
            df = sample_categorical_df(b4(make_poisson_train(w)), Y_train[:,0:2], "Train")
            df2 = create_maxpoint_prediction(df)
            plot_predictions_3( df2, "poisson", "Train", silent=True)
            del df
            del df2


        # fig, ax = plt.subplots(4, 4, figsize=(8, 8))
        # for i in range(4):
        #     for j in range(4):
        #         sns.boxplot(y=model(x_train_scaled).mean()[:,4*i+j], x=Y_train[:,4*i+j], ax=ax[i][j])
        # plt.show()
        #
        # fig, ax = plt.subplots(4, 4, figsize=(8, 8))
        # for i in range(4):
        #     for j in range(4):
        #         sns.boxplot(y=model(x_test_scaled).mean()[:,4*i+j], x=Y_test[:,4*i+j], ax=ax[i][j])
        # plt.show()
        #
        # def make_wdl(x):
        #     return np.sign(x[:,0]-x[:,1])
        # def make_wl(x):
        #     return np.sign(x.mean()[:,0]-x.mean()[:,1])
        # wdl_train = make_wdl(Y_train)
        # wdl_test = make_wdl(Y_test)
        # sample_train = make_wdl(model(x_train_scaled))
        # sample_test = make_wdl(model(x_test_scaled))
        # sample_train = make_wl(model(x_train_scaled))
        # sample_test = make_wl(model(x_test_scaled))
        #
        # print(Counter(sample_test==wdl_test))
        # print(Counter(sample_train == wdl_train))
        # print(np.sum(sample_test==wdl_test)/len(wdl_test))
        # print(np.sum(sample_train==wdl_train)/len(wdl_train))
        # print(confusion_matrix(sample_test, wdl_test))
        # print(confusion_matrix(sample_train ,  wdl_train))
        #
        # arraygs = np.array([i // 7 for i in range(49)])[np.newaxis, ...]
        # arraygc = np.array([np.mod(i, 7) for i in range(49)])[np.newaxis, ...]
        #
        # def make_train_sample():
        #     yhat = model(x_train_scaled)
        #     sample = yhat.sample().numpy()
        #     df = pd.DataFrame({"Where":np.tile(["Home", "Away"], reps=len(wdl_train)//2),
        #                        "GS":Y_train[:,0].astype(int),
        #                        "GC":Y_train[:,1].astype(int),
        #                        "pGS":sample[:,0].astype(int),
        #                        "pGC":sample[:,1].astype(int),
        #                        "est1":yhat.mean()[:,0],
        #                        "est2":yhat.mean()[:,1],
        #                            # "pGS":np.argmax(sample, axis=1)//7,
        #                        # "pGC":np.mod(np.argmax(sample, axis=1), 7),
        #                        # "est1":np.sum(yhat.mean() * arraygs, axis=1),
        #                        # "est2":np.sum(yhat.mean() * arraygc, axis=1),
        #                        "Prefix": "poisson",
        #                        "dataset": "Train",
        #                        "act":"Y_train[:,0:2]",
        #                        "Team1":"Team1",
        #                        "Team2": "Team2",
        #                        "match":range(len(wdl_train))
        #                        })
        #     return df
        #
        # def make_test_sample():
        #     yhat = model(x_test_scaled)
        #     sample = yhat.sample().numpy()
        #     df = pd.DataFrame({"Where":np.tile(["Home", "Away"], reps=len(wdl_test)//2),
        #                        "GS":Y_test[:,0].astype(int),
        #                        "GC":Y_test[:,1].astype(int),
        #                        "pGS":sample[:,0].astype(int),
        #                        "pGC":sample[:,1].astype(int),
        #                        "est1":yhat.mean()[:,0],
        #                        "est2":yhat.mean()[:,1],
        #                        # "pGS":np.argmax(sample, axis=1)//7,
        #                        # "pGC":np.mod(np.argmax(sample, axis=1), 7),
        #                        # "est1":np.sum(yhat.mean() * arraygs, axis=1),
        #                        # "est2":np.sum(yhat.mean() * arraygc, axis=1),
        #                        "Prefix": "poisson",
        #                        "dataset": "Test",
        #                        "act":"Y_train[:,0:2]",
        #                        "Team1":"Team1",
        #                        "Team2": "Team2",
        #                        "match":range(len(wdl_test))
        #                        })
        #     return df
        #
        #
        # df = pd.concat([make_test_sample() for _ in range(200)], axis=0)
        # plot_predictions_3( create_maxpoint_prediction(df), "poisson", "Test", )
        #
        # df = pd.concat([make_train_sample() for _ in range(200)], axis=0)
        # plot_predictions_3( create_maxpoint_prediction(df), "poisson", "Train")
        #



    if False:
        w = -1
        for i in range(mixcom):
        #for i in range(4):
            plot_softprob_simple(make_mixture_probs_train(weights[w], smweights[w]).components_distribution[0].mean()[i], title="component "+str(i), prefix="component "+str(i))


    if False:
        pred_probs = np.mean(make_mixture_probs_pred(weights, smweights).mean().numpy(), axis=0)
        for i in range(18):
            plot_softprob_simple(pred_probs[i], prefix=df_pred.Team1.iloc[i]+" - "+df_pred.Team2.iloc[i])

    if False:
        pred_probs = np.mean(make_mixture_probs_pred(weights, smweights).mean().numpy(), axis=0)
        fig, ax = plt.subplots(3, 6, figsize=(20, 10))
        plt.subplots_adjust(left=0.02, bottom=0.03, right=0.98, top=0.97, wspace=0.18, hspace=0.08)
        for i in range(9):
            pr = (pred_probs[2*i] + np.dot(pred_probs[2*i+1], home_away_inversion_matrix))/2
            plot_softprob_grid(pr, ax[i//3][2*np.mod(i,3)], ax[i//3][2*np.mod(i,3)+1], prefix=df_pred.Team1.iloc[i*2] + " - " + df_pred.Team2.iloc[i*2])
        plt.show()

    if True:
        plot_softprob_simple(np.mean(outputs_[::2], axis=0), prefix="Train Seasons Summary")
        plot_softprob_simple(np.mean(test_outputs_[::2], axis=0), prefix="Test Seasons Summary")

        print("Best points - Train")
        print(np.dot(np.mean(outputs_[::2], axis=0), point_matrix).reshape((7,7)).transpose()[::-1])
        print("Best points - Test")
        print(np.dot(np.mean(test_outputs_[::2], axis=0), point_matrix).reshape((7,7)).transpose()[::-1])

