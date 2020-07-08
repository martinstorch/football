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
import csv
import pandas as pd

pd.set_option('expand_frame_repr', False)

import numpy as np

# np.set_printoptions(threshold=50)
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

# @ops.RegisterGradient("BernoulliSample_ST")
# def bernoulliSample_ST(op, grad):
#    return [tf.clip_by_norm(grad, 20.0), tf.zeros(tf.shape(op.inputs[1]))]


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


def plot_predictions_3(df, prefix, dataset, silent=False):
    df = df.loc[(df.Prefix == prefix) & (df.dataset == dataset)].copy()
    if not silent:
        fig = plt.figure(figsize=(18, 4))
        ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=2, rowspan=1)
        ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=1, rowspan=1)
        ax2.axis('off')
        ax3 = plt.subplot2grid((1, 4), (0, 3), colspan=1, rowspan=1)
        ax3.axis('off')

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
    t1.columns = ["None", "Tendency", "Diff", "Full"][:len(t1.columns)]
    t1.index = t1.index.droplevel(level=0)
    if not silent:
        t1.plot(kind='bar', stacked=True, ax=ax1)
        ax1.set_title("{}".format(prefix)).set_size(15)

        _, _, autotexts = ax2.pie(pie_chart_values,
                                  labels=["None", "Tendency", "Diff", "Full"], autopct='%1.1f%%', startangle=90)
        for t in autotexts:
            t.set_color("white")
        ax2.set_title("{}: {:.04f} ({})".format(prefix, avg_points, dataset)).set_size(20)

        percentages = [pie_chart_values[0],  # None
                       pie_chart_values[1] + pie_chart_values[2] + pie_chart_values[3],  # Tendency
                       pie_chart_values[2] + pie_chart_values[3],  # GDiff
                       pie_chart_values[3],  # Full
                       ]
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


def dispatch_main(target_distr, model_dir, train_steps, train_data, test_data,
                  checkpoints, save_steps, data_dir, max_to_keep,
                  reset_variables, skip_plotting, target_system, modes, use_swa, histograms, useBWIN):
    """Train and evaluate the model."""
    # train_file_name, test_file_name, teamnames = maybe_download(train_data, test_data)
    # Specify file path below if want to find the output easily
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print(model_dir)

    global point_scheme
    if target_system == "TCS":
        point_scheme = point_scheme_tcs
    elif target_system == "Pistor":
        point_scheme = point_scheme_pistor
    elif target_system == "Sky":
        point_scheme = point_scheme_sky
    elif target_system == "GoalDiff":
        point_scheme = point_scheme_goal_diff
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

    return model_data, point_scheme, label_column_names


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
                         FLAGS.save_steps, FLAGS.data_dir, FLAGS.max_to_keep,
                         FLAGS.reset_variables, FLAGS.skip_plotting, FLAGS.target_system,
                         FLAGS.action, FLAGS.swa, FLAGS.histograms,
                         FLAGS.useBWIN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--skip_plotting", type=bool,
        default=True,
        # default=False,
        help="Print plots of predicted data"
    )
    parser.add_argument(
        "--histograms", type=bool,
        # default=True,
        default=False,
        help="create histogram data in summary files"
    )
    parser.add_argument(
        "--train_steps", type=int,
        default=2000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--save_steps", type=int,
        # default=200,
        default=500,
        help="Number of training steps between checkpoint files."
    )
    parser.add_argument(
        "--reset_variables", type=str,  # nargs='+',
        default="cbsp",
        # default=300,
        help="List of variable names to be re-initialized during upgrade"
    )
    parser.add_argument(
        "--max_to_keep", type=int,
        default=500,
        help="Number of checkpoint files to keep."
    )
    parser.add_argument(
        "--train_data", type=str,
        # default="0910,1112,1314,1516,1718,1920", #
        # default="1314,1415,1516,1617,1718,1819,1920", #
        #default="0910,1011,1112,1213,1314",
        default="0910,1011,1112,1213,1314,1415,1516,1617,1718", #
        #default="1617,1718",  #
        # default="1819,1920",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data", type=str,
        # default="1011,1213,1415,1617,1819", #
        # default="0910,1011,1112,1213", #
        # default="1617,1718", #
        default="1819,1920",
        help="Path to the test data."
    )
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
        "--target_system",
        type=str,
        #default="Pistor",
        default="Sky",
        # default="TCS",
        # default="GoalDiff",
        help="Point system to optimize for"
    )
    parser.add_argument(
        "--swa", type=bool,
        # default=True,
        default=False,
        help="Run in Stochastic Weight Averaging mode."
    )
    parser.add_argument(
        "--useBWIN", type=bool,
        default=True,
        # default=False,
        help="Run in Stochastic Weight Averaging mode."
    )
    parser.add_argument(
        "--action",
        type=str,
        # default="static",
        default="train",
        # default="eval_stop",
        # default="eval",
        # default="predict",
        # default="upgrade",
        # default="train_eval",
        # default="upgrade,train,eval,predict",
        help="What to do"
    )
    parser.add_argument(
        "--checkpoints", type=str,
        # default="560000:",
        # default="60000:92000",
        default="-1",  # slice(-2, None)
        # default="100:",
        # default="",
        help="Range of checkpoints for evaluation / prediction. Format: "
    )
    FLAGS, unparsed = parser.parse_known_args()
    print([sys.argv[0]] + unparsed)
    print(FLAGS)
    model_data, point_scheme, label_column_names = main([sys.argv[0]] + unparsed)

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

    X = features_arrays["match_input_layer"]
    label_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 46, 47, 48, 49, 50, 51, 52, 53]
    X = np.concatenate([X,
                        np.mean(labels_array[features_arrays["match_history_t1"][:, -5:]][:, :, label_filter], axis=1),
                        np.mean(labels_array[features_arrays["match_history_t2"][:, -5:]][:, :, label_filter], axis=1),
                        np.mean(labels_array[features_arrays["match_history_t12"][:, -2:]][:, :, label_filter], axis=1)
                        # labels_array[features_arrays["match_history_t2"][:,-5:]][:,:,label_filter].reshape((-1,5*len(label_filter))),
                        # labels_array[features_arrays["match_history_t12"][:,-2:]][:,:,label_filter].reshape((-1,2*len(label_filter)))
                        ], axis=1)

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
    feature_names += label_column_names_extended

    print(X.shape)

    #  train_idx = train_idx[::2]
    #  test_idx = test_idx[::2]
    #  pred_idx = pred_idx[::2]

    X_train = X[train_idx]
    X_test = X[test_idx]
    X_pred = X[pred_idx]
    bwin_index = [1, 23, 24, 25]
    X_train_bwin = np.take(X[train_idx], bwin_index, axis=1)
    X_test_bwin = np.take(X[test_idx], bwin_index, axis=1)
    X_pred_bwin = np.take(X[pred_idx], bwin_index, axis=1)

    # spi_index = [210, 211, 212, 213, 214, 215, 216, 217, 218]
    spi_index = [1, 214, 215, 216, 217, 218]
    X_train_spi = np.take(X[train_idx], spi_index, axis=1)
    X_test_spi = np.take(X[test_idx], spi_index, axis=1)
    X_pred_spi = np.take(X[pred_idx], spi_index, axis=1)

    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors
    tfk = tfp.math.psd_kernels

    if tf.test.gpu_device_name() != '/device:GPU:0':
        print('WARNING: GPU device not found.')
    else:
        print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

    # For numeric stability, set the default floating-point dtype to float64
    tf.keras.backend.set_floatx('float64')


    #feature_idx = list(range(0,30))
    #feature_idx = bwin_index
    feature_idx = range(X_train.shape[1])

    Y_train = labels_array[train_idx, 0:16]
    Y_test = labels_array[test_idx, 0:16]

    x_train = X_train[:, feature_idx].astype(np.float64)
    x_test = X_test[:, feature_idx].astype(np.float64)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
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

    def create_maxpoint_prediction(df):
        dfpoints = np.stack([point_matrix[np.maximum(0, np.minimum(np.round(df.pGS), 6)) * 7 + np.maximum(0, np.minimum(np.round(df.pGC), 6)), i] for i in range(49)],
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


    n_train_samples = x_train.shape[0]

    d = Y_train.shape[1]

    #@tf.function(autograph=False)
    # def make_poisson(weights):
    #     outputs = tf.matmul(x_train_scaled, weights[:-1])+weights[-1:]
    #     return tfd.Independent(tfd.Poisson(tf.math.softplus(outputs)), reinterpreted_batch_ndims=2)
    #
    # def make_poisson_test(weights):
    #     outputs = tf.matmul(x_test_scaled, weights[:-1])+weights[-1:]
    #     return tfd.Independent(tfd.Poisson(tf.math.softplus(outputs)), reinterpreted_batch_ndims=2)

    def make_joint_model(x):
        def make_poisson(weights):
            outputs = tf.matmul(x, weights[:-1]) + weights[-1:]
            return tfd.Independent(tfd.Poisson(tf.math.softplus(outputs)), reinterpreted_batch_ndims=2)

        return make_poisson, tfd.JointDistributionNamed(dict(
            weight_mean=tfd.Independent(tfd.Normal(loc=np.zeros(shape=[x.shape[1] + 1, d]),
                                                   scale=np.ones(shape=[x.shape[1] + 1, d])),
                                        reinterpreted_batch_ndims=2),
            weight_scale=tfd.Independent(
                tfd.LogNormal(loc=np.zeros(shape=[x.shape[1] + 1, d]),
                              scale=np.ones(shape=[x.shape[1] + 1, d])), reinterpreted_batch_ndims=2),
            weights=lambda weight_scale, weight_mean: tfd.Independent(
                tfd.Normal(loc=weight_mean, scale=1.0 * weight_scale), reinterpreted_batch_ndims=2),
            outputs=make_poisson
        ))

    make_poisson_train, joint_model = make_joint_model(x_train_scaled)
    make_poisson_test, joint_model_test = make_joint_model(x_test_scaled)

    # joint_model
    # joint_model.sample()
    # joint_model.log_prob(joint_model.sample())
    # joint_model.mean()
    # {s:v.shape for s,v in joint_model.sample().items()}

    def target_log_prob(weight_mean, weight_scale, weights):
        return joint_model.log_prob({
            'weight_mean': weight_mean,
            'weight_scale': weight_scale,
            'weights': weights,
            'outputs': Y_train
        })

    def target_log_prob_test(weight_mean, weight_scale, weights):
        return joint_model_test.log_prob({
            'weight_mean': weight_mean,
            'weight_scale': weight_scale,
            'weights': weights,
            'outputs': Y_test
        })


    num_results = 500
    num_burnin_steps = 1000

    #sampler = tfp.mcmc.TransformedTransitionKernel(
    sampler = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=tf.cast(0.1, tf.float64),
            num_leapfrog_steps=10)
            #, bijector=[tfb.Identity(), tfb.Identity(), tfb.Identity()])

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=tf.cast(0.75, tf.float64))

    #initial_state = [tf.ones_like(s) for s in list(joint_model.sample().values())[:3]]
    initial_state = [np.zeros(shape=[x_train.shape[1]+1, d]),
                     np.ones(shape=[x_train.shape[1]+1, d]), np.random.randn(x_train.shape[1]+1, d)]

    @tf.function(autograph=False)
    def sample():
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=adaptive_sampler,
            trace_fn=lambda states, kernel_results: states[0]
            )

    states = sample()
    #states, kernel_results = r
    #(kernel_results)
    weight_mean, weight_scale, weights = states.all_states
    current_state = [weight_mean[-1], weight_scale[-1], weights[-1]]
    print(target_log_prob(*initial_state))
    print(target_log_prob(*current_state))

    loss_curve = [target_log_prob(weight_mean[i], weight_scale[i], weights[i]) for i in range(weights.shape[0])]
    loss_curve_test = [target_log_prob_test(weight_mean[i], weight_scale[i], weights[i]) for i in range(weights.shape[0])]
    fig, ax = plt.subplots(2,2)
    sns.kdeplot(weight_mean[-1].numpy().flatten(), ax=ax[0][0])
    sns.kdeplot(weight_scale[-1].numpy().flatten(), ax=ax[0][1])
    sns.kdeplot(weights[-1].numpy().flatten(), ax=ax[1][0])
    ax[1][1].plot(loss_curve)
    secaxy = ax[1][1].twinx()
    secaxy.plot(loss_curve_test, c="r")
    sns.kdeplot(weight_mean[0].numpy().flatten(), ax=ax[0][0])
    sns.kdeplot(weight_scale[0].numpy().flatten(), ax=ax[0][1])
    sns.kdeplot(weights[0].numpy().flatten(), ax=ax[1][0])
    plt.show()
    #plt.close()

    initial_state = current_state

    #yhat = make_poisson(weights[-10])
    #df = sample_train_df(yhat)
    #plot_predictions_3(df, "poisson", "Train")
    #yhat.mean()
    #make_poisson(weights[-1]).sample().numpy()

    #sample = yhat.sample().numpy()

    df = pd.concat([sample_df(make_poisson_train(weights[w]), Y_train, "Train") for w in range(0, weights.shape[0])], axis=0)
    df2 = create_maxpoint_prediction(df)
    plot_predictions_3( df2, "poisson", "Train")
    #
    df = pd.concat([sample_df(make_poisson_test(weights[w]), Y_test, "Test") for w in range(0, weights.shape[0])], axis=0)
    df2 = create_maxpoint_prediction(df)
    plot_predictions_3( df2, "poisson", "Test", silent=True)

    joint_model.resolve_graph()

    w = tf.transpose(weights, (1,2,0))
    w = tf.reshape(w, (x_train.shape[1]+1, -1))
    b = tfb.Reshape(event_shape_out=(-1, d, weights.shape[0]), event_shape_in=(-1, d * weights.shape[0]))
    b2 = tfb.Transpose((2,0,1))
    b3 = tfb.Reshape(event_shape_out=(-1, d), event_shape_in=[weights.shape[0], -1, d])
    b4 = b3(b2(b))

    #a = make_poisson_test(w)
    df = sample_df(b4(make_poisson_test(w)), Y_test, "Test")
    df2 = create_maxpoint_prediction(df)
    plot_predictions_3( df2, "poisson", "Test", silent=True)
    del df
    del df2
    df = sample_df(b4(make_poisson_train(w)), Y_train, "Train")
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
