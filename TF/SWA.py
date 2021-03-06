# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:39:42 2019

@author: 811087
"""

import os
import tensorflow as tf
import pandas as pd
pd.set_option('expand_frame_repr', False)
import numpy as np

#model_dir = "D:/Models/simple_test"
#model_dir = "C:/Models/simple36_sky_1819_3"
#model_dir = "C:/Models/simple36_pistor_1819_2"

eval_dir = "c:/Models/conv1_auto_sky5/eval_test"
model_dir = os.path.abspath(os.path.join(eval_dir, os.pardir))

event_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.startswith('events.out.tfevents.')]

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

all_checkpoints = find_checkpoints_in_scope(model_dir, "-5", True)
print(all_checkpoints)
new_global_step = all_checkpoints.global_step.max()
all_checkpoints = all_checkpoints.global_step.astype(str).tolist()
print(all_checkpoints)

#all_checkpoints = ["34000", "34200", "34400", "34600", "34800", "35000", "35200", "35400"]
#new_global_step = 35600
#all_checkpoints = ["26200", "24200", "27800", "28200", "31200"]
#all_checkpoints = ["14400", "12200", "27800", "28200", "31200"]
#all_checkpoints = ["7011", "6811", "6610", "6009", "5207"]
#new_global_step = 6500
#
#file = model_dir+"/eval/events.out.tfevents.1548700176.DESKTOP-VPR2GCA"
#file = model_dir+"/eval_test/events.out.tfevents.1549392619.DESKTOP-VPR2GCA"
#file = model_dir+"/eval/events.out.tfevents.1549018157.14HW010662"

all_scores = []
for file in event_files:
  kv = [(e.step, v.tag[8:], v.simple_value) for e in tf.train.summary_iterator(file) for v in e.summary.value if "summary" in v.tag and "z_point" in v.tag]
  all_scores.extend(kv)


df = pd.DataFrame(all_scores).drop_duplicates()
#df = df.loc[df[0]>15000]
df = df.groupby([0,1]).mean().reset_index()  # remove duplicates in index
df = df.pivot(index=0, columns=1, values=2)
#df = df.loc[df["sp/z_points"]<1.0]
#df = df.loc[df.index<9000]
df.columns
df.index
df.describe()
df["mean"]=np.mean(df.values, axis=1)
print(df.agg(["mean", "std"]).T)
x = df.sort_values(["mean"]).tail(10)
x = df.sort_values(["sp/z_points", "cp/z_points"]).tail(10)
print(x)
print(x.index)

#all_checkpoints = ['23108','23309']
#new_global_step = 34555
all_checkpoints = [str(s) for s in sorted(list(x.index))]
new_global_step = np.max(list(x.index))
print(all_checkpoints)
print(new_global_step)

#all_checkpoints = all_checkpoints[-4:]


checkpoint = model_dir + "/model.ckpt-"+str(new_global_step)
# Add ops to save and restore all the variables.
tf.reset_default_graph()
print("reading meta-graph ...")
saver = tf.train.import_meta_graph(checkpoint+'.meta')
print("done")
graph = tf.get_default_graph()

model_vars = tf.trainable_variables()
print(model_vars)

# Add SWA algorithm

tf.reset_default_graph()
averages = [] # [[]]*len(model_vars) 
weights = [] 
with tf.Session(graph=graph) as sess: 
  
  placeholder=[tf.placeholder(dtype=v.dtype, shape=v.shape) for v in model_vars] 
  print("placeholder: ", placeholder) 

  for cp in all_checkpoints: 
    checkpoint = model_dir + "/model.ckpt-"+cp 
    saver.restore(sess, checkpoint) 
    values = sess.run(model_vars) 
    #print(values[1]) 
    if averages==[]: 
      averages = values 
      weights = [[v] for v in values] 
    else: 
      averages = [a+v for a,v in zip(averages, values)] 
      for w,v in zip(weights, values): 
        w.append(v) 
  averages = [a/len(all_checkpoints) for a in averages]     
  feed_dict = {plh:a for plh,a in zip(placeholder, averages)} 
  swa_to_weights = tf.group(*(tf.assign(var, avg) for var, avg in zip(model_vars, placeholder))) 
  sess.run(swa_to_weights, feed_dict=feed_dict) 
  values = sess.run(model_vars) 
  print("saving model ...") 
  directory = model_dir+'/swa'   
  if not os.path.exists(directory): 
    print("creating ", directory)   
    os.makedirs(directory) 
  saver.save(sess, model_dir+'/swa/model', global_step=new_global_step, strip_default_attrs=False) 
  print("done") 

ws = [np.stack(w, axis=0) for w in weights] 
means = [np.mean(w, axis=0) for w in ws]   
std = [np.std(w, axis=0) for w in ws]   
ratio =  [np.std(w, axis=0)/(np.mean(np.absolute(w), axis=0)+1e-36) for w in ws]   

pd.DataFrame({"name":[v.name.replace("rnn/multi_rnn_cell/cell_","") for v in model_vars], 
              "size":[np.prod(r.shape) for r in ratio], 
              "mean":[np.mean(r) for r in ratio], 
              "std":[np.std(r) for r in ratio], 
              "min":[np.min(r) for r in ratio], 
              "max":[np.max(r) for r in ratio], 
              "absmean":[np.mean(np.absolute(w)) for w in weights], 
              }).sort_values("mean") 

#checkpoint = 'D:/Models/conv1_auto_pistor\model.ckpt-27677'  
#from tensorflow.python.tools import inspect_checkpoint 
#tensors = inspect_checkpoint.print_tensors_in_checkpoint_file(file_name=checkpoint, tensor_name='',all_tensors=True) 
    

#for e in tf.train.summary_iterator(model_dir+"/eval_test/events.out.tfevents.1548406606.14HW010662"):
#  print(e.step, e.wall_time)  
#  for v in e.summary.value:
#      if "global" in v.tag:
#        print(v.tag, v.simple_value)
#
#for e in tf.train.summary_iterator(model_dir+"/eval_test/events.out.tfevents.1549618562.14HW010662"):
#  print(e.step, e.wall_time)  
#  for v in e.summary.value:
#      if True or "z_points" in v.tag:
#        print(v.tag, v.simple_value)
#for e in tf.train.summary_iterator(model_dir+"/eval/events.out.tfevents.1548319287.14HW010662"):
#  print(e.step, e.wall_time, e.file_version, e.log_message.level, e.session_log.status   )  
#  for v in e.summary.value:
#      if "summary" in v.tag:
#        print(e.step, v.tag, v.simple_value)
#
#
#file = model_dir+"/eval_test/events.out.tfevents.1548406606.14HW010662"
#file = model_dir+"/eval/events.out.tfevents.1548445101.DESKTOP-VPR2GCA"
#file = model_dir+"/eval/events.out.tfevents.1548449896.DESKTOP-VPR2GCA"
#
#
#print(df.mean())
#print(df.std())

# Reassign "CNN" variables to new checkpoint
  
model_vars = tf.trainable_variables()
print(model_vars)
scope = ["CNN"]

vars_in_scope = [v for v in model_vars if any(s in v.name for s in scope)]      
print(vars_in_scope)

source = 7232
target = 34555

tf.reset_default_graph()
with tf.Session(graph=graph) as sess: 
  
  placeholder=[tf.placeholder(dtype=v.dtype, shape=v.shape) for v in vars_in_scope] 
  print("placeholder: ", placeholder) 

  checkpoint = model_dir + "/model.ckpt-"+str(source)
  saver.restore(sess, checkpoint) 
  values = sess.run(vars_in_scope) 

  checkpoint = model_dir + "/model.ckpt-"+str(target)
  saver.restore(sess, checkpoint) 

  feed_dict = {plh:v for plh,v in zip(placeholder, values)} 
  assign_new = tf.group(*(tf.assign(var, value) for var, value in zip(vars_in_scope, placeholder))) 

  sess.run(assign_new, feed_dict=feed_dict) 
  new_values = sess.run(model_vars) 
  print("saving model ...") 
  saver.save(sess, checkpoint) 
  print("done") 
  