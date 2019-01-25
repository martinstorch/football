# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:39:42 2019

@author: 811087
"""

import os
import tensorflow as tf

model_dir = "D:/Models/simple36_pistor_1819_2"
tf.reset_default_graph()

checkpoint = model_dir + "/model.ckpt-34000"
all_checkpoints = ["34000", "34200", "34400", "34600", "34800", "35000", "35200", "35400"]
new_global_step = 35600
all_checkpoints = ["26200", "24200", "27800", "28200", "31200"]
new_global_step = 28000
# Add ops to save and restore all the variables.
tf.reset_default_graph()
print("reading meta-graph ...")
saver = tf.train.import_meta_graph(checkpoint+'.meta')
print("done")
graph = tf.get_default_graph()

model_vars = tf.trainable_variables()
print(model_vars)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print(update_ops)    
# Add SWA algorithm

tf.reset_default_graph()

averages = [] # [[]]*len(model_vars)
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
    else:
      averages = [a+v for a,v in zip(averages, values)]
  averages = [a/len(all_checkpoints) for a in averages]     
  #print(averages[1])  
  feed_dict = {plh:a for plh,a in zip(placeholder, averages)}
  swa_to_weights = tf.group(*(tf.assign(var, avg) for var, avg in zip(model_vars, placeholder)))
  sess.run(swa_to_weights, feed_dict=feed_dict)
  values = sess.run(model_vars)
  #print(values[1])
  print("saving model ...")
  directory = model_dir+'/swa'  
  if not os.path.exists(directory):
    print("creating ", directory)  
    os.makedirs(directory)
  saver.save(sess, model_dir+'/swa/model', global_step=new_global_step, strip_default_attrs=False) 
  print("done")

    
    
#  
#    with tf.name_scope('SWA'):
#        #swa = StochasticWeightAveraging()
#        #swa_op = swa.apply(var_list=model_vars)
#        # Make backup variables
#        with tf.variable_scope('BackupVariables', reuse=True):
#          with tf.variable_scope(cp):
#            backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
#                                           initializer=var.initialized_value())
#                           for var in model_vars]
#            #print(backup_vars)
#            sess.run(tf.variables_initializer(backup_vars))
#            vars_dict[cp]=backup_vars
#            averages = [a+[b] for a,b in zip(averages, backup_vars)]
#    
#  average_nodes = [tf.add_n(a)/len(a) for a in averages]
#  print(average_nodes)
#  swa_to_weights = tf.group(*(tf.assign(var, avg) for var, avg in zip(model_vars, average_nodes)))
#  #tf.global_variables_initializer()
#  sess.run(swa_to_weights)
#  print(testnode)
#  print(sess.run(testnode))
#  saver.save(sess, model_dir+'/swa_model', global_step=0) 


for e in tf.train.summary_iterator(model_dir+"/eval_test/events.out.tfevents.1548406606.14HW010662"):
  print(e.step, e.wall_time)  
  for v in e.summary.value:
      if "global" in v.tag:
        print(v.tag, v.simple_value)

for e in tf.train.summary_iterator(model_dir+"/evaluation_test/events.out.tfevents.1548406489.14HW010662"):
  print(e.step, e.wall_time)  
  for v in e.summary.value:
      if "z_points" in v.tag:
        print(v.tag, v.simple_value)
for e in tf.train.summary_iterator(model_dir+"/eval/events.out.tfevents.1548319287.14HW010662"):
  print(e.step, e.wall_time, e.file_version, e.log_message.level, e.session_log.status   )  
  for v in e.summary.value:
      if "summary" in v.tag:
        print(e.step, v.tag, v.simple_value)


import pandas as pd
file = model_dir+"/eval_test/events.out.tfevents.1548406606.14HW010662"
kv = [(e.step, v.tag, v.simple_value) for e in tf.train.summary_iterator(file) for v in e.summary.value if "summary" in v.tag]
df = pd.DataFrame(kv)  
df = df.pivot(index=0, columns=1, values=2)
df.columns
df.index
df.describe()
df.mean()
df.std()
