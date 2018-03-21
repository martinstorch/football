# -*- coding: utf-8 -*-
import tensorflow as tf

#class LinearModel(object):
#  @property 
#  estimator_model 
#  
#  def __init__(self, model_dir, columns):

def create_estimator(model_dir, columns):    
    
  def model(features, labels, mode):
        loss = None
        train = None
        eval_metric_ops = None
        export_outputs = None
        accuracy1 = tf.ones(shape=[tf.shape(features["HomeTeam"])[0]])
        confmatrix = features["HomeTeam"]
        # Build a linear model and predict values
#        print(features)
#        print(labels)
#        print(columns)
        
        X = tf.feature_column.input_layer(features, columns)
        with tf.variable_scope("Linear"):
          W = tf.get_variable("W", shape=[X.shape[1],1], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
          b = tf.get_variable("b", shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
          y = tf.matmul(X,W) + b
        #y = tf.feature_column.linear_model(features, columns)
        y_ = tf.cast(tf.round(y), tf.int64)
        
        #print(y)
        #print(labels)
        print(mode)
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
          #loss = tf.reduce_mean(tf.square(y[:,0] - tf.to_float(labels)),0)
          loss = tf.losses.mean_squared_error(y[:,0] , labels)
          # Summaries to display for TRAINING and TESTING
          tf.summary.scalar("loss", loss)    
          confmatrix = tf.confusion_matrix(labels, y_)
          accuracy0 = (tf.reduce_mean(tf.cast(tf.equal(labels, y_), tf.float32)),tf.no_op())
          accuracy1 = tf.cast(tf.equal(labels, y_), tf.float32)
          accuracy = tf.metrics.accuracy(labels, y_)  
          
          print(accuracy0)
          print(accuracy)
          eval_metric_ops = {
              "accuracy": tf.metrics.mean(accuracy1)
              , "mae": tf.metrics.mean_absolute_error(labels=tf.cast(labels, tf.float32), predictions=y)
              #,"confusion": (confmatrix, tf.no_op())
              , "accuracy2": accuracy
          }


        predictions = {
          "predictions":  y ,
          "estimations":  y_ 
#          , "confusion": confmatrix
#          , "accuracy": accuracy1
#          "weights": W[:,0], 
#          "bias": b[:],
        }
        #tf.summary.image("X", tf.reshape(tf.random_normal([10, 10]), [-1, 10, 10, 1])) # dummy, my inputs are images
        with tf.Session() as sess:
          sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    
        # Training sub-graph
        if mode == tf.estimator.ModeKeys.TRAIN:
          global_step = tf.train.get_global_step()
          optimizer = tf.train.GradientDescentOptimizer(0.01)
          with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
          train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
          
        
        print(model_dir)
        
        export_outputs = {
            "predictions": tf.estimator.export.RegressionOutput(y)
            #, "estimations": tf.estimator.export.RegressionOutput(tf.cast(y_, tf.float32))
        }
          
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss= loss, train_op=train
                                          #,scaffold = tf.train.Scaffold(summary_op = tf.summary.merge_all())
                                          #, training_hooks=[tf.train.SummarySaverHook(save_steps=10,output_dir='C:/tmp/Football/models/train',summary_op = tf.summary.merge_all())]
                                          #, evaluation_hooks=[tf.train.SummarySaverHook(save_steps=10,output_dir='C:/tmp/Football/models/test',summary_op = tf.summary.merge_all())]
                                          , eval_metric_ops=eval_metric_ops
                                          , export_outputs=export_outputs
                                          )

  return tf.estimator.Estimator(model_fn=model, model_dir=model_dir,
                                config = tf.estimator.RunConfig(save_checkpoints_steps=100,save_summary_steps=100))


