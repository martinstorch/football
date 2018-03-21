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
          W = tf.get_variable("W", shape=[X.shape[1],1], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.001))
          b = tf.get_variable("b", shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
          log_lambda = tf.matmul(X,W) + b
          y = tf.exp(log_lambda)
          #y = y[:,0]
        #y = tf.feature_column.linear_model(features, columns)
        y_round = tf.round(y)
        y_int = tf.cast(y_round, tf.int64)
        
        predictions = {
          "predictions":  y ,
          "estimations":  y_int 
        }
        export_outputs = {
            "predictions": tf.estimator.export.RegressionOutput(y)
            #, "estimations": tf.estimator.export.RegressionOutput(tf.cast(y_, tf.float32))
        }
          
        if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(
              mode=mode, predictions=predictions, export_outputs=export_outputs)

        labels_float = tf.cast(labels, tf.float32)
        #loss = tf.reduce_mean(tf.square(y[:,0] - tf.to_float(labels)),0)
        #loss = tf.losses.mean_squared_error(y[:,0] , labels)
        logfactorial = [0.000000000000000,
            0.000000000000000,
            0.693147180559945,
            1.791759469228055,
            3.178053830347946,
            4.787491742782046,
            6.579251212010101,
            8.525161361065415,
            10.604602902745251,
            12.801827480081469]

        tflf = tf.constant(logfactorial, dtype=tf.float32, shape=[10])
        labels_onehot = tf.one_hot(labels, 10)
#        print(y)
#        print(tflf)
#        
#        prob_num = tf.exp(-y)*tf.pow(y, labels_float)
#        prob_denom = tf.exp(tf.reduce_sum(tflf*labels_onehot,1))
#        print(prob_num)
#        print(prob_denom)
#        prob = prob_num / prob_denom
#        loss = tf.reduce_mean(-tf.log(prob))
#        
        loss = y[:,0] - labels_float*tf.log(y[:,0]) + tf.reduce_sum(tflf*labels_onehot,1)
        #loss = tf.nn.log_poisson_loss(labels_float, log_lambda[:,0])
        loss = tf.reduce_mean(loss)
        # Summaries to display for TRAINING and TESTING
        tf.summary.scalar("loss", loss)    
        accuracy1 = tf.cast(tf.equal(labels, y_int), tf.float32)
        accuracy = tf.metrics.accuracy(labels, y_int)  
        tf.summary.scalar("accuracy", accuracy[0]) 
        mae = tf.metrics.mean_absolute_error(labels=labels_float, predictions=y)
        tf.summary.scalar("mae", mae[0]) 
        mse = tf.metrics.mean_squared_error(labels=labels_float, predictions=y )
        tf.summary.scalar("mse", mse[0]) 
        print(accuracy)
        eval_metric_ops = {
            "accuracy": tf.metrics.mean(accuracy1)
            , "mae": mae
            , "mse": mse
            #,"confusion": (confmatrix, tf.no_op())
            , "accuracy2": accuracy
        }

        if mode == tf.estimator.ModeKeys.EVAL:
          return tf.estimator.EstimatorSpec(
              mode=mode, predictions=predictions, loss= loss, eval_metric_ops=eval_metric_ops)

        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
          
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions 
                                          , loss= loss, train_op=train
                                          , eval_metric_ops=eval_metric_ops)

  return tf.estimator.Estimator(model_fn=model, model_dir=model_dir,
                                config = tf.estimator.RunConfig(save_checkpoints_steps=100,save_summary_steps=100))


