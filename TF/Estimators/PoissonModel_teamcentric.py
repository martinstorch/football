# -*- coding: utf-8 -*-
import tensorflow as tf

#class LinearModel(object):
#  @property 
#  estimator_model 
#  
#  def __init__(self, model_dir, columns):

def create_estimator(model_dir, columns):    
  
  def buildGraph(features, columns, index): 
        with tf.variable_scope("Linear"):
          b = tf.get_variable("b", shape=[2], dtype=tf.float32, initializer=tf.zeros_initializer())
          log_lambda = b
          for c in columns:
            X = tf.feature_column.input_layer(features, [c])
            W = tf.get_variable("W"+c.name, shape=[X.shape[1],2], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.001))
            log_lambda = tf.matmul(X,W) + log_lambda 
            with tf.variable_scope(c.name):
              tf.summary.histogram("Weight", W)
              tf.summary.histogram("Inputs", X)
              tf.summary.histogram("Outputs", log_lambda)
        if False:
          X = tf.feature_column.input_layer(features, columns)
          with tf.variable_scope("Input_Layer"):
                tf.summary.histogram("Inputs", X)
  
  #        X = tf.layers.dense(inputs=X, units=60, activation=tf.nn.tanh, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
  #        with tf.variable_scope("Layer1"):
  #              tf.summary.histogram("Outputs", X)
  #        X = tf.layers.dense(inputs=X, units=60, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
  #        with tf.variable_scope("Layer2"):
  #              tf.summary.histogram("Outputs", X)
  #        X = tf.layers.dense(inputs=X, units=30, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.5), kernel_initializer=tf.random_normal_initializer(stddev=1))
  #        X = tf.layers.dense(inputs=X, units=30, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.5), kernel_initializer=tf.random_normal_initializer(stddev=1))
          log_lambda = tf.layers.dense(inputs=X, units=2, activation=None, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
          with tf.variable_scope("Output_Layer"):
                tf.summary.histogram("Inputs", X)
                tf.summary.histogram("Outputs", log_lambda)
              #tf.summary.histogram("Weights", log_lambda.kernel)
              #tf.summary.histogram("Bias", log_lambda.bias)
              
        #log_lambda = tf.exp(log_lambda)
#        with tf.variable_scope("Linear"):
#          W = tf.get_variable("W", shape=[X.shape[1],1], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.001))
#          b = tf.get_variable("b", shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
#          log_lambda = tf.matmul(X,W) + b
        #b = tf.get_variable("b"+index, shape=[2], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        y = tf.exp(log_lambda)
        #y = tf.maximum(log_lambda+b, 0.1)
          #y = y[:,0]
        #y = tf.feature_column.linear_model(features, columns)
        return log_lambda, y
        
  def model(features, labels, mode):
#        loss = None
#        train = None
#        eval_metric_ops = None
#        export_outputs = None
        accuracy1 = tf.ones(shape=[tf.shape(features["Team1"])[0]])
        # Build a linear model and predict values
#        print(features)
#        print(labels)
#        print(columns)
        log_lambda, y = buildGraph(features, columns, "1")
        #log_lambda2, y2 = buildGraph(features, columns, "2")
        
        y2 = tf.reshape(y[:,1], [-1,1])
        y = tf.reshape(y[:,0], [-1,1])
        log_lambda2 = tf.reshape(log_lambda[:,1], [-1,1])
        log_lambda = tf.reshape(log_lambda[:,0], [-1,1])
        
        y_round = tf.round(y)
        y_int = tf.cast(y_round, tf.int64)
        
        predictions = {
          "predictions":  y ,
          "estimations":  y_int ,
          "predictions2":  y2 
        }
        export_outputs = {
            "predictions": tf.estimator.export.RegressionOutput(y)
            #, "estimations": tf.estimator.export.RegressionOutput(tf.cast(y_, tf.float32))
        }
        with tf.variable_scope("Output"):
          tf.summary.histogram("y", y)
          tf.summary.histogram("y2",y2)
          tf.summary.histogram("log_lambda", log_lambda)
          tf.summary.histogram("log_lambda2", log_lambda2)
        
        for c in columns:
          #print(c)  
          X = tf.feature_column.input_layer(features, [c])
          predictions[c.name] = X
          
        if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(
              mode=mode, predictions=predictions, export_outputs=export_outputs)

        labels = features["OwnGoals"]  
        labels_float = tf.cast(labels, tf.float32)
        labels_float2 = tf.cast(features["OpponentGoals"], tf.float32)

        with tf.variable_scope("Labels"):
          tf.summary.histogram("labels", labels)
          tf.summary.histogram("labels_float2",labels_float2)
        
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
        labels_onehot2 = tf.one_hot(features["OpponentGoals"], 10)
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
        gs = features["OwnGoals"]
        gc = features["OpponentGoals"]
        pGS = tf.cast(tf.round(y), tf.int64)
        pGC = tf.cast(tf.round(y2), tf.int64)

        is_home = tf.cast(tf.equal(features["Where"] , "Home"), tf.float32)
        is_draw = tf.cast(tf.equal(gs,gc), tf.float32)
        is_win = tf.cast(tf.greater(gs,gc), tf.float32)
        is_loss = tf.cast(tf.less(gs,gc), tf.float32)
        is_full = tf.cast(tf.equal(gs,pGS) & tf.equal(gc,pGC), tf.float32)
        is_diff = tf.cast(tf.equal((gs-gc),(pGS-pGC)), tf.float32)

        is_tendency = tf.cast(tf.equal(tf.sign(gs-gc) , tf.sign(pGS-pGC)), tf.float32)
        points = is_draw * (4 * is_full + 2 * is_diff)
        points = points + (1-is_draw) * is_home * (2 * is_tendency + is_diff + is_full)
        points = points + (1-is_draw) * (1-is_home) * (4 * is_tendency + is_diff + 2*is_full)
        
        loss1 = y[:,0] - labels_float*tf.log(y[:,0]) + tf.reduce_sum(tflf*labels_onehot,1)
        loss2 = y2[:,0] - labels_float2*tf.log(y2[:,0]) + tf.reduce_sum(tflf*labels_onehot2,1)
        #loss1 = tf.nn.log_poisson_loss(labels_float, log_lambda[:,0])

        gsf = tf.cast(gs, tf.float32)
        gcf = tf.cast(gc, tf.float32)
        fGS = y
        fGC = y2
        print(features["Team1"].shape[0])
        print(tf.shape(features["Team1"]))
        offset = fGS*0+0.2 #tf.constant(0.5, dtype=tf.float32, shape=tf.stack(features["Team1"].shape[0]))
        def soft_equals(x,y):
          return 3 * tf.sigmoid(2*x-2*y+offset) * tf.sigmoid(2*y-2*x+offset)
        def soft_greater(x,y):
          return 2*tf.sigmoid(tf.nn.relu(2*x-2*y-1*offset))-1
        
        soft_is_full = soft_equals(fGS,gsf) * soft_equals(fGC,gcf)
        soft_is_diff = soft_equals((gsf-gcf),(fGS-fGC))
        soft_win = is_win * soft_greater(fGS,fGC)
        soft_loss = is_loss * soft_greater(fGC,fGS)
        soft_draw = is_draw * soft_equals(fGS,fGC) 
        soft_is_tendency = soft_win + soft_loss + soft_draw
        soft_points = is_draw * (4 * is_full + 2 * is_diff)
        soft_points = soft_points + (1-is_draw) * is_home * (2 * soft_is_tendency + soft_is_diff + soft_is_full)
        soft_points = soft_points + (1-is_draw) * (1-is_home) * (4 * soft_is_tendency + soft_is_diff + 2*soft_is_full)
        
#        with tf.Session():
#          offset=0.2
#          print(soft_equals(1,1).eval())
#          print(soft_equals(1,1.2).eval())
#          print(soft_equals(1.2,1).eval())
#          print(soft_equals(1,1.4).eval())
#          print(soft_equals(1,1.6).eval())
#          print(soft_equals(1,1.8).eval())
#          print(soft_equals(0.5,1.8).eval())
#          print()
#          print(soft_greater(0.5,1.8).eval())
#          print(soft_greater(0.5,0.1).eval())
#          print(soft_greater(2,1).eval())
#          print(soft_greater(2.5,2).eval())
#          print(soft_greater(2.7,2).eval())
#          print(soft_greater(5,0).eval())
        
        # Summaries to display for TRAINING and TESTING
#        tf.summary.scalar("loss1", loss1)    
#        tf.summary.scalar("loss2", loss2)    
        accuracy1 = tf.cast(tf.equal(labels, y_int), tf.float32)
        accuracy = tf.metrics.accuracy(labels, y_int)  
        tf.summary.scalar("accuracy", accuracy[0]) 
        mae = tf.metrics.mean_absolute_error(labels=labels_float, predictions=y)
        mae2 = tf.metrics.mean_absolute_error(labels=labels_float2, predictions=y2)
        tf.summary.scalar("mae", mae[0]) 
        tf.summary.scalar("mae2", mae2[0]) 
        mse = tf.metrics.mean_squared_error(labels=labels_float, predictions=y )
        mse2 = tf.metrics.mean_squared_error(labels=labels_float2, predictions=y2 )
        tf.summary.scalar("mse", mse[0]) 
        tf.summary.scalar("mse2", mse2[0]) 

        loss = tf.reduce_mean(0*loss1+0*loss2-soft_points)#-soft_win*0.5-5*soft_loss) # loss1+loss2
        tf.summary.scalar("loss", loss)    

        #print(accuracy)
        eval_metric_ops = {
            "accuracy": tf.metrics.mean(accuracy1)
            , "loss1": tf.metrics.mean(loss1)
            , "loss2": tf.metrics.mean(loss2)
            , "mae": mae
            , "mae2": mae2
            , "mse": mse
            , "mse2": mse2
            , "points": tf.metrics.mean(points)
            , "is_tendency": tf.metrics.mean(is_tendency)
            , "is_diff": tf.metrics.mean(is_diff)
            , "is_full": tf.metrics.mean(is_full)
            , "soft_points": tf.metrics.mean(soft_points)
            , "soft_is_tendency": tf.metrics.mean(soft_is_tendency)
            , "soft_is_diff": tf.metrics.mean(soft_is_diff)
            , "soft_is_full": tf.metrics.mean(soft_is_full)
            , "soft_win": tf.metrics.mean(soft_win)
            , "soft_loss": tf.metrics.mean(soft_loss)
            , "soft_draw": tf.metrics.mean(soft_draw)
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


