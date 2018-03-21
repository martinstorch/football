setwd("~/LearningR/Bundesliga")

#install.packages("dplyr")
#install.packages("party")
##install.packages("rattle",  dependencies=c("Depends", "Suggests"))

#require(devtools)
#install.packages('xgboost')

library(xgboost)

library(dplyr)
#require(expm)
#library(car)
library(party)
#library(rattle)
# rattle()
require(Matrix)
require(data.table)

#######################################################################################################################
# predict goals separately


buildMask <- function() {
  cn<-expand.grid(0:4, 0:4)
  mask<-list()
  for (key in paste(cn$Var1, cn$Var2))
    mask[[key]]<-matrix(5,5,data=c(0), dimnames = list(0:4, 0:4))
  for (i in 0:4)
    for (j in 0:4) {  # draw
      mask[[paste(i,i)]][j+1,j+1]<-2
      mask[[paste(i,i)]][i+1,i+1]<-6
    }
  for (i in 1:4)
    for (j in 0:(i-1))  { 
      for (k in 1:4)
        for (l in 0:(k-1)) { # home 
          mask[[paste(i,j)]][k+1,l+1]<-2
          if (i-j==k-l) mask[[paste(i,j)]][k+1,l+1]<-3
        }
      mask[[paste(i,j)]][i+1,j+1]<-4
    }
  for (i in 0:3)
    for (j in (i+1):4)  { 
      for (k in 0:3)
        for (l in (k+1):4) { # home 
          mask[[paste(i,j)]][k+1,l+1]<-4
          if (i-j==k-l) mask[[paste(i,j)]][k+1,l+1]<-5
        }
      mask[[paste(i,j)]][i+1,j+1]<-7
    }
  return(mask)
}

summarizePrediction<-function(data, gspred, gcpred, label="") {
  
  resultKPIs<-data.frame(pred_tend=sign(gspred-gcpred), act_tend=sign(data$GS-data$GC), 
                         pred_diff=gspred-gcpred, act_diff=data$GS-data$GC, 
                         pred_GS=gspred, pred_GC=gcpred, act_GS=data$GS, act_GC=data$GC,
                         where=data$where, row.names = NULL) %>% 
    mutate(
      pred_score=factor(ifelse(where=="Home", paste(pred_GS, pred_GC, sep = ":"), paste(pred_GC, pred_GS, sep = ":")), 
                        levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                                  "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                                  "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE)
      
    ) %>%
    mutate( fullhit=pred_GS==act_GS & pred_GC==act_GC, 
            gdhit=pred_diff == act_diff, 
            tendhit=pred_tend == act_tend,
            bidres=factor("None", levels=c("None", "Tendency", "GoalDiff", "Full"), ordered=TRUE), bidpoints=0
  )
  resultKPIs<-within(resultKPIs,{
    bidres[tendhit]="Tendency"
    bidres[gdhit]="GoalDiff"
    bidres[fullhit]="Full"
  })
  resultKPIs<-within(resultKPIs,{
    bidpoints[tendhit & where=='Home' & act_tend ==  1]=2
    bidpoints[tendhit & where=='Home' & act_tend == -1]=4
    bidpoints[gdhit   & where=='Home' & act_tend ==  1]=3
    bidpoints[gdhit   & where=='Home' & act_tend ==  0]=2
    bidpoints[gdhit   & where=='Home' & act_tend == -1]=5
    bidpoints[fullhit & where=='Home' & act_tend ==  1]=4
    bidpoints[fullhit & where=='Home' & act_tend ==  0]=6
    bidpoints[fullhit & where=='Home' & act_tend == -1]=7
    bidpoints[tendhit & where=='Away' & act_tend ==  1]=4
    bidpoints[tendhit & where=='Away' & act_tend == -1]=2
    bidpoints[gdhit   & where=='Away' & act_tend ==  1]=5
    bidpoints[gdhit   & where=='Away' & act_tend ==  0]=2
    bidpoints[gdhit   & where=='Away' & act_tend == -1]=3
    bidpoints[fullhit & where=='Away' & act_tend ==  1]=7
    bidpoints[fullhit & where=='Away' & act_tend ==  0]=6
    bidpoints[fullhit & where=='Away' & act_tend == -1]=4
  })
  
  print(resultKPIs %>% select(pred_GS, pred_GC) %>% table)
  print(resultKPIs %>% select(pred_tend, act_tend) %>% table)
  print(resultKPIs %>% select(pred_diff) %>% table)
  print(resultKPIs %>% select(pred_diff, act_diff) %>% table)

  scoreTable<-resultKPIs %>% select(pred_score, bidres) %>% table

  print(scoreTable)
  print(colSums(scoreTable)/sum(scoreTable)*100)
  print(scoreTable/rowSums(scoreTable)*100)

  summaryTable<-resultKPIs %>% 
    group_by(where) %>%
    summarise(tend_pct=mean(tendhit), diff_pct=mean(gdhit), exact_pct=mean(fullhit), avg_points=mean(bidpoints), total_points=sum(bidpoints)) 
    
  summaryTableAll<-resultKPIs %>% 
    summarise(tend_pct=mean(tendhit), diff_pct=mean(gdhit), exact_pct=mean(fullhit), avg_points=mean(bidpoints), total_points=sum(bidpoints)) 

  resultKPIs %>% filter(where=="Home") %>% select (x=pred_score, y=bidres) %>% 
    plot(xlab=label, main=
           summaryTable %>% filter(where=="Home") %>% transmute(where=as.character(where), round(tend_pct, 3)*100, round(diff_pct, 3)*100, round(exact_pct, 3)*100, round(avg_points, 2), total_points) %>% paste(collapse=" / ")
    )
         
  resultKPIs %>% filter(where=="Away") %>% select (x=pred_score, y=bidres) %>% 
    plot(xlab=label, main=
           summaryTable %>% filter(where=="Away") %>% transmute(where=as.character(where), round(tend_pct, 3)*100, round(diff_pct, 3)*100, round(exact_pct, 3)*100, round(avg_points, 2), total_points) %>% paste(collapse=" / ")
    )
  
  resultKPIs %>% select (x=pred_score, y=bidres) %>% 
    plot(xlab=label, main=
           summaryTableAll %>% transmute(where="All", round(tend_pct, 3)*100, round(diff_pct, 3)*100, round(exact_pct, 3)*100, round(avg_points, 2), total_points/2) %>% paste(collapse=" / ")
    )
  print(label)
  print(summaryTable %>% data.frame)
  return(summaryTable)
}  

evaluateNumeric<-function(model, newdata, label="") {
  pred<-predict(model, newdata=newdata)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  summarizePrediction(data = newdata, round(gspred), round(gcpred), label)
  return(data.frame(gspred, gcpred))
}

evaluateFactor<-function(model, newdata, label="") {
  pred<-predict(model, newdata=newdata)
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  summarizePrediction(data = newdata, gspred, gcpred, label)
  return(data.frame(gspred, gcpred))
}

evaluateMaxExpectation <- function(model, newdata, label) {
  predProbsList<-predict(model, newdata = newdata , type = "prob")
  predoutcomes<-t(sapply(predProbsList, cbind))
  colnames(predoutcomes)<-colnames(predProbsList[[1]])
  predoutcomes<-predoutcomes[,order(colnames(predoutcomes))]  
  colnames(predoutcomes)<-substr(colnames(predoutcomes), 7,9)
  substr(colnames(predoutcomes), 2,2)<-":"
  
  # matchidx<-match(paste(newdata$GS, newdata$GC, sep = ":"), colnames(predoutcomes))
  # matchedprobs<-predoutcomes[cbind(1:length(matchidx), matchidx)]
  # summary(matchedprobs)
  # 
  # matchidx4<-match(paste(gspred4, gcpred4, sep = ":"), colnames(predoutcomes))
  # matchedprobs4<-predoutcomes[cbind(1:length(matchidx4), matchidx4)]
  # summary(matchedprobs4)
  # 
  mask<-mask[order(names(mask))]
  expectedValues<-sapply(1:25, function(i) predoutcomes %*% as.vector(t(mask[i][[1]])))
  colnames(expectedValues)<-names(mask)
  
  maxEV<-apply(expectedValues, 1, max)
  whichMaxEV<-apply(expectedValues, 1, which.max)
  
  pred<-colnames(expectedValues)[whichMaxEV]
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  result<-summarizePrediction(data = test, gspred, gcpred, label)
}

limitMaxScore<-function(dataframe, limit) {
  GS<-dataframe$GS
  GC<-dataframe$GC
  eg1<-ifelse(GS>limit,GS-limit,0)
  eg2<-ifelse(GC>limit,GC-limit,0)
  eg3<-ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0)
  eg4<-ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)
  dataframe$GS<-eg3
  dataframe$GC<-eg4
  return(dataframe)
  # data.frame(GS=GS, GC=GC) %>% 
  #   mutate(eg1=ifelse(GS>limit,GS-limit,0), eg2=ifelse(GC>limit,GC-limit,0)) %>% 
  #   mutate(eg3=ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0), eg4=ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)) %>%
  #   transmute(GS=eg3, GC=eg4)
}

buildFactorScore<-function(dataframe) {
  
  dataframe %>% 
  mutate(score=factor(paste(GS, GC, sep = ":"), 
                      levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                               "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                               "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE)) %>%
  select(-GS, -GC)
}

################################################################################
# data preparation for learning

xy<-read.csv(file="BLfeatures.csv", stringsAsFactors = TRUE, as.is = FALSE)

train<-xy %>% filter(season %in% c("2016_17","2014_15") ) 
test<-xy%>% filter(season == "2015_16")

trainY<-train %>% select(GS, GC)
testY<-test %>% select(GS, GC)

xyMatrix <- sparse.model.matrix(~.-1, data = xy)
oobidx<-xyMatrix[,"season2016_17"]==1
trainXM <- xyMatrix[!oobidx,-2:-1]
testXM <- xyMatrix[oobidx, -2:-1]

trainYM<-xyMatrix[!oobidx,1:2]
testYM<-xyMatrix[oobidx,1:2]

mask<-buildMask()
gLev<-sort(buildFactorScore(data.frame(expand.grid(GS=0:4, GC=0:4)))$score)









################################################################################
# build models and evaluate training data scores

model_cv_goals <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree=1000, mtry=20))
result_cv<-evaluateNumeric(model_cv_goals, newdata = train, label = "Model 1 Train")

data_cv<-data.frame(gspred=result_cv$gspred, gcpred=result_cv$gcpred, train)
model2<-cforest(GS*GC ~ ., data=data_cv %>% select(GS, GC, gspred, gcpred), controls = cforest_unbiased(ntree=50, mtry=2))
result_cv2<-evaluateNumeric(model2, newdata = data_cv, label = "Model 2 Train")

train_limit3<-limitMaxScore(train, 3)
model3 <- cforest(GS*GC ~ ., data=train_limit3, controls = cforest_unbiased(ntree=1000, mtry=20))
result_cv3<-evaluateNumeric(model3, newdata = train, label = "Model 3 Train")

data_cv4<-data.frame(gspred=result_cv$gspred, gcpred=result_cv$gcpred, 
                     gspred3=result_cv3$gspred, gcpred3=result_cv3$gcpred, train_limit3) %>% 
    buildFactorScore() %>%
    select(score, gspred, gcpred, gspred3, gcpred3, team, oppTeam, where)

model4<-cforest(score ~ ., data=data_cv4, controls = cforest_unbiased(ntree=1000, mtry=5))
#sort(varimp(model4), decreasing = TRUE)
result_cv4<-evaluateFactor(model4, newdata = data.frame(train, gspred=result_cv$gspred, gcpred=result_cv$gcpred, gspred3=result_cv3$gspred, gcpred3=result_cv3$gcpred), label = "Model 4 Train")

train_factor<-limitMaxScore(train, 3) %>% buildFactorScore()

model5<-cforest(score ~ ., data=train_factor, controls = cforest_unbiased(ntree=1000, mtry=5))
#sort(varimp(model4), decreasing = TRUE)
result_cv5<-evaluateFactor(model5, newdata = train, label = "Model 5 Train")


# train_factor %>% select (GS, GC) %>% table
# train %>% select (GS, GC) %>% table
# train_factor %>% select (score) %>% table

evaluateXgbNumeric<-function(modelGS, modelGC, newdata, newlabels, label="") {
  gspred<-predict(modelGS, newdata=newdata)
  gcpred<-predict(modelGC, newdata=newdata)
  comparedata<-as.data.frame(as.matrix(newlabels))
  comparedata$where=as.factor(ifelse(newdata[,"whereHome"]==1, "Home", "Away"))
  summarizePrediction(data = comparedata, round(gspred), round(gcpred), label)
  return(data.frame(gspred, gcpred))
}

xgbparams<-list(max_depth = 3, eta = 0.05, subsample=0.5, colsample_bytree=0.8, nthread = 3, objective = "count:poisson", lambda=0.01, alpha=0.01)
model_cv_xg_GS <- xgboost(data = trainXM, label=trainYM[,1], params=xgbparams, print_every_n=10, nrounds=500)
model_cv_xg_GC <- xgb.train(data = xgb.DMatrix(trainXM, label=trainYM[,2]), params=xgbparams, print_every_n=10, nrounds=500)

dtrainGS<-xgb.DMatrix(trainXM, label=trainYM[,1])
dtestGS<-xgb.DMatrix(testXM, label=testYM[,1])
watchlistGS <- list(train=dtrainGS, test=dtestGS)
xgbparams<-list(gamma=15, max_depth = 3, eta = 0.005, subsample=0.7, colsample_bytree=0.5, nthread = 3, objective = "reg:linear", lambda=0.01, alpha=0.01)
model_cv_xg_GS <- xgb.train(data = dtrainGS, params=xgbparams, print_every_n=10, nrounds=1000, watchlist=watchlistGS)
plot(model_cv_xg_GS$evaluation_log[,2][[1]], lwd=1, type="l")
points(model_cv_xg_GS$evaluation_log[,3][[1]], lwd=1, type="l")
min(model_cv_xg_GS$evaluation_log[,3][[1]])
abline(v=which.min(model_cv_xg_GS$evaluation_log[,3][[1]]), col="red")

dtrainGC<-xgb.DMatrix(trainXM, label=trainYM[,2])
dtestGC<-xgb.DMatrix(testXM, label=testYM[,2])
watchlistGC <- list(train=dtrainGC, test=dtestGC)
model_cv_xg_GC <- xgb.train(data = dtrainGC, params=xgbparams, print_every_n=10, nrounds=1000, watchlist=watchlistGC)
plot(model_cv_xg_GC$evaluation_log[,2][[1]], lwd=1, type="l")
points(model_cv_xg_GC$evaluation_log[,3][[1]], lwd=1, type="l")
min(model_cv_xg_GC$evaluation_log[,3][[1]])
abline(v=which.min(model_cv_xg_GC$evaluation_log[,3][[1]]), col="red")


mat <- xgb.importance (feature_names = colnames(trainXM),model = model_cv_xg_GS)
xgb.plot.importance (importance_matrix = mat[1:40]) 
mat <- xgb.importance (feature_names = colnames(trainXM),model = model_cv_xg_GC)
xgb.plot.importance (importance_matrix = mat[1:40]) 
model_xgcv_GS <- xgb.cv(nfold = 4, data = trainXM, label=trainYM[,1], params=xgbparams, print_every_n=10, nrounds=2500)
str(model_xgcv_GS)

smoothScatter(evaluateXgbNumeric(model_cv_xg_GS, model_cv_xg_GC, trainXM, trainYM, "xbg reg:linear train"))
smoothScatter(evaluateXgbNumeric(model_cv_xg_GS, model_cv_xg_GC, testXM, testYM, "xbg reg:linear test"))

train_pred_poiss<-evaluateXgbNumeric(model_cv_xg_GS, model_cv_xg_GC, trainXM, trainYM, "xbg reg:linear train")
test_pred_poiss<-evaluateXgbNumeric(model_cv_xg_GS, model_cv_xg_GC, testXM, testYM, "xbg reg:linear test")

watchlistGS2 <- list(train=xgb.DMatrix(cbind(trainXM[,1:50], as.matrix(train_pred_poiss)), label=trainYM[,1]),
                     test=xgb.DMatrix(cbind(testXM[,1:50], as.matrix(test_pred_poiss)), label=testYM[,1]))
watchlistGC2 <- list(train=xgb.DMatrix(cbind(trainXM[,1:50], as.matrix(train_pred_poiss)), label=trainYM[,2]),
                     test=xgb.DMatrix(cbind(testXM[,1:50], as.matrix(test_pred_poiss)), label=testYM[,2]))

xgbparams<-list(gamma=8, max_depth = 3, eta = 0.005, subsample=0.7, colsample_bytree=0.3, nthread = 3, objective = "count:poisson", lambda=0.01, alpha=0.01)
model_cv_xg_GS2 <- xgb.train(nrounds=2000, watchlist=watchlistGS2, params=xgbparams, data = watchlistGS2[["train"]], print_every_n=10)
model_cv_xg_GC2 <- xgb.train(nrounds=2000, watchlist=watchlistGC2, params=xgbparams, data = watchlistGC2[["train"]], print_every_n=10)

smoothScatter(evaluateXgbNumeric(model_cv_xg_GS2, model_cv_xg_GC2, cbind(trainXM[,1:50], as.matrix(train_pred_poiss)), trainYM, "xbg reg:linear train"))
smoothScatter(evaluateXgbNumeric(model_cv_xg_GS2, model_cv_xg_GC2, cbind(testXM[,1:50], as.matrix(test_pred_poiss)), testYM, "xbg reg:linear test"))


trainLabels<-(as.data.frame(as.matrix(trainYM)) %>% limitMaxScore(4) %>% buildFactorScore())$score
testLabels<-(as.data.frame(as.matrix(testYM))  %>% buildFactorScore())$score

xgbparamsFactor<-list(gamma=10, max_depth = 3, eta = 0.01, subsample=0.3, colsample_bytree=0.05, nthread = 3, objective = "multi:softmax", num_class=26)
model_cv_xg_Score <- xgboost(obj=objective, params=xgbparamsFactor, data = cbind(trainXM, as.matrix(train_pred_poiss)), label=trainLabels, print_every_n=1, nrounds=2)

model_cv_xg_Score <- xgboost(params=xgbparamsFactor, data = cbind(trainXM, as.matrix(train_pred_poiss)), label=trainLabels, print_every_n=1, nrounds=200)

##########################################################################################################################

trainLabels<-(as.data.frame(as.matrix(trainYM)) %>% limitMaxScore(4) %>% buildFactorScore())$score
testLabels<-(as.data.frame(as.matrix(testYM)) %>% limitMaxScore(4) %>% buildFactorScore())$score

trainMatrix<-xgb.DMatrix(trainXM, label=as.integer(trainLabels)-1)
attr(trainMatrix, "where")<-(as.data.frame(as.matrix(trainXM)) %>% select(whereHome))$where

testMatrix<-xgb.DMatrix(testXM, label=as.integer(testLabels)-1)
attr(testMatrix, "where")<-(as.data.frame(as.matrix(testXM)) %>% select(whereHome))$where

watchlist<-list(train=trainMatrix, test=testMatrix)

evaltarget <- function(preds, dtrain) {
  labels<-levels(trainLabels)[getinfo(dtrain, "label")+1]
  predlabels<-levels(trainLabels)[preds+1]
  where<-attr(dtrain, "where")
  GS<-as.integer(substr(as.character(labels), start = 1, stop = 1))
  GC<-as.integer(substr(as.character(labels), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))

 # summarizePrediction(data = data.frame(GS, GC, where=ifelse(where==1, "Home", "Away")), pGS, pGC, "label")
  
  HG<-ifelse(where==1, GS, GC)
  AG<-ifelse(where==1, GC, GS)
  pHG<-ifelse(where==1, pGS, pGC)
  pAG<-ifelse(where==1, pGC, pGS)
  # print(table(pHG, pAG))
  # print(table(pHG-pAG, HG-AG))
  x1<-sum(sign(HG-AG)==sign(pHG-pAG))
  x2<-sum(sign(HG-AG)==-1 & sign(pHG-pAG)==-1)
  x3<-sum(HG-AG==pHG-pAG & HG!=AG)
  x4<-sum(HG==pHG & AG==pAG & HG>AG)
  x5<-sum(HG==pHG & AG==pAG & HG<AG)
  x6<-sum(HG==pHG & AG==pAG & HG==AG)
  totalpoints<- x1*2 + x2*2  + x3 + x4+ x5*2+x6*4
#  print(data.frame(x1, x2, x3, x4, x5, x6, totalpoints, l=length(predlabels), home=sum(where==1)))
  avgpoints <- totalpoints / length(predlabels)
  return(list(metric = "avgpoints", value = avgpoints))
}
xgbparamsFactor<-list(gamma=10, max_depth = 3, eta = 0.01, subsample=0.3, colsample_bytree=0.05, nthread = 1, num_class=nlevels(trainLabels))
modeleval<-xgboost(verbose=1, params = xgbparamsFactor, data = trainMatrix , nrounds =5, feval=evaltarget, maximize=TRUE, print_every_n=1)

prediction_estimate<-function(){
  lGS<-as.integer(substr(as.character(levels(trainLabels)), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(levels(trainLabels)), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(levels(trainLabels)), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(levels(trainLabels)), start = 3, stop = 3))
  p1<-expand.grid(lGS, pGS)
  p2<-expand.grid(lGC, pGC)
  p3<-ifelse(p1$Var1==p1$Var2 & p2$Var1==p2$Var2, 1, 0) # full hit
  p4<-abs((p1$Var1-p2$Var1) - (p1$Var2-p2$Var2))*-0.4 + 2 # goal diff score 
  p5<-ifelse(sign(p1$Var1-p2$Var1) == sign(p1$Var2-p2$Var2), 0.5, 0) # tendendy score 
  p6<-abs((p1$Var1+p2$Var1) - (p1$Var2+p2$Var2))*-0.05 + 0.7 # total goals  score 
  p7<-(2*p3+3*p4+p5+p6)
  predest<-matrix(p7, ncol=25, byrow = TRUE)
  colnames(predest)<-levels(trainLabels)
  rownames(predest)<-levels(trainLabels)
  predest<-predest/rowSums(predest)
}
predest<-prediction_estimate()

calcPoints<-function(GS1, GC1, GS2, GC2, home){
  x1<-sign(GS1-GC1)==sign(GS2-GC2) # tendency
  x2<-(GS1-GC1)==(GS2-GC2) # goaldiff
  x3<-(GS1==GS2) & (GC1==GC2) # full
  
  c4<-ifelse(GS1==GC1, ifelse(x1, 2, 0)+ifelse(x3, 4, 0),
             ifelse((home & GS1>GC1)|(!home & GS1<GC1),
                    ifelse(x1, 2, 0)+ifelse(x2, 1, 0)+ifelse(x3, 1, 0),
                    ifelse(x1, 4, 0)+ifelse(x2, 1, 0)+ifelse(x3, 2, 0)
             )
  )
  return(c4)
}


objective<-function(preds, dtrain)
{
  labels<-levels(trainLabels)[getinfo(dtrain, "label")+1]
  predlabels<-levels(trainLabels)[preds+1]
  where<-attr(dtrain, "where")
  GS<-as.integer(substr(as.character(labels), start = 1, stop = 1))
  GC<-as.integer(substr(as.character(labels), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))
  HG<-ifelse(where==1, GS, GC)
  AG<-ifelse(where==1, GC, GS)
  pHG<-ifelse(where==1, pGS, pGC)
  pAG<-ifelse(where==1, pGC, pGS)
  lGS<-as.integer(substr(as.character(levels(trainLabels)), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(levels(trainLabels)), start = 3, stop = 3))
#  print(table(sign(pGS-pGC), sign(GS-GC)))
  g1<-lGS-lGC
  g2<-expand.grid(sign(lGS-lGC), sign(GS-GC)-sign(pGS-pGC))
  g4<-expand.grid(sign(lGS-lGC), sign(GS-GC))
  g3<-ifelse(g4$Var1==g4$Var2, -1, 1)*(1+g2$Var2^2) # labels - prediction

  gdiff2<-expand.grid((lGS-lGC), (GS-GC)-(pGS-pGC))
  gdiff4<-expand.grid((lGS-lGC), (GS-GC))
  gdiff3<-ifelse(gdiff4$Var1==gdiff4$Var2, -1, 1)*(1+abs(gdiff2$Var2)) # labels - prediction
  
  g5<-expand.grid(lGS, GS==pGS & GC==pGC)
  g6<-expand.grid(lGS, GS)
  g7<-expand.grid(lGC, GC)
  g8<-ifelse(g5$Var2, 1, 2) * ifelse(g6$Var1==g6$Var2 & g7$Var1==g7$Var2, -1, 1)

  grad<-g3*0.2+gdiff3*0.1+g8*0.05
  hess<-runif(n=length(preds)*nlevels(trainLabels), min=-0.15, max=0.15)*g3*0.7
  # hess<-rep(-0.1, length(preds)*nlevels(trainLabels))
  hess<- g3 * 1.9 * runif(n=length(preds)*nlevels(trainLabels), min=-0.1, max=0.1)
  hess<- (5-abs(gdiff4$Var2-gdiff4$Var1))*(5-gdiff2$Var2)*0.03
            #ifelse(g6$Var1==g6$Var2 & g7$Var1==g7$Var2, 0.35, 0.05)

  # print(table(sign(pGS-pGC), sign(GS-GC)))
  # print(head(data.frame(GS, GC, pGS, pGC), 10))
  # m<-matrix(data = grad, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])
  # m<-matrix(data = hess, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])

  return(list(grad=grad, hess=hess))
}
xgbparamsFactor<-list(gamma=10, max_depth = 2, eta = 0.001, subsample=0.5, colsample_bytree=0.8, nthread = 1, num_class=nlevels(trainLabels))
#modeleval<-xgboost(verbose=1, obj=objective, params = xgbparamsFactor, data = trainMatrix , nrounds =35, feval=evaltarget, maximize=TRUE, print_every_n=1)



xgbparamsFactor<-list(gamma=10, max_depth = 10, eta = 1, subsample=0.5, colsample_bytree=0.8, nthread = 10, num_class=nlevels(trainLabels))
modelsoftprob<-xgb.train(nrounds =100, watchlist=watchlist, verbose=1, objective="multi:softprob", eval_metric=c("mlogloss", "merror"), params = xgbparamsFactor, data = trainMatrix , print_every_n=1)
printLearningCurve(modelsoftprob, maximise = FALSE)
result<-evaluateXgbFactor(modelsoftprob, trainXM, trainYM, "train XGB multisoft function")
result<-evaluateXgbFactor(modelsoftprob, testXM, testYM, "test XGB multisoft function")

predprob<-predict(modelsoftprob, newdata=trainXM, type="prob")
dim(predprob)<-c(nrow(trainXM), nlevels(trainLabels))
colnames(predprob)<-levels(trainLabels)

summary(rowSums(predprob))
summary(predprob[, "3:1"])
predprob[1, ]
max(predprob[1, ])
which.max(predprob[1, ])

buildMaskMatrix<-function(){
  e<-expand.grid(gLev, gLev)
  pGS<-as.integer(substr(as.character(e$Var1), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(e$Var1), start = 3, stop = 3))
  aGS<-as.integer(substr(as.character(e$Var2), start = 1, stop = 1))
  aGC<-as.integer(substr(as.character(e$Var2), start = 3, stop = 3))

  hp<-calcPoints(GS1 = pGS, GC1 = pGC, GS2 = aGS, GC2 = aGC, home = TRUE)
  hm<-matrix(hp, byrow = TRUE, dimnames = list(levels(gLev), levels(gLev)), ncol=nlevels(gLev), nrow=nlevels(gLev))
  ap<-calcPoints(GS1 = pGS, GC1 = pGC, GS2 = aGS, GC2 = aGC, home = FALSE)
  am<-matrix(ap, byrow = TRUE, dimnames = list(levels(gLev), levels(gLev)), ncol=nlevels(gLev), nrow=nlevels(gLev))
  return(list(home=hm, away=am))
}

maskm<-buildMaskMatrix()

plotImportance<-function(model, n=ncol(trainXM), cols=colnames(trainXM)){
  mat <- xgb.importance (feature_names = cols, model = model)
  xgb.plot.importance (importance_matrix = mat[1:n]) 
}

printImportance<-function(model, n=ncol(trainXM), cols=colnames(trainXM)){
  mat <- xgb.importance (feature_names = cols, model = model)
  if (n==ncol(trainXM))
    print.data.frame (mat) 
  else
    print.data.frame (mat[1:n]) 
}

objective<-function(preds, dtrain)
{
  ylabels<-getinfo(dtrain, "label")+1
  labels<-levels(gLev)[ylabels]
  predlabels<-levels(gLev)[preds+1]
  where<-attr(dtrain, "where")
  GS<-as.integer(substr(as.character(labels), start = 1, stop = 1))
  GC<-as.integer(substr(as.character(labels), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))
  pL<-calcPoints(GS1 = GS, GC1 = GC, GS2 = pGS, GC2 = pGC, home = where==1)
  
  lGS<-as.integer(substr(as.character(levels(gLev)), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(levels(gLev)), start = 3, stop = 3))
  #  print(table(sign(pGS-pGC), sign(GS-GC)))
  p1<-expand.grid(1:nlevels(gLev), preds+1)
  prob<-predest[cbind(p1$Var2,p1$Var1)]
  
  probmatrix<-predest[preds+1,]
  homepoints<-rowSums(probmatrix * maskm$home[ylabels,])
  awaypoints<-rowSums(probmatrix * maskm$away[ylabels,])
  points<-data.frame(homepoints, awaypoints, where) %>% mutate(points=ifelse(where==1, homepoints, awaypoints))
  points<-points$points
  # print(points)
  
  c1<-expand.grid(lGS,where)
  c2<-expand.grid(lGS,GS)
  c3<-expand.grid(lGC,GC)
  
  c4<-calcPoints(GS1 = c2$Var1, GC1 = c3$Var1, GS2 = c2$Var2, GC2 = c3$Var2, home = c1$Var2==1)
  
  L <- expand.grid(lGS, pL)$Var2
  L <- expand.grid(lGS, points)$Var2
  #pc2<-expand.grid(lGS,pGS)
  #pc3<-expand.grid(lGC,pGC)
  
  #L<-calcPoints(GS1 = pc2$Var1, GC1 = pc3$Var1, GS2 = pc2$Var2, GC2 = pc3$Var2, home = c1$Var2==1)
  
  grad<-prob*(c4-L)*25 #+runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
  #grad<-runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
  hess<-abs(grad*(1-2*prob))/25
  #hess<-rnorm(n=length(preds)*nlevels(gLev), mean = -0.5, sd=1)*grad*0.3
  #hess<-grad
  hess<-runif(n=length(preds)*nlevels(gLev), min=-0.05, max=0.05)*grad*0.2
  #hess<-rep(5.0, length(preds)*nlevels(gLev))
  # print(table(sign(pGS-pGC), sign(GS-GC)))
  # print(head(data.frame(GS, GC, pGS, pGC), 10))
  # m<-matrix(data = c4, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])
  # m<-matrix(data = L, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])
  # m<-matrix(data = prob, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])
  # m<-matrix(data = grad, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])
  # m<-matrix(data = hess, byrow = TRUE, ncol=nlevels(trainLabels), dimnames = list(rownames=NULL, colnames=levels(trainLabels)))
  # print(m[1:10,])
  
  return(list(grad= -grad, hess=hess))
}
# 508 xgbparamsFactor<-list(gamma=0, max_depth = 2, eta = 0.001, subsample=0.3, colsample_bytree=0.1, nthread = 6, num_class=nlevels(trainLabels))
#grad<-prob*(c4-L)*10 #+runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
#hess<-runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)*grad*0.3

xgbparamsFactor<-list(gamma=0, max_depth = 4, eta = 1, subsample=0.5, colsample_bytree=0.2, nthread = 6, num_class=nlevels(trainLabels))
modeleval<-xgb.train(nrounds =100, watchlist=watchlist, verbose=1, obj=objective, params = xgbparamsFactor, data = trainMatrix , feval=evaltarget, maximize=TRUE, print_every_n=1)
printLearningCurve(modeleval, maximise = TRUE)
result<-evaluateXgbFactor(modeleval, testXM, testYM, "test XGB objective function")
result<-evaluateXgbFactor(modeleval, trainXM, trainYM, "train XGB objective function")
plotImportance(modeleval, 40)
printImportance(modeleval)


evaluateXgbFactor<-function(model, newdata, newlabels, label="", levels1=levels(buildFactorScore(train)$score)) {
  comparedata<-as.data.frame(as.matrix(newlabels))
  comparedata$where=as.factor(ifelse(newdata[,"whereHome"]==1, "Home", "Away"))
  pred<-predict(model, newdata=newdata)
  pred<-levels1[pred+1]
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  summarizePrediction(data = comparedata, gspred, gcpred, label)
  return(data.frame(gspred, gcpred))
}
result<-evaluateXgbFactor(modeleval, trainXM, trainYM, "train XGB objective function")
result<-evaluateXgbFactor(modeleval, testXM, testYM, "test XGB objective function")

printLearningCurve<-function(model, maximise=FALSE){
  ylim<-range(c(model$evaluation_log[,2][[1]], model$evaluation_log[,3][[1]]))
  plot(model$evaluation_log[,2][[1]], lwd=1, type="l", ylim=ylim)
  points(model$evaluation_log[,3][[1]], lwd=1, type="l")
  if (maximise==TRUE) {
    print(max(model$evaluation_log[,3][[1]]))
    abline(v=which.max(model$evaluation_log[,3][[1]]), col="red")
  } else {
    print(min(model$evaluation_log[,3][[1]]))
    abline(v=which.min(model$evaluation_log[,3][[1]]), col="red")
  }
}
printLearningCurve(modeleval, maximise = TRUE)








evaluateXgbFactor<-function(model, newdata, newlabels, label="") {
  preds<-predict(model, newdata=newdata)
  predlabels<-levels(trainLabels)[preds+1]
  gspred<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  gcpred<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))
  comparedata<-as.data.frame(as.matrix(newlabels))
  comparedata$where=as.factor(ifelse(newdata[,"whereHome"]==1, "Home", "Away"))
  summarizePrediction(data = comparedata, (gspred), (gcpred), label)
  return(data.frame(gspred, gcpred))
}
evaluateXgbFactor(modeleval, testMatrix$)

objective<-function(preds, labels)
{
  print((preds))
  print(str(preds))
  print((labels))
  print(str(labels))
  grad<-preds*1.0
  hess<-preds*0
  return(list(grad=grad, hess=hess))
}

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

xgbparamsFactor<-list(objective = "multi:softmax", gamma=10, max_depth = 3, eta = 0.01, subsample=0.3, colsample_bytree=0.05, nthread = 1, num_class=26)
model_cv_xg_Score <- xgboost(obj=objective, params=xgbparamsFactor, data = trainXM, label=trainLabels, print_every_n=1, nrounds=2)

modeleval<-xgb.train(verbose=1, params = xgbparamsFactor, data = xgb.DMatrix(trainXM, label=trainLabels) , nrounds =5, feval=evalerror, maximize=FALSE, print_every_n=1)
modeleval<-xgboost(verbose=1, params = xgbparamsFactor, data = xgb.DMatrix(trainXM, label=trainLabels) , nrounds =5, feval=evalerror, maximize=FALSE, print_every_n=1)

where<-getinfo(xgb.DMatrix(trainXM, label=trainLabels), "whereHome")

(levels(trainLabels)[getinfo(xgb.DMatrix(trainXM, label=trainLabels), "label")])



evaluateXgbFactor<-function(model, newdata, newlabels, label="", levels1=levels(buildFactorScore(train)$score)) {
  comparedata<-as.data.frame(as.matrix(newlabels))
  comparedata$where=as.factor(ifelse(newdata[,"whereHome"]==1, "Home", "Away"))
  pred<-predict(model, newdata=newdata)
  pred<-levels1[pred]
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  summarizePrediction(data = comparedata, gspred, gcpred, label)
  return(data.frame(gspred, gcpred))
}
result<-evaluateXgbFactor(model_cv_xg_Score, cbind(trainXM, as.matrix(train_pred_poiss)), trainYM, "xbg multi:softmax train")
result<-evaluateXgbFactor(model_cv_xg_Score, cbind(testXM, as.matrix(test_pred_poiss)), testYM, "xbg multi:softmax test")

summarizePrediction(data = test, test$GS, test$GC, "100% correct") # 1618
summarizePrediction(data = test, ifelse(test$where=="Home", 2, 1), ifelse(test$where=="Home", 1, 2), "2:1 home win") # 411
summarizePrediction(data = test, ifelse(test$where=="Home", 1, 2), ifelse(test$where=="Home", 2, 1), "1:2 away win") # 407

homegames<-train %>% filter(where=="Home") %>% select(GS, GC)
randidx<-sample(nrow(homegames), size = nrow(test), replace=TRUE)
summarizePrediction(data = test, 
                    ifelse(test$where=="Home", homegames[randidx,1], homegames[randidx,2]), 
                    ifelse(test$where=="Home", homegames[randidx,2], homegames[randidx,1]), "random draw from distribution") # 340




summarizePrediction(data = test, test$GS, test$GC, "random guess")






levels(trainLabels)
table(predict(model_cv_xg_Score, newdata=trainXM), trainLabels)

table(predict(model_cv_xg_Score, newdata=trainXM))
table(trainLabels)


str(trainXM)
str(trainYM)

summary(predict(model_cv_xg_GS, trainXM))
summary(predict(model_cv_xg_GC, trainXM))


evaluateNumeric<-function(model, newdata, label="") {
  pred<-predict(model, newdata=newdata)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  summarizePrediction(data = newdata, round(gspred), round(gcpred), label)
  return(data.frame(gspred, gcpred))
}


xgb.DMatrix(train)

data_all_multifactor<-cbind(data_raw, score=data_multifactor$score) %>% select(-GS, -GC)

model6<-cforest(score ~ ., data=data_all_multifactor, controls = cforest_unbiased(ntree=500, mtry=5))

pred6<-predict(model6, newdata = data_all_multifactor )
gspred6<-as.integer(substr(pred6,1,1))
gcpred6<-as.integer(substr(pred6,3,3))
result<-summarizePrediction(data = train, gspred6, gcpred6)

model7<-cforest(score ~ ., data=data_multifactor %>% select(score, gspred, gcpred), 
                controls = cforest_unbiased(ntree=500, mtry=2))
pred7<-predict(model7, newdata = data_multifactor )
gspred7<-as.integer(substr(pred7,1,1))
gcpred7<-as.integer(substr(pred7,3,3))
result<-summarizePrediction(data = train, gspred7, gcpred7)


# model_cv_goals1 <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree=1000, mtry=20))
# model_cv_goals2 <- cforest(GS*GC ~ ., data=train%>%filter(GS!=GC), controls = cforest_unbiased(ntree=1000, mtry=10))
# model_cv_goals3 <- cforest(GS*GC ~ ., data=train%>%select(GS, GC, team, oppTeam, where) , controls = cforest_unbiased(ntree=1000, mtry=2))
# model_cv_goals4 <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree=500, mtry=5))
# 
# pred1<-predict(model_cv_goals1, newdata = train)
# gspred1<-sapply(pred1, function(x) x[,'GS'])
# gcpred1<-sapply(pred1, function(x) x[,'GC'])
# result<-summarizePrediction(xy = train, round(gspred1), round(gcpred1))
# pred2<-predict(model_cv_goals2, newdata = train)
# gspred2<-sapply(pred2, function(x) x[,'GS'])
# gcpred2<-sapply(pred2, function(x) x[,'GC'])
# result<-summarizePrediction(xy = train, round(gspred2), round(gcpred2))
# pred3<-predict(model_cv_goals3, newdata = train)
# gspred3<-sapply(pred3, function(x) x[,'GS'])
# gcpred3<-sapply(pred3, function(x) x[,'GC'])
# result<-summarizePrediction(xy = train, round(gspred3), round(gcpred3))
# pred4<-predict(model_cv_goals4, newdata = train)
# gspred4<-sapply(pred4, function(x) x[,'GS'])
# gcpred4<-sapply(pred4, function(x) x[,'GC'])
# result<-summarizePrediction(xy = train, round(gspred4), round(gcpred4))
# 
# data_is_draw<-train %>% mutate(isDraw=as.factor(1-abs(sign(GS-GC)))) %>% select(-GS, -GC)
# model_is_draw<-cforest(isDraw ~ ., data=data_is_draw, controls = cforest_unbiased(ntree=1000, mtry=3))
# drawPred<-predict(model_is_draw, newdata=data_is_draw, type="prob")
# drawProb<-t(sapply(drawPred, cbind))[,2]
# 
# data_train8<-train %>% 
#   mutate(eg1=ifelse(GS>4,GS-4,0), eg2=ifelse(GC>4,GC-4,0)) %>% 
#   mutate(eg3=ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0), eg4=ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)) %>%
#   mutate(score=factor(paste(eg3, eg4, sep = ":"), 
#                       levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
#                                "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
#                                "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE))%>% 
#   select(score, team, oppTeam, where)
# data_train8<-data.frame(data_train8, gspred1, gcpred1, gspred2, gcpred2, gspred3, gcpred3, gspred4, gcpred4
#                         #, drawProb
#                         )
# 
# model8<-cforest(score ~ ., data=data_train8, controls = cforest_unbiased(ntree=1000, mtry=5))
# #sort(varimp(model8), decreasing = TRUE)
# 
# pred8<-predict(model8, newdata = data_train8 )
# gspred8<-as.integer(substr(pred8,1,1))
# gcpred8<-as.integer(substr(pred8,3,3))
# result<-summarizePrediction(xy = train, gspred8, gcpred8)
# 
gc(TRUE)

################################################################################
# evaluate oob test data scores

result_test<-evaluateNumeric(model_cv_goals, newdata = test, label = "Model 1 Test")
data_test<-data.frame(gspred=result_test$gspred, gcpred=result_test$gcpred, test)
result_test2<-evaluateNumeric(model2, newdata = data_test, label = "Model 2 Test")
result_test3<-evaluateNumeric(model3, newdata = test, label = "Model 3 Test")
data_test4<-data.frame(data_test, gspred3=result_test3$gspred, gcpred3=result_test3$gcpred)
result_test4<-evaluateFactor(model4, newdata = data_test4, label = "Model 4 Test")
result_test5<-evaluateMaxExpectation(model4, newdata = data_test4, label = "Model 4 MaxExpect Test")
result_test51<-evaluateFactor(model5, newdata = test, label = "Model 5 Test")
result_test52<-evaluateMaxExpectation(model5, newdata = test, label = "Model 5 MaxExpect")



# pred3<-predict(model3, newdata=test, type="prob")
# pred3Draws<-t(sapply(pred3, cbind))
# summary(pred3Draws[,2])
# boxplot(pred3Draws[,2]  ~ test$GS==test$GC)
# hist(pred3Draws[,2])
# smoothScatter(pred3Draws[,2], test$GS==test$GC)




pred6<-predict(model6, newdata = data4 )
gspred6<-as.integer(substr(pred6,1,1))
gcpred6<-as.integer(substr(pred6,3,3))
result<-summarizePrediction(data = test, gspred6, gcpred6)

pred7<-predict(model7, newdata = data4 )
pred7[pred7=="4:0"]<-"3:0"
pred7[pred7=="0:4"]<-"0:3"
gspred7<-as.integer(substr(pred7,1,1))
gcpred7<-as.integer(substr(pred7,3,3))
result<-summarizePrediction(data = test, gspred7, gcpred7)

#result<-summarizePrediction(data = test, rep(2, nrow(test)), rep(1, nrow(test)))
#dummy<-data.frame(where=test$where, gspred=2, gcpred=1)
#dummy$gspred[dummy$where=="Away"]<-1
#dummy$gcpred[dummy$where=="Away"]<-2
#result<-summarizePrediction(data = test, dummy$gspred, dummy$gcpred)


pred1<-predict(model_cv_goals1, newdata = test)
gspred1<-sapply(pred1, function(x) x[,'GS'])
gcpred1<-sapply(pred1, function(x) x[,'GC'])
result<-summarizePrediction(data = test, round(gspred1), round(gcpred1))
pred2<-predict(model_cv_goals2, newdata = test)
gspred2<-sapply(pred2, function(x) x[,'GS'])
gcpred2<-sapply(pred2, function(x) x[,'GC'])
result<-summarizePrediction(data = test, round(gspred2), round(gcpred2))
pred3<-predict(model_cv_goals3, newdata = test)
gspred3<-sapply(pred3, function(x) x[,'GS'])
gcpred3<-sapply(pred3, function(x) x[,'GC'])
result<-summarizePrediction(data = test, round(gspred3), round(gcpred3))
pred4<-predict(model_cv_goals4, newdata = test)
gspred4<-sapply(pred4, function(x) x[,'GS'])
gcpred4<-sapply(pred4, function(x) x[,'GC'])
result<-summarizePrediction(data = test, round(gspred4), round(gcpred4))

cv_data_is_draw<-test %>% mutate(isDraw=as.factor(1-abs(sign(GS-GC)))) %>% select(-GS, -GC)
cv_drawPred<-predict(model_is_draw, newdata=cv_data_is_draw, type="prob")
cv_drawProb<-t(sapply(cv_drawPred, cbind))[,2]

data_test8<-data.frame(test, gspred1, gcpred1, gspred2, gcpred2, gspred3, gcpred3, gspred4, gcpred4
                       #, drawProb=cv_drawProb
                       )

pred8<-predict(model8, newdata = data_test8 )
gspred8<-as.integer(substr(pred8,1,1))
gcpred8<-as.integer(substr(pred8,3,3))
result<-summarizePrediction(data = test, gspred8, gcpred8)

predProbsList<-predict(model8, newdata = data_test8 , type = "prob")
predoutcomes<-t(sapply(predProbsList, cbind))
colnames(predoutcomes)<-colnames(predProbsList[[1]])
predoutcomes<-predoutcomes[,order(colnames(predoutcomes))]  
colnames(predoutcomes)<-substr(colnames(predoutcomes), 7,9)
substr(colnames(predoutcomes), 2,2)<-":"

matchidx<-match(paste(test$GS, test$GC, sep = ":"), colnames(predoutcomes))
matchedprobs<-predoutcomes[cbind(1:length(matchidx), matchidx)]
summary(matchedprobs)

matchidx8<-match(paste(gspred8, gcpred8, sep = ":"), colnames(predoutcomes))
matchedprobs8<-predoutcomes[cbind(1:length(matchidx8), matchidx8)]
summary(matchedprobs8)

mask<-mask[order(names(mask))]
expectedValues<-sapply(1:25, function(i) predoutcomes %*% as.vector(t(mask[i][[1]])))
colnames(expectedValues)<-names(mask)

maxEV<-apply(expectedValues, 1, max)
whichMaxEV<-apply(expectedValues, 1, which.max)

pred9<-colnames(expectedValues)[whichMaxEV]
pred9[pred9=="4 0"]<-"3 0"
pred9[pred9=="0 4"]<-"0 3"
gspred9<-as.integer(substr(pred9,1,1))
gcpred9<-as.integer(substr(pred9,3,3))
result<-summarizePrediction(data = test, gspred9, gcpred9)


varimp_model <- cforest(I(GS-GC) ~ ., data=train, controls = cforest_unbiased(ntree=1000, mtry=20))
vi<-sort(varimp(varimp_model), decreasing = TRUE)
plot(vi)




