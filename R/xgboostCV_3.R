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


predict_from_model1 <- function(model1, newdata) {
  train_pred1<-predict(model1, newdata=newdata)
  train_pred1<-matrix(data=unlist(train_pred1), ncol=2, byrow=TRUE, dimnames = list(NULL, c("pGS","pGC")))
  home<-cbind(newdata, train_pred1) %>% filter(where=="Home") %>% 
    select(GS, GC, pGS, pGC, team, oppTeam, round, gameindex, season) %>% mutate(pGDiff=pGS-pGC)
  away<-cbind(newdata, train_pred1) %>% filter(where=="Away") %>% 
    select(GS=GC, GC=GS, pGS=pGC, pGC=pGS, team=oppTeam, oppTeam=team, round, gameindex, season) %>% mutate(pGDiff=pGS-pGC)
  games<-inner_join(home, away, by=c("team", "oppTeam", "round", "gameindex", "season", "GS", "GC")) 
  return(games)
}



################################################################################
# build models and evaluate training data scores


evaluateXgbNumeric<-function(modelGS, modelGC, newdata, newlabels, label="") {
  gspred<-predict(modelGS, newdata=newdata)
  gcpred<-predict(modelGC, newdata=newdata)
  comparedata<-as.data.frame(as.matrix(newlabels))
  comparedata$where=as.factor(ifelse(newdata[,"whereHome"]==1, "Home", "Away"))
  summarizePrediction(data = comparedata, round(gspred), round(gcpred), label)
  return(data.frame(gspred, gcpred))
}

xgbLinear<-function(){
  
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
  
}

##########################################################################################################################

evaltarget <- function(preds, dtrain) {
  labels<-levels(gLev)[getinfo(dtrain, "label")+1]
  predlabels<-levels(gLev)[preds+1]
  HG<-as.integer(substr(as.character(labels), start = 1, stop = 1))
  AG<-as.integer(substr(as.character(labels), start = 3, stop = 3))
  pHG<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  pAG<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))
  #print(paste(HG, AG, sep=":"))
  #print(paste(pHG, pAG, sep=":"))
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

prediction_estimate<-function(){
  lGS<-as.integer(substr(as.character(gLev), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(gLev), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(gLev), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(gLev), start = 3, stop = 3))
  p1<-expand.grid(lGS, pGS)
  p2<-expand.grid(lGC, pGC)
  p3<-ifelse(p1$Var1==p1$Var2 & p2$Var1==p2$Var2, 1, 0) # full hit
  p4<-abs((p1$Var1-p2$Var1) - (p1$Var2-p2$Var2))*-0.4 + 2 # goal diff score 
  p5<-ifelse(sign(p1$Var1-p2$Var1) == sign(p1$Var2-p2$Var2), 0.5, 0) # tendendy score 
  p6<-abs((p1$Var1+p2$Var1) - (p1$Var2+p2$Var2))*-0.05 + 0.7 # total goals  score 
  p7<-(2*p3+3*p4+p5+p6)
  predest<-matrix(p7, ncol=25, byrow = TRUE)
  colnames(predest)<-gLev
  rownames(predest)<-gLev
  predest<-predest/rowSums(predest)
}

prediction_estimate<-function(){
  lGS<-as.integer(substr(as.character(gLev), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(gLev), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(gLev), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(gLev), start = 3, stop = 3))
  p1<-expand.grid(lGS, pGS)
  p2<-expand.grid(lGC, pGC)
  pp<-dpois(x=p1$Var1, lambda = (p1$Var2+ifelse(p1$Var2==0, 0.85, 0.25) )) * dpois(x=p2$Var1, lambda = (p2$Var2+ifelse(p2$Var2==0, 0.6, 0.05))) 
  predest<-matrix(pp, ncol=25, byrow = TRUE)
  colnames(predest)<-gLev
  rownames(predest)<-gLev
  predest<-predest-apply(predest, 1, min)
  predest<-predest/rowSums(predest)
}

# predest<-prediction_estimate()
# sort(colSums(predest))
# sort(predest["0:0",])
# sort(predest["1:1",])
# sort(predest["2:2",])
# sort(predest["0:1",])
# sort(predest["0:2",])
# sort(predest["1:2",])
# sort(predest["1:0",])
# sort(predest["2:0",])
# sort(predest["2:1",])
# sort(predest["0:4",])
# sort(predest["3:4",])

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


plotImportance<-function(model, n=ncol(trainXM), cols=colnames(trainXM)){
  mat <- xgb.importance (feature_names = cols, model = model)
  xgb.plot.importance (importance_matrix = mat[1:n]) 
}

printImportance<-function(model, n=ncol(trainXM), cols=colnames(trainXM)){
  mat <- xgb.importance (feature_names = cols, model = model)
  if (n==length(cols))
    print.data.frame (mat) 
  else
    print.data.frame (mat[1:n]) 
  return(mat)
}

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

evaluateXgbFactorSimple<-function(model, newdata, newlabels, label="", levels1=gLev) {
  comparedata<-as.data.frame(as.matrix(newlabels))
  pred<-predict(model, newdata=newdata)
  pred<-levels1[pred+1]
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  # summarizePredictionSimple(data = comparedata, gspred, gcpred, label)
  return(data.frame(gspred, gcpred))
}

summarizePredictionSimple<-function(data, gspred, gcpred, label="", verbose=1) {
  
  resultKPIs<-data.frame(pred_tend=sign(gspred-gcpred), act_tend=sign(data$GS-data$GC), 
                         pred_diff=gspred-gcpred, act_diff=data$GS-data$GC, 
                         pred_GS=gspred, pred_GC=gcpred, act_GS=data$GS, act_GC=data$GC,
                         row.names = NULL) %>% 
    mutate(
      pred_score=factor(paste(pred_GS, pred_GC, sep = ":"), 
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
    bidpoints[tendhit & act_tend ==  1]=2
    bidpoints[tendhit & act_tend == -1]=4
    bidpoints[gdhit   & act_tend ==  1]=3
    bidpoints[gdhit   & act_tend ==  0]=2
    bidpoints[gdhit   & act_tend == -1]=5
    bidpoints[fullhit & act_tend ==  1]=4
    bidpoints[fullhit & act_tend ==  0]=6
    bidpoints[fullhit & act_tend == -1]=7
  })
  
  scoreTable<-resultKPIs %>% select(pred_score, bidres) %>% table
  
  summaryTable<-resultKPIs %>% 
    summarise(tend_pct=mean(tendhit), diff_pct=mean(gdhit), exact_pct=mean(fullhit), avg_points=mean(bidpoints), total_points=sum(bidpoints)) 
  
  resultKPIs %>% select (x=pred_score, y=bidres) %>% 
    plot(xlab=label, main=
           summaryTable %>% transmute(round(tend_pct, 3)*100, round(diff_pct, 3)*100, round(exact_pct, 3)*100, round(avg_points, 2), total_points) %>% paste(collapse=" / ")
    )
  if (verbose>0)
  {
    print(resultKPIs %>% select(pred_GS, pred_GC) %>% table)
    print(resultKPIs %>% select(pred_tend, act_tend) %>% table)
    print(resultKPIs %>% select(pred_diff) %>% table)
    print(resultKPIs %>% select(pred_diff, act_diff) %>% table)
    print(table(resultKPIs$pred_score, paste(resultKPIs$act_GS, resultKPIs$act_GC, sep=":")))
    print(scoreTable)
    print(colSums(scoreTable)/sum(scoreTable)*100)
    print(scoreTable/rowSums(scoreTable)*100)
    print(label)
    print(summaryTable %>% data.frame)
  }
  return(summaryTable)
}  



calcWeightedPoints<-function(model, newdata){
  pred_params<-predict(object=model, newdata = newdata, reshape = TRUE, outputmargin = TRUE)
  colnames(pred_params)<-gLev
  pred_prob<-exp(pred_params)
  pred_prob<-pred_prob/rowSums(pred_prob)
  
  ylabels<-getinfo(newdata, "label")+1
  wpoints<-rowSums(pred_prob * maskm$home[ylabels,])
  return(wpoints)
}
 
printLearningCurve<-function(model, maximise=FALSE, cutoff=1, info=list(), data=NULL){
  if (!is.null(data)){
    train_means<-data[,2][[1]]
    test_means<-data[,3][[1]]
  }
  else {
    train_means<-model$evaluation_log[,2][[1]]
    test_means<-model$evaluation_log[,3][[1]]
  }
  if (maximise==TRUE) {
    best_mean<-max(test_means)
    best_iter<-which.max(test_means)
  } else {
    best_mean<-min(test_means)
    best_iter<-which.min(test_means)
  }
  l<-length(train_means)
  if (cutoff>l) 
    cutoff<-l
  ylim<-range(c(test_means[cutoff:l]))
  plot(test_means, lwd=2, type="l", ylim=ylim, 
       main=paste(best_mean, best_iter, sep=" / "), 
       #sub=printtext, 
       xlab="Iteration", ylab="TestScore", col="blue")
  printdata<-as.data.frame(c(model$params, best_iter=best_iter, info))
  printtexts<-paste(names(printdata), printdata, sep="=")
  abline(v=best_iter, col="red")
  abline(h=best_mean, col="red", lty="dashed", lwd=2)
  #legend("bottomleft", legend = paste(names(printdata), printdata, sep="="), cex=0.7, bty = "n")
  mtext(paste(printtexts[1:6], collapse = ", "), cex=0.7, line = 0, outer = FALSE)
  mtext(paste(printtexts[7:length(printtexts)], collapse = ", "), cex=0.7, line = -1, outer = FALSE)
  
  par(new = T)
  plot(train_means, lwd=1, type="l", axes=FALSE, xlab=NA, ylab=NA)
  axis(side = 4)
  mtext(side = 4, line = 3, 'Train Score')
}

printLearningCurveCV<-function(model, maximise=FALSE, cutoff=1, info=list()){
  iteration<-model$evaluation_log[,1][[1]]
  train_means<-model$evaluation_log[,2][[1]]
  train_sd<-model$evaluation_log[,3][[1]]
  test_means<-model$evaluation_log[,4][[1]]
  test_sd<-model$evaluation_log[,5][[1]]
  if (maximise==TRUE) {
    best_mean<-max(test_means)
    best_iter<-which.max(test_means)
  } else {
    best_mean<-min(test_means)
    best_iter<-which.min(test_means)
  }
  l<-length(train_means)
  printdata<-as.data.frame(c(model$params, best_iter=best_iter, info))
  printtexts<-paste(names(printdata), printdata, sep="=")
  ylim<-range(c(train_means[cutoff:l], test_means[cutoff:l]+test_sd[cutoff:l], test_means[cutoff:l]-test_sd[cutoff:l]))
  plot(train_means, lwd=1, type="l", ylim=ylim, 
       main=paste(best_mean, test_sd[best_iter]), 
       #sub=printtext, 
       xlab="Iteration", ylab="Score")
  polygon(c(rev(iteration), iteration), c(rev(test_means+0.5*test_sd), test_means-0.5*test_sd), 
          col = adjustcolor('lightblue', alpha.f = 0.3), border = "darkblue", lty = 'dotted')
  polygon(c(rev(iteration), iteration), c(rev(train_means+0.5*train_sd), train_means-0.5*train_sd), 
          col = adjustcolor('grey80', alpha.f = 0.3), border = "darkgray", lty = 'dotted')
  points(train_means, lwd=1, type="l", ylim=ylim)
  points(test_means, lwd=1, type="l", col="blue")
  abline(v=best_iter, col="red")
  abline(h=best_mean, col="red", lty="dashed")
  #legend("bottomleft", legend = paste(names(printdata), printdata, sep="="), cex=0.7, bty = "n")
  mtext(paste(printtexts[1:6], collapse = ", "), cex=0.7, line = 0, outer = FALSE)
  mtext(paste(printtexts[7:length(printtexts)], collapse = ", "), cex=0.7, line = -1, outer = FALSE)
}

printSoftMax <- function(model3, testMatrix, trainMatrix) {
  pred_test_pred<-predict(object=model3, newdata = testMatrix, reshape = TRUE, outputmargin = TRUE)
  pred_train_pred<-predict(object=model3, newdata = trainMatrix, reshape = TRUE, outputmargin = TRUE)
  colnames(pred_test_pred)<-gLev
  colnames(pred_train_pred)<-gLev
  pred_test_prob<-exp(pred_test_pred)
  pred_test_prob<-pred_test_prob/rowSums(pred_test_prob)
  pred_train_prob<-exp(pred_train_pred)
  pred_train_prob<-pred_train_prob/rowSums(pred_train_prob)
  par(mfrow=c(2,2))
  plot(value~Var2, data=melt(pred_test_pred), main="Test Outputs")
  plot(value~Var2, data=melt(pred_test_prob), main="Test Probs", xlab=NA)
  abline(h=0.04, col="red")
  plot(value~Var2, data=melt(pred_train_pred), main="Train Outputs", xlab=NA, ylim=range(pred_train_pred))
  plot(value~Var2, data=melt(pred_train_prob), main="Train Probs", xlab=NA)
  abline(h=0.04, col="red")
  par(mfrow=c(1,1))
}

calcSoftMaxProb <- function(preds) {
  m<-matrix(preds, ncol=nlevels(gLev))
  colnames(m)<-gLev
  m<-exp(m)
  probs<-m/rowSums(m)
  maxProbs<-apply(probs, 1, max)
  return(maxProbs)
}

mav <- function(x,n=5){stats::filter(x,rep(1/n,n), sides=2)}

printBoostingScores <- function(wpoint_log, evaluation_log) {
  if (is.null(wpoint_log)){
    return()
  }
  n<-nrow(evaluation_log)
  maxindex<-which.max(mav(evaluation_log$test_avgpoints))
  par(mfrow=c(2,2))
  # train_maxp
  plot(wpoint_log$train_wpoints, ylim=c(min(wpoint_log$test_wpoints),max(evaluation_log$train_avgpoints)), type="l", col="red")
  points(wpoint_log$test_wpoints, col="green", type="l")
  points(evaluation_log$train_avgpoints, type="l")
  points(evaluation_log$test_avgpoints, col="blue", type="l")
  abline(h=max(evaluation_log$test_avgpoints), lty="dotted", col="blue")
  points(n, evaluation_log$test_avgpoints[n], col="blue", type="p", pch=1, cex=3)
  points(maxindex, evaluation_log$test_avgpoints[maxindex], col="blue", type="p", pch=1, cex=3, lty="dotted")
  points(evaluation_log$train_avgpoints / wpoint_log$train_wpoints, type="l", lty="dashed")
  points(evaluation_log$test_avgpoints / wpoint_log$test_wpoints, type="l", lty="dashed", col="blue")
  points(evaluation_log$train_avgpoints - wpoint_log$train_wpoints, type="l", lty=6)
  points(evaluation_log$test_avgpoints - wpoint_log$test_wpoints, type="l", lty=6, col="blue")
  par(new = T)
  plot(wpoint_log$train_maxp, lwd=1, type="l", axes=FALSE, xlab=NA, ylab=NA, ylim=c(0,1), lty=5)
  points(wpoint_log$test_maxp, lwd=1, type="l", xlab=NA, ylab=NA, ylim=c(0,1), lty=5, col="blue")
  axis(side = 4)
  #mtext(side = 4, line = 3, 'Probabilities')
  
  plot(wpoint_log$train_wpoints, wpoint_log$test_wpoints, col="green", type="l", ylim=c(min(wpoint_log$test_wpoints),max(evaluation_log$train_avgpoints)))
  points(wpoint_log$train_wpoints, evaluation_log$train_avgpoints, type="l")
  points(wpoint_log$train_wpoints, evaluation_log$test_avgpoints, col="blue", type="l")
  abline(h=max(evaluation_log$test_avgpoints), lty="dotted", col="blue")
  points(wpoint_log$train_wpoints[n], evaluation_log$test_avgpoints[n], col="blue", type="p", pch=1, cex=3)
  points(wpoint_log$train_wpoints[maxindex], evaluation_log$test_avgpoints[maxindex], col="blue", type="p", pch=1, cex=3, lty="dotted")
  points(wpoint_log$train_wpoints, evaluation_log$train_avgpoints / wpoint_log$train_wpoints, type="l", lty="dashed")
  points(wpoint_log$train_wpoints, evaluation_log$test_avgpoints / wpoint_log$test_wpoints, type="l", lty="dashed", col="blue")
  par(new = T)
  plot(wpoint_log$train_wpoints, wpoint_log$train_maxp, lwd=1, type="l", axes=FALSE, xlab=NA, ylab=NA, ylim=c(0,1), lty=5)
  points(wpoint_log$train_wpoints, wpoint_log$test_maxp, lwd=1, type="l", xlab=NA, ylab=NA, ylim=c(0,1), lty=5, col="blue")
  axis(side = 4)
  #mtext(side = 4, line = 3, 'Probabilities')
  
  plot(evaluation_log$train_avgpoints, wpoint_log$train_wpoints, 
       ylim=c(min(wpoint_log$test_wpoints),max(evaluation_log$test_avgpoints)), 
       xlim=c(range(evaluation_log$train_avgpoints[-1:-10])),
       type="p", col="red", pch="+")
  points(evaluation_log$train_avgpoints, wpoint_log$test_wpoints, col="green", type="p")
  points(evaluation_log$train_avgpoints, evaluation_log$test_avgpoints, col="blue", type="p", pch="+")
  abline(h=max(evaluation_log$test_avgpoints), lty="dotted", col="blue")
  points(evaluation_log$train_avgpoints[n], evaluation_log$test_avgpoints[n], col="blue", type="p", pch=1, cex=3)
  points(evaluation_log$train_avgpoints[maxindex], evaluation_log$test_avgpoints[maxindex], col="blue", type="p", pch=1, cex=3, lty="dotted")
  #points(evaluation_log$train_avgpoints, evaluation_log$train_avgpoints / wpoint_log$train_wpoints, type="l", lty="dashed")
  #points(evaluation_log$train_avgpoints, evaluation_log$test_avgpoints / wpoint_log$test_wpoints, type="l", lty="dashed", col="blue")
  par(new = T)
  plot(evaluation_log$train_avgpoints,wpoint_log$train_maxp, lwd=1, type="l", axes=FALSE, xlab=NA, ylab=NA, ylim=c(0,1), lty=5)
  points(evaluation_log$train_avgpoints,wpoint_log$test_maxp, lwd=1, type="l", xlab=NA, ylab=NA, ylim=c(0,1), lty=5, col="blue")
  axis(side = 4)
  #mtext(side = 4, line = 3, 'Probabilities')
  
  plot(wpoint_log$test_wpoints, wpoint_log$train_wpoints, ylim=c(min(wpoint_log$train_wpoints),max(evaluation_log$train_avgpoints)), type="l", col="red")
  points(wpoint_log$test_wpoints, evaluation_log$train_avgpoints, type="l")
  points(wpoint_log$test_wpoints, evaluation_log$test_avgpoints, col="blue", type="l")
  abline(h=max(evaluation_log$test_avgpoints), lty="dotted", col="blue")
  points(wpoint_log$test_wpoints[n], evaluation_log$test_avgpoints[n], col="blue", type="p", pch=1, cex=3)
  points(wpoint_log$test_wpoints[maxindex], evaluation_log$test_avgpoints[maxindex], col="blue", type="p", pch=1, cex=3, lty="dotted")
  points(wpoint_log$test_wpoints, evaluation_log$train_avgpoints / wpoint_log$train_wpoints, type="l", lty="dashed")
  points(wpoint_log$test_wpoints, evaluation_log$test_avgpoints / wpoint_log$test_wpoints, type="l", lty="dashed", col="blue")
  par(new = T)
  plot(wpoint_log$test_wpoints, wpoint_log$train_maxp, lwd=1, type="l", axes=FALSE, xlab=NA, ylab=NA, ylim=c(0,1), lty=5)
  points(wpoint_log$test_wpoints, wpoint_log$test_maxp, lwd=1, type="l", xlab=NA, ylab=NA, ylim=c(0,1), lty=5, col="blue")
  axis(side = 4)
  #mtext(side = 4, line = 3, 'Probabilities')
  
  par(mfrow=c(1,1))
}


calcScores <- function(allScores, model3) {
  actualGoals<-allScores %>% select(GS, GC)
  trainResult<-evaluateXgbFactorSimple(model3, trainMatrix, actualGoals[!allScores$oob,], "train XGB objective function")
  
  trainGS<-trainResult$gspred
  trainGC<-trainResult$gcpred
  trainScore<-mean(calcPoints(actualGoals$GS[!allScores$oob], actualGoals$GC[!allScores$oob], trainGS, trainGC, home=TRUE))
  
  testResult<-evaluateXgbFactorSimple(model3, testMatrix, actualGoals[allScores$oob,], "test XGB objective function")
  testGS<-testResult$gspred
  testGC<-testResult$gcpred
  testScore<-mean(calcPoints(actualGoals$GS[allScores$oob], actualGoals$GC[allScores$oob], testGS, testGC, home=TRUE))
  return(data.frame(train_avgpoints=trainScore, test_avgpoints=testScore))
}


################################################################################
# data preparation for learning

mask<-buildMask()
gLev<-sort(buildFactorScore(data.frame(expand.grid(GS=0:4, GC=0:4)))$score)
predest<-prediction_estimate()
maskm<-buildMaskMatrix()

xy<-read.csv(file="BLfeatures_2.csv", stringsAsFactors = TRUE, as.is = FALSE)

train_num<-xy
train_factor<-limitMaxScore(xy, 4) %>% buildFactorScore()
labels_num<-train_num %>% select(GS, GC)
labels_factor<-train_factor$score 
folds <-split(seq_along(xy$season), xy$season)

parameters<-data.frame(
  ntree=2000,
  mtry=3, # mtry=2,
  minsplit = 15,
  nparallel = 10
)
print (parameters)

cforest_ctrl<-cforest_unbiased(
  ntree=parameters$ntree,
  mtry=parameters$mtry,
  minsplit = parameters$minsplit
)

fold_models<-list()
for (fn in names(folds)){
  print(sprintf("Fold: %s", fn))
  f<-folds[[fn]]
  train_num_fold<-train_num
  train_num_fold$oob<-FALSE
  train_num_fold$oob[f]<-TRUE
  # side_info<-train_num_fold %>% 
  #   select(gameindex, round, season, team, oppTeam, where,
  #          mt_Shotstarget, mt_Shotstarget_where, t_diffRank, t_diffGoals_where, t_diffBothGoals_where,
  #          pois1_Shots, SVD_GDiff, t_diffGoals, mt_Shots, t_diffBothGoals, mt_Shots_where,
  #          t_diffBothGoals2H_where, SVD1_oppFouls, mt_Yellow_where, SVD_Shotstarget, 
  #          starts_with("t_sc_team_"), oob)
  sidehome<-train_num_fold %>% filter(where=="Home") %>% select(-where) %>%
    select(team, oppTeam, round, gameindex, season, oob)
  # sideaway<-side_info %>% filter(where=="Away") %>% mutate(z = team, team=oppTeam, oppTeam=z) %>% select(-where, -z)
  # side_info_joined<-inner_join(sidehome, sideaway, by=c("team", "oppTeam", "round", "gameindex", "season", "oob")) 
  
  fold_models[[fn]]<-list(models=list())
  rf_predictions<-sidehome
  for (p in 1:parameters$nparallel){
    seed <- sample(1:2^15, 1)
    set.seed(seed)
    print(seed)
    columns<-sample(13:ncol(train_num), size = 100, replace = FALSE)
    train_data_rf<-train_num[-f,] %>%
      select(GS, GC, team, oppTeam, round, gameindex, season, contains("promoted"), columns)
    new_data_rf<-train_num %>%
      select(where, GS, GC, team, oppTeam, round, gameindex, season, contains("promoted"), columns)
    
    model1 <- cforest(GS*GC ~ ., data=train_data_rf, controls = cforest_ctrl)
    model1_prediction <- predict_from_model1(model1, newdata=new_data_rf)
    colnames(model1_prediction)<-gsub("\\.x", paste0(".x",p), colnames(model1_prediction))
    colnames(model1_prediction)<-gsub("\\.y", paste0(".y",p), colnames(model1_prediction))
    rf_predictions<-inner_join(rf_predictions, model1_prediction %>% select(-GS, -GC), by=c("team", "oppTeam", "round", "gameindex", "season"))
    fold_models[[fn]][["models"]][[p]]<-list(seed=seed)
    model1<-NULL
    gc()
  }
  
  allScores<-
    inner_join(rf_predictions, 
               train_num %>% filter(where=="Home") %>% select(GS, GC, gameindex, round, season, team, oppTeam), 
               by=c("team", "oppTeam", "round", "gameindex", "season")
    ) %>%
    limitMaxScore(4) %>% buildFactorScore() 

  fold_models[[fn]][["allScores"]]<-allScores
}

saveRDS(fold_models, file = "xgb\\model3_outputs.rds")  
fold_models<-readRDS(file = "xgb\\model3_outputs.rds")  

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
for (i in 1:10) {
  eff_seed <- sample(1:2^15, 1)
  print(sprintf("Seed for session: %s", eff_seed))
  set.seed(eff_seed)
  #set.seed(2761)
  
  parameters<-data.frame(
    seed=eff_seed,
    gamma=0, #sample(c(0, 0.2, 0.5, 1, 2, 5),1), # 0,1,3,5, 0,10,15,20,25,40 
    max_depth = 1, #sample(c(1,2,3),1),#, 40, # 1,2,3,4,5,10, 20,30, 40,50,80  ,20,30
    eta = sample(c(0.1,0.03, 0.01, 0.003),1), # 0.001,0.003,, 
    subsample=0.45, #sample(c(0.3, 0.45, 0.5, 0.6),1), 
    colsample_bytree=sample(c(0.05, 0.1, 0.2),1), # , 0.3, 0.4, 0.5, 0.8
    nparallel = 2,
    # parGrad=sample(c(10,15,20,25,30),1),
    # parHessNegProb=sample(c(0.3,0.5,0.7),1),
    # parHessShift= sample(c(-4, -3, -2, -1, 0),1),
    # parHessShiftSd=sample(c(1,2,3),1),
    algo=sample(c(2,3),1)
  )
  ctrl_parameters<-data.frame(
    nrounds=100,
    eval_metric=ifelse(parameters$algo<=2, "merror", "mlogloss")
  )
  
  print (parameters)

  for (fn in names(folds)){
    print(sprintf("Fold: %s", fn))
    f<-folds[[fn]]
    ensResult<-list()
    for (p in 1:parameters$nparallel){
      #print(sprintf("Loop repeat: %s", p))
      sub_seed <- sample(1:2^15, 1)
      #print(sprintf("Seed for session: %s", sub_seed))
      set.seed(sub_seed)
      
      xgbparamsFactor<-list(gamma=parameters$gamma, 
                            max_depth = parameters$max_depth, 
                            eta = parameters$eta, 
                            subsample=parameters$subsample, 
                            colsample_bytree=parameters$colsample_bytree)
      
      
      
      allScores<-fold_models[[fn]][["allScores"]]
      allScores<-inner_join(allScores, train_num, by=c("team", "oppTeam", "round", "gameindex", "season"))

      priorProbs<-table(allScores$GS[!allScores$oob], allScores$GC[!allScores$oob])
      priorProbs<-priorProbs[as.character(0:4), as.character(0:4)] # cut away 
      priorProbs<-priorProbs/sum(priorProbs)
      gLevPriorProbs<-sapply(gLev, function(x) priorProbs[substr(as.character(x),1,1),substr(as.character(x),3,3)])
      names(gLevPriorProbs)<-gLev

      allScores$weights<-1
      trainData<-allScores[!allScores$oob,]
      trainScores<-data.frame()
      
      trainScores<-trainData
      trainScores$weights<-calcPoints(trainData$GS, trainData$GC, trainData$GS, trainData$GC, home = TRUE)
      for (gs in 0:4){
        for (gc in 0:4){
          trainData$weights<-calcPoints(rep(gs, times=nrow(trainData)), rep(gc, times=nrow(trainData)), trainData$GS, trainData$GC, home = TRUE)
          newTrainData<-trainData[trainData$weights>0,]
          newTrainData$weights<-newTrainData$weights # +20*gLevPriorProbs[paste(gs, gc, sep=":")] # weight = reward + priorprob*20
          if (nrow(newTrainData)>0){
            newTrainData$GS<-gs
            newTrainData$GC<-gc
          }
          trainScores<-rbind(trainScores, newTrainData)
        }
      }
      allScores<-rbind(trainScores, allScores[allScores$oob,])  
      labels<-data.frame(
        GS=as.integer(substr(as.character(allScores$score),1,1)),
        GC=as.integer(substr(as.character(allScores$score),3,3)))
      allgamesFillNA<-allScores %>% select(-gameindex, -score, -oob, -GS, -GC, -weights)        # , -season, -round 
      allgamesFillNA[is.na(allgamesFillNA <- allgamesFillNA)] <- -999

      
      xgbMatrix<-sparse.model.matrix(~.-1, data = allgamesFillNA)

      trainMatrix<-xgb.DMatrix(xgbMatrix[!allScores$oob,], label=as.integer(allScores$score[!allScores$oob])-1, missing = -999, weight=allScores$weights[!allScores$oob])
      testMatrix <-xgb.DMatrix(xgbMatrix[allScores$oob,], label=as.integer(allScores$score[allScores$oob])-1, missing = -999)
      watchlist<-list(train=trainMatrix, test=testMatrix)

      if (parameters$algo>1){
        
        model3<-xgb.train(nrounds =1,  #early_stopping_rounds = 50, 
                          params = xgbparamsFactor, 
                          data = trainMatrix , 
                          objective="multi:softmax", eval_metric=ctrl_parameters$eval_metric, maximize=FALSE, 
                          print_every_n=1, verbose=1, 
                          nthread = 3,  num_class=nlevels(gLev) 
                          , watchlist = watchlist
        )
        evaluation_log<-data.frame()
        wpoint_log<-data.frame()
        # setinfo(trainMatrix, "base_margin", runif(n=nlevels(gLev)*nrow(trainMatrix), min=-1, max=1))
        # setinfo(testMatrix, "base_margin", runif(n=nlevels(gLev)*nrow(testMatrix), min=-1, max=1))
        last5<-function(x,n=5){stats::filter(x,rep(1/n,n), sides=1)}

        ylabels<-getinfo(trainMatrix, "label")+1
        for (j in 1:ctrl_parameters$nrounds){
          ptrain <- predict(model3, trainMatrix, outputmargin=TRUE)
          ptest  <- predict(model3, testMatrix, outputmargin=TRUE)
          setinfo(trainMatrix, "base_margin", ptrain) #+ptrain*rnorm(n=length(ptrain), mean = 0.01, sd = 0.01)) # disturb train weights with random noise
          setinfo(testMatrix, "base_margin", ptest)
          
          model3<-xgb.train(nrounds =1, 
                            params = xgbparamsFactor,#, min_child_weight=0,
                            data = trainMatrix , 
                            objective="multi:softmax", eval_metric=ctrl_parameters$eval_metric, maximize=FALSE, 
                            print_every_n=1, verbose=1, 
                            nthread = 3,  num_class=nlevels(gLev) 
                            , watchlist = watchlist
          )
          wpoints<-data.frame(
            train_wpoints=mean(calcWeightedPoints(model3, trainMatrix)),
            test_wpoints=mean(calcWeightedPoints(model3, testMatrix)),
            train_maxp=mean(calcSoftMaxProb(preds = ptrain)),
            test_maxp=mean(calcSoftMaxProb(preds = ptest)))
          wpoint_log<-rbind(wpoint_log, wpoints)
          scores<-calcScores(allScores, model3)
          evaluation_log<-rbind(evaluation_log, cbind(model3$evaluation_log, scores))
          w1<-evaluation_log$train_avgpoints / wpoint_log$train_wpoints
          
          if (j %% 5 == 0){
            #printSoftMax(model3, testMatrix, trainMatrix)
            print(cbind(wpoints, scores))
            #print(w1[(j-4):j])
            #with(wpoint_log, print((train_wpoints-lag(train_wpoints,5))[(j-4):j]))
          }
          #xgb.plot.deepness(model = model3)
          #xgb.dump(model=model3, fname="model3.save.txt", with_stats = TRUE, dump_format = "text")
          if (j >= 10 & scores$train_avgpoints>2.2) {
            ensResult<-c(ensResult, list(evaluateXgbFactorSimple(model3, testMatrix, actualGoals[allScores$oob,], "test ensemble XGB objective function")))   
          }
        }
      }
      else
      {
        evaluation_log<-NULL
        wpoint_log<-NULL
        model3<-xgb.train(nrounds =100,  #early_stopping_rounds = 10, 
                          params = xgbparamsFactor, 
                          data = trainMatrix , 
                          objective="multi:softmax", eval_metric=ctrl_parameters$eval_metric, #c("merror"), #"mlogloss", #feval=evaltarget, maximize=TRUE, 
                          print_every_n=1, verbose=1, 
                          nthread = 10,  num_class=nlevels(gLev) 
                          , watchlist = watchlist
        )
      }
      
      #plotImportance(model3, n=50, cols=colnames(trainMatrix))
      #varImp<-printImportance(model3, n=ncol(trainMatrix), cols=colnames(trainMatrix))
      #xgb.plot.deepness(model = model3)
      #xgb.dump(model=model3, fname="model3.save.txt", with_stats = TRUE, dump_format = "text")
      #xgb.plot.tree(feature_names = colnames(xgbMatrix), model = model3, n_first_tree = 24)
      #xgb.plot.multi.trees(feature_names = colnames(xgbMatrix), model = model3)
      
      scores<-calcScores(allScores, model3)
      actualGoals<-allScores %>% select(GS, GC)
      trainResult<-evaluateXgbFactorSimple(model3, trainMatrix, actualGoals[!allScores$oob,], "train XGB objective function")
      trainGS<-trainResult$gspred
      trainGC<-trainResult$gcpred

      testResult<-evaluateXgbFactorSimple(model3, testMatrix, actualGoals[allScores$oob,], "test XGB objective function")
      testGS<-testResult$gspred
      testGC<-testResult$gcpred

      ensResult<-c(ensResult, list(data.frame(gspred=testGS, gcpred=testGC)))
      #table(paste(testGS, testGC), paste(actualGoals$GS[allScores$oob], actualGoals$GC[allScores$oob], sep=":"))
      #result_rf_log<-data.frame()
      result_rf_log<-read.csv(file = "xgb\\result_xgb_log_3.csv")
      result<-data.frame(time=date(), parameters, sub_seed=sub_seed, fold=fn, k=p,
                         trainScore=scores$train_avgpoints,
                         testScore=scores$test_avgpoints)
                         #best_iter=model3$best_iteration,
                         #best_ntreelimit=model3$best_ntreelimit)
      result_rf_log<-rbind(result_rf_log, result)
      write.csv(result_rf_log, file = "xgb\\result_xgb_log_3.csv", row.names = FALSE)
      
      pdf(paste0("xgb\\plot", sub_seed, ".pdf"))
      summarizePredictionSimple(actualGoals[!allScores$oob,], trainGS, trainGC, label=paste("Train", paste(names(result), result, sep="=", collapse = ", ")), verbose=0)
      summarizePredictionSimple(actualGoals[allScores$oob,], testGS, testGC, label=paste("Test", paste(names(result), result, sep="=", collapse = ", ")), verbose=0)
      printLearningCurve(model3, maximise = FALSE, cutoff=10, parameters[,-(2:6)], data = evaluation_log)
      printSoftMax(model3, testMatrix, trainMatrix)
      printBoostingScores(wpoint_log, evaluation_log)
      dev.off()
      sink(paste0("xgb\\cvdata", sub_seed, ".log"))
      cat("seed=", sub_seed, "\n")
      print(result)
      printBoostingScores(wpoint_log, evaluation_log)
      printSoftMax(model3, testMatrix, trainMatrix)
      par(mfrow=c(3,1))
      summarizePredictionSimple(actualGoals[allScores$oob,], testGS, testGC, label=paste("Test", paste(names(result), result, sep="=", collapse = ", ")), verbose=1)
      summarizePredictionSimple(actualGoals[!allScores$oob,], trainGS, trainGC, label=paste("Train", paste(names(result), result, sep="=", collapse = ", ")), verbose=1)
      printLearningCurve(model3, maximise = FALSE, cutoff=10, parameters[,-(2:6)], data = evaluation_log)
      sink()
      par(mfrow=c(1,1))
      print(result)
      
    }    
    mostFreqElement<-function(z) { apply(z, 1, function(x) as.integer(names(sort(table(x), decreasing = TRUE))[1])) }
    
    ensGS<-sapply(ensResult, function(x) x$gspred)
    ensGC<-sapply(ensResult, function(x) x$gcpred)
    
    ensTendency<-sign(ensGS-ensGC)
    ensTendencyVote<-mostFreqElement(ensTendency)
    for (ii in 1:length(ensResult)){
      w <-which(ensTendency[,ii] != ensTendencyVote)
      deleteIndex<-matrix(c(w,rep(ii, times=length(w))), ncol=2)
      ensGS[deleteIndex]<-NA
      ensGC[deleteIndex]<-NA
    }
    ensGDiff<-ensGS-ensGC
    ensGDiffVote<-mostFreqElement(ensGDiff)
    for (ii in 1:length(ensResult)){
      w <-which(ensGDiff[,ii] != ensGDiffVote)
      deleteIndex<-matrix(c(w,rep(ii, times=length(w))), ncol=2)
      ensGS[deleteIndex]<-NA
      ensGC[deleteIndex]<-NA
    }
    ensGoals<-ensGS
    ensGoalsVote<-mostFreqElement(ensGoals)
    for (ii in 1:length(ensResult)){
      w <-which(ensGoals[,ii] != ensGoalsVote)
      deleteIndex<-matrix(c(w,rep(ii, times=length(w))), ncol=2)
      ensGS[deleteIndex]<-NA
      ensGC[deleteIndex]<-NA
    }
    ensPredGS<-apply(ensGS, 1, median, na.rm=TRUE)
    ensPredGC<-apply(ensGC, 1, median, na.rm=TRUE)
    ensTestScore<-mean(calcPoints(actualGoals$GS[allScores$oob], actualGoals$GC[allScores$oob], ensPredGS, ensPredGC, home=TRUE))

    ensRes<-data.frame(time=date(), parameters, fold=fn, testScore=ensTestScore)
    summarizePredictionSimple(actualGoals[allScores$oob,], ensPredGS, ensPredGC, label=paste("Ensemble Test", paste(names(ensRes), ensRes, sep="=", collapse = ", ")), verbose=0)
    
    result_rf_log<-read.csv(file = "xgb\\result_xgb_log_3.csv")
    result<-data.frame(time=date(), parameters, sub_seed=sub_seed, fold=fn, k=0,
                       trainScore=NA,
                       testScore=ensTestScore)
    result_rf_log<-rbind(result_rf_log, result)
    write.csv(result_rf_log, file = "xgb\\result_xgb_log_3.csv", row.names = FALSE)
  }
}

result_log<-result_rf_log %>% filter(algo==3) 

result_log<-result_rf_log %>% filter(testScore >= 1.4  & trainScore > 1.6 & algo>=8) %>% filter(algo==15)
result_log<-result_rf_log  %>% filter(algo==7 & testScore >= 1.05 & trainScore > 1.6)
result_log<-result_rf_log  %>% filter(algo==7 & subsample==0.6 & colsample_bytree==0.4)
result_log<-result_rf_log  %>% filter((algo==8 | algo==9) & testScore >= 1.05 & trainScore > 1.6)
result_log<-result_rf_log %>% filter(testScore >= 1.4  & trainScore > 1.6 & algo==12)
result_log<-result_rf_log %>% filter((colsample_bytree>=0.2 | subsample>=0.2) & algo==12)
result_log<-result_rf_log %>% filter(testScore >= 1.4  & trainScore > 1.6 & algo==16) %>%
  mutate(algo=ifelse(row_number()>=717,17,16))

result_log <- result_log %>% 
  filter(eta<=0.03 & gamma==1 & subsample<=0.6)

result_log<-result_rf_log %>% filter(trainScore > 2 & trainScore <= 4 & algo>=8) %>% filter(algo %in% c(8,19))
result_log <- result_log %>% 
  filter(eta==0.001)
#& colsample_bytree==0.4)
#& gamma==1 & subsample==0.45)

result_log <- tail(result_log, 120)
result_log <- tail(result_rf_log, 48)

boxplot(testScore~algo, data=result_log)
boxplot(testScore~algo+fold, data=result_log)
boxplot(trainScore~algo, data=result_log)
boxplot(trainScore~algo+fold, data=result_log)
aggregate(testScore~fold+algo, data=result_log, FUN=mean)

aggregate(testScore~fold+nparallel, data=result_log, FUN=mean)

plot(testScore~trainScore, data=result_log %>% filter(algo==1))

with (result_log %>% filter(algo<=3), 
plot(y=testScore, x=trainScore, col=fold, pch=algo)
)
plot(y=result_log$testScore, x=result_log$trainScore, col=result_log$fold, pch=result_log$algo)


smoothScatter((result_log )$trainScore, result_log$testScore)
abline(lm(testScore~trainScore, data=result_log ))

result_log_f<-result_log %>% filter(fold=="2014_15")
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2015_16")
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2016_17")
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))

result_log_f<-result_log %>% filter(fold=="2014_15" & algo==16)
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2015_16" & algo==16)
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2016_17" & algo==16)
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2014_15" & algo==17)
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2015_16" & algo==17)
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))
result_log_f<-result_log %>% filter(fold=="2016_17" & algo==17)
smoothScatter((result_log_f )$trainScore, result_log_f$testScore)
abline(lm(testScore~trainScore, data=result_log_f ))


smoothScatter((result_log %>% filter(algo==1))$trainScore, result_log$testScore)
abline(lm(testScore~trainScore, data=result_log ))

plot(testScore~trainScore, data=result_log )
abline(lm(testScore~trainScore, data=result_log ))

plot(lm(testScore~trainScore, data=result_log))

boxplot(testScore~fold, data=result_log )
boxplot(testScore~algo+fold, data=result_log )
boxplot(testScore~max_depth, data=result_log )
boxplot(testScore~max_depth+fold, data=result_log)
boxplot(testScore~gamma, data=result_log)
boxplot(testScore~gamma+fold, data=result_log)
boxplot(testScore~max_depth*gamma, data=result_log)
boxplot(testScore~subsample, data=result_log)
boxplot(testScore~subsample+fold, data=result_log)
boxplot(testScore~colsample_bytree, data=result_log)
boxplot(testScore~colsample_bytree*fold, data=result_log)
boxplot(testScore~(eta), data=result_log)
boxplot(testScore~(eta*fold), data=result_log)

scoreModel<-lm(testScore ~ fold*(max_depth+gamma+subsample+colsample_bytree+eta), data=result_log)
scoreModel<-lm(testScore ~ fold+(max_depth+gamma+subsample+colsample_bytree+eta), data=result_log)
summary(scoreModel)
sort(coefficients(scoreModel))
anova(scoreModel)

scoreModel<-lm(testScore ~ fold+(max_depth+gamma+colsample_bytree+eta), data=result_log %>% filter(subsample==0.5))
scoreModel<-lm(testScore ~ fold*(gamma+colsample_bytree+eta), data=result_log %>% filter(subsample==0.5 & max_depth==5))


result_log %>% filter(algo>=15 ) %>%
  group_by(fold, max_depth) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-2*sd) %>%
  arrange(meansd) %>%
  print.data.frame()

result_log %>% filter(algo>=15 ) %>%
  group_by(seed, algo, max_depth, gamma, eta, subsample, colsample_bytree) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-sd) %>%
  arrange(-meansd) %>%
  print.data.frame()

best_seed<- result_rf_log %>% filter(seed %in% c(22564,9450) )

best_seed %>% group_by(seed, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-sd) %>%
  arrange(meansd) 
  

result_log%>%  group_by(fold, algo) %>%
  summarize(train=mean(trainScore), trainsd=sd(trainScore), trainmed=median(trainScore),
            test=mean(testScore), testsd=sd(testScore), testmed=median(testScore),
            n=n(), q=quantile(testScore, 0.1)) %>% 
  arrange(test) %>%
  print.data.frame()

  q1<-((result_log%>% filter(algo==2) %>% group_by(fold) %>% summarize(q=quantile(testScore, 0.1))%>% ungroup())$q[1])
  q2<-((result_log%>% filter(algo==2) %>% group_by(fold) %>% summarize(q=quantile(testScore, 0.1))%>% ungroup())$q[2])
  q3<-((result_log%>% filter(algo==2) %>% group_by(fold) %>% summarize(q=quantile(testScore, 0.1))%>% ungroup())$q[3])

  abline(h=q1)
  abline(h=q2)
  abline(h=q3)

lowscores<-result_log %>% filter(fold=="2014_15" & testScore < q1 |
                                   fold=="2015_16" & testScore < q2 |
                                   fold=="2016_17" & testScore < q3 )

plot(testScore~fold, data=lowscores)

lowscores[,3:14] %>% mutate(shift=(parHessShift+parHessShiftSd)*parHessNegProb) %>% arrange(trainScore) %>% print.data.frame()
lowscores[,3:14] %>% mutate(shift=(parHessShift+parHessShiftSd)*parHessNegProb) %>% select(shift, testScore) %>% plot()

with(result_log[,3:14] %>% mutate(shift=(parHessShift+parHessShiftSd)*1) %>% select(shift, testScore, fold),
     boxplot(testScore ~ shift*fold))

boxplot(testScore~(parHessNegProb*fold), data=result_log%>% filter(algo==2 & parHessShift+parHessShiftSd>=0))

boxplot(testScore~(parHessNegProb*fold), data=result_log%>% filter(algo==2 & parHessShift+parHessShiftSd>=0))

result_log %>% filter(algo==2 & parHessShift+parHessShiftSd>=0) %>%
  group_by(parHessNegProb, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-2*sd) %>%
  arrange(meansd, fold, parHessNegProb) %>%
  print.data.frame()

boxplot(testScore~(max_depth* colsample_bytree*fold), data=result_log%>% filter(algo==2 & parHessShift+parHessShiftSd>=0 & parHessNegProb==0.5))

result_log %>% filter(algo==2 & parHessShift+parHessShiftSd>=0 & parHessNegProb==0.5) %>%
  group_by(max_depth, colsample_bytree,fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-2*sd) %>%
  arrange(meansd, fold, max_depth) %>%
  print.data.frame()


result_log %>% filter(algo==2 & parHessShift+parHessShiftSd>=0 & parHessNegProb==0.5) %>%
  group_by(max_depth, colsample_bytree, eta, subsample, parGrad, gamma) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-sd) %>%
  arrange(meansd) %>%
  print.data.frame()

result_log %>% filter(algo==2 & parHessShift+parHessShiftSd>=0 & parHessNegProb==0.5 & 
                        max_depth==2 & colsample_bytree==0.2 & eta==0.1 & subsample==0.3 & parGrad==30 & gamma==1) %>%
  group_by(fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-sd) %>%
  arrange(meansd) %>%
  print.data.frame()

result_log %>% filter(algo==2 & #parHessShift+parHessShiftSd>=0 & parHessNegProb==0.5 & parGrad==30 &  & gamma==0 & colsample_bytree==0.1 & eta==0.001 & subsample==0.1
                        max_depth==5 ) %>%
  group_by(fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-sd) %>%
  arrange(meansd) %>%
  print.data.frame()


  str((result_log%>% filter(algo==2) %>% group_by(fold) %>%
    summarize(q=quantile(testScore, 0.1)) %>% ungroup())$q)

plot(testScore~(mtry), data=result_log)
plot(testScore~(ntree), data=result_log)
plot(testScore~(minsplit), data=result_log)
summary(result_log)

plot(testScore~trainScore, data=result_log)
plot(testScore~trainScore, data=result_log)

result_log %>% filter(algo==2& ntree==2000) %>%
  group_by(mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, mtry)

result_log %>% filter(algo==2 ) %>%
  group_by(ntree, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, ntree)


result_log %>% filter(algo==2& ntree==2000 & mtry==2) %>%
  group_by(minsplit, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, minsplit)

result_log %>% filter(algo==2 & mtry==2 & minsplit==15 & ntree==2000) %>%
  group_by(minsplit, ntree, mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, ntree, minsplit, mtry) %>%
  print.data.frame()


result_log %>% filter(algo==2) %>%
  group_by(minsplit, ntree, mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-2*sd) %>%
  arrange(meansd, fold, ntree, minsplit, mtry) %>%
  print.data.frame()

result_log %>% filter(algo==2 & ntree==2000 & mtry==2 & minsplit==15) %>%
  group_by(minsplit, ntree, mtry) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-sd) %>%
  arrange(meansd, ntree, minsplit, mtry) %>%
  print.data.frame()

xy %>% filter(where=="Home") %>% transmute(t = sign(GS - GC), season=season) %>% 
  group_by(season, t) %>% 
  summarize(records=n()) %>%
  group_by(season) %>% 
  mutate(games=sum(n())) %>%
  ungroup() %>%   
  mutate(ratio=records/games)




#xy<-xy %>% select(-starts_with("pois"))

# train<-xy %>% filter(season %in% c("2016_17","2015_16") ) 
# test<-xy%>% filter(season == "2014_15")
# oobidx<-xy$season==as.character(unique(test$season))
# table(oobidx)  

# trainY<-train %>% select(GS, GC)
# testY<-test %>% select(GS, GC)

xyFillNA<-xy
xyFillNA[is.na(xyFillNA <- xy)] <- -999

xyMatrix <- sparse.model.matrix(~.-1, data = xyFillNA)
trainXM <- xyMatrix[,-2:-1]
trainYM <- xyMatrix[,1:2]

trainLabels<-(as.data.frame(as.matrix(trainYM)) %>% limitMaxScore(4) %>% buildFactorScore())$score

trainMatrix<-xgb.DMatrix(trainXM, label=as.integer(trainLabels)-1, missing = -999)
attr(trainMatrix, "where")<-(as.data.frame(as.matrix(trainXM)) %>% select(whereHome))$where


#watchlist<-list(train=trainMatrix, test=testMatrix)

#xgbparamsFactor<-list(gamma=5, max_depth = 4, eta = 0.01, subsample=0.5, colsample_bytree=0.05, nthread = 6, num_class=nlevels(trainLabels))

#special<-345
for (i in 1:10) {
  resultlog<-read.csv(file = "resultlog.csv")
  eff_seed <- sample(1:2^15, 1)
  #eff_seed<-ifelse(special==345 & i==1, 21735, eff_seed)
  print(sprintf("Seed for session: %s", eff_seed))
  set.seed(eff_seed)
  
  xgbparamsFactor<-list(gamma=sample(c(3,5,10,15,20),1), 
                        max_depth = sample(c(2,3,4,5,7,10,20),1), 
                        eta = sample(c(0.1,0.01,0.001,0.003,0.03, 0.0003),1), 
                        subsample=sample(c(0.45,0.5),1), 
                        colsample_bytree=sample(c(0.05, 0.1, 0.2, 0.15),1),
                        num_parallel_tree=sample(c(1,2,3,5),1))
  print(as.data.frame(xgbparamsFactor))
  nrounds<-ifelse(xgbparamsFactor[["num_parallel_tree"]]>5, 40, 200)
  early_stopping_rounds<-ifelse(xgbparamsFactor[["num_parallel_tree"]]>5, 10, 50)
  
  modelcv<-xgb.cv(nrounds =nrounds, early_stopping_rounds = early_stopping_rounds, 
                  params = xgbparamsFactor, data = trainMatrix , 
                  obj=objective, feval=evaltarget, maximize=TRUE, 
                  print_every_n=1, verbose=1, prediction=FALSE,
                  nthread = 10, num_class=nlevels(trainLabels),
                  folds = rep(split(seq_along(xy$season), xy$season), 1))
  
  printLearningCurveCV(modelcv, maximise = TRUE, cutoff = 10, info=list(seed=eff_seed))
  pdf(paste0("plot", eff_seed, ".pdf"))
  printLearningCurveCV(modelcv, maximise = TRUE, cutoff = 10, info=list(seed=eff_seed))
  print(modelcv, verbose=TRUE)
  dev.off()
  
  sink(paste0("cvdata", eff_seed, ".log"))
  cat("seed=", eff_seed, "\n")
  print(modelcv, verbose=TRUE)
  print.data.frame(modelcv$evaluation_log)
  sink()
  
  resultlog<-rbind(resultlog,
                   data.frame(time=date(), seed=eff_seed, best_iter=modelcv$best_iteration, 
                              best_score=modelcv$evaluation_log$test_avgpoints_mean[modelcv$best_iteration],
                              best_score_stddev=modelcv$evaluation_log$test_avgpoints_std[modelcv$best_iteration],
                              modelcv$params))
  write.csv(resultlog, file = "resultlog.csv", row.names = FALSE)
}

resultlog<-resultlog[resultlog$best_score<1.7,]

print.data.frame(resultlog[order(resultlog$best_score, decreasing = TRUE),3:12])

plot(best_score ~ gamma, data=resultlog[resultlog$best_score>1.6,])
plot(best_score ~ log(eta), data=resultlog[resultlog$best_score>1.6,])
plot(best_score ~ max_depth, data=resultlog[resultlog$best_score>1.6,])
plot(best_score ~ subsample, data=resultlog[resultlog$best_score>1.6,])
plot(best_score ~ log(colsample_bytree), data=resultlog[resultlog$best_score>1.6,])
plot(best_score ~ log(num_parallel_tree), data=resultlog[resultlog$best_score>1.6,])


resultlog<-rbind(resultlog, data.frame(time=date(), seed=eff_seed, best_iter=12,
           best_score=1.866013,
           best_score_stddev=0.060680, xgbparamsFactor,
           nthread=10, num_class=25, silent=1))


modeleval<-xgb.train(nrounds =100, watchlist=watchlist, verbose=1, obj=objective, params = xgbparamsFactor, data = trainMatrix , feval=evaltarget, maximize=TRUE, print_every_n=1)
printLearningCurve(modeleval, maximise = TRUE)
result<-evaluateXgbFactor(modeleval, testXM, testYM, "test XGB objective function")
result<-evaluateXgbFactor(modeleval, trainXM, trainYM, "train XGB objective function")
plotImportance(modeleval, 40)
printImportance(modeleval)






