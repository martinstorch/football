setwd("~/LearningR/Bundesliga")

#install.packages("dplyr")
#install.packages("party")
##install.packages("rattle",  dependencies=c("Depends", "Suggests"))

#require(devtools)
#install.packages('xgboost')

current_date<-format(Sys.time(), "%Y%m%d")
newpredfilename<-paste0("newpredictions_", current_date, ".csv")


inputFeatures<-c("BLfeatures_2.csv", "BLfeatures_2013.csv")
newSeason<-"BLfeatures_2017.csv"
intermediateOutput<-"pred_201718_rf_features"

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

predict_test_from_model1 <- function(model1, newdata) {
  train_pred1<-predict(model1, newdata=newdata)
  train_pred1<-matrix(data=unlist(train_pred1), ncol=2, byrow=TRUE, dimnames = list(NULL, c("pGS","pGC")))
  home<-cbind(newdata, train_pred1) %>% filter(where=="Home") %>% 
    select(pGS, pGC, team, oppTeam, round, gameindex, season) %>% mutate(pGDiff=pGS-pGC)
  away<-cbind(newdata, train_pred1) %>% filter(where=="Away") %>% 
    select(pGS=pGC, pGC=pGS, team=oppTeam, oppTeam=team, round, gameindex, season) %>% mutate(pGDiff=pGS-pGC)
  games<-inner_join(home, away, by=c("team", "oppTeam", "round", "gameindex", "season")) 
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
  #p7<-(2*p3+p6)
  predest<-matrix(p7, ncol=25, byrow = TRUE)
  colnames(predest)<-gLev
  rownames(predest)<-gLev
  predest<-predest/rowSums(predest)
}


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

calcWeightedPoints<-function(model, newdata){
  pred_params<-predict(object=model, newdata = newdata, reshape = TRUE, outputmargin = TRUE)
  colnames(pred_params)<-gLev
  pred_prob<-exp(pred_params)
  pred_prob<-pred_prob/rowSums(pred_prob)
  
  ylabels<-getinfo(newdata, "label")+1
  wpoints<-rowSums(pred_prob * maskm$home[ylabels,])
  return(wpoints)
}

calcSoftMaxProb <- function(preds) {
  m<-matrix(preds, ncol=nlevels(gLev))
  colnames(m)<-gLev
  m<-exp(m)
  probs<-m/rowSums(m)
  maxProbs<-apply(probs, 1, max)
  return(maxProbs)
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

printLearningCurve<-function(model, maximise=FALSE, cutoff=1, info=list()){
  train_means<-model$evaluation_log[,2][[1]]
  test_means<-model$evaluation_log[,3][[1]]
  if (maximise==TRUE) {
    best_mean<-max(test_means)
    best_iter<-which.max(test_means)
  } else {
    best_mean<-min(test_means)
    best_iter<-which.min(test_means)
  }
  l<-length(train_means)
  ylim<-range(c(train_means[cutoff:l], test_means[cutoff:l]))
  plot(train_means, lwd=1, type="l", ylim=ylim, 
       main=paste(best_mean, best_iter, sep=" / "), 
       #sub=printtext, 
       xlab="Iteration", ylab="Score")
  points(test_means, lwd=1, type="l", col="blue")
  printdata<-as.data.frame(c(model$params, best_iter=best_iter, info))
  printtexts<-paste(names(printdata), printdata, sep="=")
  abline(v=best_iter, col="red")
  abline(h=best_mean, col="red", lty="dashed")
  #legend("bottomleft", legend = paste(names(printdata), printdata, sep="="), cex=0.7, bty = "n")
  mtext(paste(printtexts[1:6], collapse = ", "), cex=0.7, line = 0, outer = FALSE)
  mtext(paste(printtexts[7:length(printtexts)], collapse = ", "), cex=0.7, line = -1, outer = FALSE)
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

printSoftMaxGame <- function(model3, testMatrix, trainMatrix, gameindex) {
  pred_test_pred<-predict(object=model3, newdata = testMatrix, reshape = TRUE, outputmargin = TRUE)
  pred_train_pred<-predict(object=model3, newdata = trainMatrix, reshape = TRUE, outputmargin = TRUE)
  colnames(pred_test_pred)<-gLev
  colnames(pred_train_pred)<-gLev
  pred_test_prob<-exp(pred_test_pred)
  pred_test_prob<-pred_test_prob/rowSums(pred_test_prob)
  pred_train_prob<-exp(pred_train_pred)
  pred_train_prob<-pred_train_prob/rowSums(pred_train_prob)
  par(mfrow=c(2,2))
  plot(value~Var2, data=melt(pred_test_pred[c(gameindex,gameindex),]), main="Test Outputs")
  plot(value~Var2, data=melt(pred_test_prob[c(gameindex,gameindex),]), main="Test Probs", xlab=NA)
  abline(h=0.04, col="red")
  plot(value~Var2, data=melt(pred_train_pred), main="Train Outputs", xlab=NA, ylim=range(pred_train_pred))
  plot(value~Var2, data=melt(pred_train_prob), main="Train Probs", xlab=NA)
  abline(h=0.04, col="red")
  par(mfrow=c(1,1))
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
    print(scoreTable)
    print(colSums(scoreTable)/sum(scoreTable)*100)
    print(scoreTable/rowSums(scoreTable)*100)
    print(label)
    print(summaryTable %>% data.frame)
  }
  return(summaryTable)
}  


################################################################################
# data preparation for learning


mask<-buildMask()
gLev<-sort(buildFactorScore(data.frame(expand.grid(GS=0:4, GC=0:4)))$score)
predest<-prediction_estimate()
maskm<-buildMaskMatrix()

xy<-data.frame()
for (f in inputFeatures){
  xyf<-read.csv(file=f, stringsAsFactors = TRUE, as.is = FALSE)
  xy<-bind_rows(xy, xyf)
}
test_num<-read.csv(file=newSeason, stringsAsFactors = TRUE, as.is = FALSE)
lastround<-max(test_num$round)
print(lastround)
test_num$GS[test_num$round==lastround]<-NA
test_num$GC[test_num$round==lastround]<-NA
xy<-bind_rows(xy, test_num[!is.na(test_num$GS),])

#xy %>% select(season, round, GS, GC) %>% tail(100)

test_num <- test_num %>% filter(round==lastround) %>% select(-GS, -GC)

teams<-levels(as.factor(c(as.character(xy$team), as.character(test_num$team))))
seasons<-levels(as.factor(c(as.character(xy$season), as.character(test_num$season))))
dows<-levels(as.factor(c(as.character(xy$dow), as.character(test_num$dow))))

xy$team<-factor(xy$team, levels=teams)
xy$oppTeam<-factor(xy$oppTeam, levels=teams)
xy$season<-factor(xy$season, levels=seasons)
xy$dow<-factor(xy$dow, levels=dows)
test_num$team<-factor(test_num$team, levels=teams)
test_num$oppTeam<-factor(test_num$oppTeam, levels=teams)
test_num$season<-factor(test_num$season, levels=seasons)
test_num$dow<-factor(test_num$dow, levels=dows)


train_num<-xy
test_num<-bind_rows(train_num, test_num)[(nrow(train_num)+1):(nrow(train_num)+nrow(test_num)),] # adjust for missing columns

print(test_num[test_num$round==lastround,1:10])
print(sort(levels(xy$team)))

#################################################################################################################################

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

sidehome<-train_num %>% filter(where=="Home") %>% select(-where) %>%
  select(team, oppTeam, round, gameindex, season)
rf_predictions<-sidehome

testmatches<-test_num %>% filter(where=="Home") %>% select(-where) %>%
  select(team, oppTeam, round, gameindex, season)
testmatchpredictions<-testmatches

for (p in 1:parameters$nparallel){
  seed <- sample(1:2^15, 1)
  set.seed(seed)
  print(seed)
  columns<-sample(13:ncol(train_num), size = 100, replace = FALSE)
  train_data_rf<-train_num %>%
    select(GS, GC, team, oppTeam, round, gameindex, season, contains("promoted"), columns)
  new_data_rf<-train_num %>%
    select(where, GS, GC, team, oppTeam, round, gameindex, season, contains("promoted"), columns)
  
  model1 <- cforest(GS*GC ~ ., data=train_data_rf, controls = cforest_ctrl)
  model1_prediction <- predict_from_model1(model1, newdata=new_data_rf)
  colnames(model1_prediction)<-gsub("\\.x", paste0(".x",p), colnames(model1_prediction))
  colnames(model1_prediction)<-gsub("\\.y", paste0(".y",p), colnames(model1_prediction))
  rf_predictions<-inner_join(rf_predictions, model1_prediction %>% select(-GS, -GC), by=c("team", "oppTeam", "round", "gameindex", "season"))

  model1_test_prediction <- predict_test_from_model1(model1, newdata=test_num)
  colnames(model1_test_prediction)<-gsub("\\.x", paste0(".x",p), colnames(model1_test_prediction))
  colnames(model1_test_prediction)<-gsub("\\.y", paste0(".y",p), colnames(model1_test_prediction))
  testmatchpredictions<-inner_join(testmatchpredictions, model1_test_prediction, by=c("team", "oppTeam", "round", "gameindex", "season"))

  model1<-NULL
  gc()
}

allScores<-
  inner_join(rf_predictions, 
             train_num %>% filter(where=="Home") %>% select(GS, GC, gameindex, round, season, team, oppTeam), 
             by=c("team", "oppTeam", "round", "gameindex", "season")
  ) %>%
  limitMaxScore(4) %>% buildFactorScore() 

write.csv(allScores, file = paste0(intermediateOutput, ".csv"), row.names = FALSE)  
saveRDS(allScores, file = paste0(intermediateOutput, ".rds"))  

write.csv(testmatchpredictions, file = paste0("test_", intermediateOutput, ".csv"), row.names = FALSE)  
saveRDS(testmatchpredictions, file = paste0("test_", intermediateOutput, ".rds"))  

##################

new_data<-testmatchpredictions
new_data %>% select(team, oppTeam, round, starts_with("pG"), season, gameindex)

rfpred<-melt(new_data %>% select(gameindex, team, oppTeam, round, starts_with("pG"), season, -starts_with("pGDiff")))
rfpreddata<-data.frame()
for (z in c(paste0("x",1:10), paste0("y",1:10))){
  rfpreddata<-rbind(rfpreddata,
                    inner_join(
                      rfpred %>% filter(variable==paste0("pGS.",z)) %>% select(team, oppTeam, GS=value),
                      rfpred %>% filter(variable==paste0("pGC.",z)) %>% select(team, oppTeam, GC=value),
                      by = c("team", "oppTeam")
                    ) %>% mutate(diff=GS-GC, variable=z, teamid=as.integer(team))
  )
}
rfpreddata<-inner_join(rfpreddata, testmatchpredictions %>% select(gameindex, team, oppTeam), by = c("team", "oppTeam"))
rfpreddata %>% arrange(team, variable)

with(rfpreddata, {
  plot(GC~GS, col=gameindex, pch=gameindex) #, xlim=c(0,2), ylim=c(0,2))
  abline(a = 0, b=1)
  abline(a = mean(GS)+mean(GC), b=-1, lty=2)
  legend(1.2, 2.2,
         legend=unique(paste(team, oppTeam, sep=" - ")), 
         text.col = unique(gameindex), 
         col = unique(gameindex), 
         pch = unique(gameindex), 
         cex=0.8, y.intersp=1.2, seg.len = 0) 
  #ncol=3, xjust=1, cex=0.8)
  points(GC~GS, col=gameindex, pch=gameindex, cex=2, lwd=3,
         data=rfpreddata %>% group_by(team, oppTeam, gameindex) %>% summarize(GC=mean(GC), GS=mean(GS)))   #xlim=c(0,2), ylim=c(0,2))
})


##################




allScores<-readRDS(file = paste0(intermediateOutput, ".rds"))
allScores<-inner_join(allScores, train_num, by=c("team", "oppTeam", "round", "gameindex", "season"))

eff_seed <- sample(1:2^15, 1)
print(sprintf("Seed for session: %s", eff_seed))
set.seed(eff_seed)

parameters<-data.frame(
  seed=eff_seed,
  gamma=25,#sample(c(0,10,15,20,25,40),1), # 0,1,3,5, 
  max_depth = 40, # 1,2,3,4,5,10, 20,30, 40,50,80  
  eta = 0.03, #sample(c(0.1,0.01,0.03, 0.3),1), # 0.001,0.003,
  subsample=0.6,#sample(c(0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 0.8),1), 
  colsample_bytree=0.4,#sample(c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8),1),
  nparallel = 4,
  algo=8
)
parameters<-data.frame(
  seed=eff_seed,
  gamma=1,#sample(c(0,10,15,20,25,40),1), # 0,1,3,5, 
  max_depth = 3, # 1,2,3,4,5,10, 20,30, 40,50,80  
  eta = 0.001, #sample(c(0.1,0.01,0.03, 0.3),1), # 0.001,0.003,
  subsample=0.45,#sample(c(0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 0.8),1), 
  colsample_bytree=0.4,#sample(c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8),1),
  nparallel = 4,
  algo=19
)
parameters<-data.frame(
  seed=eff_seed,
  gamma=0,#sample(c(0,10,15,20,25,40),1), # 0,1,3,5, 
  max_depth = 1, # 1,2,3,4,5,10, 20,30, 40,50,80  
  eta = 0.003, #sample(c(0.1,0.01,0.03, 0.3),1), # 0.001,0.003,
  subsample=0.45,#sample(c(0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 0.8),1), 
  colsample_bytree=0.05,#sample(c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8),1),
  nparallel = 4,
  algo=1
)
print (parameters)

xgbparamsFactor<-list(gamma=parameters$gamma, 
                      max_depth = parameters$max_depth, 
                      eta = parameters$eta, 
                      subsample=parameters$subsample, 
                      colsample_bytree=parameters$colsample_bytree)

allScores$weights<-1
trainData<-allScores
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
allScores<-trainScores

labels<-data.frame(
  GS=as.integer(substr(as.character(allScores$score),1,1)),
  GC=as.integer(substr(as.character(allScores$score),3,3)))
allgamesFillNA<-allScores %>% select(-gameindex, -score, -GS, -GC, -weights)        # , -season, -round 
allgamesFillNA[is.na(allgamesFillNA <- allgamesFillNA)] <- -999

#allgamesFillNA <-allgamesFillNA %>% select(team, oppTeam, round, team_promoted, oppTeam_promoted, starts_with("pG"), season)
allgamesFillNA <-allgamesFillNA %>% select(-starts_with("pois1"))


xgbMatrix<-sparse.model.matrix(~.-1, data = allgamesFillNA)
trainMatrix<-xgb.DMatrix(xgbMatrix, label=as.integer(allScores$score)-1, missing = -999, weight=allScores$weights)
watchlist<-list(train=trainMatrix)


##############################################
# Load new data

test_num<-read.csv(file=newSeason, stringsAsFactors = TRUE, as.is = FALSE)

test_num$team<-factor(test_num$team, levels=teams)
test_num$oppTeam<-factor(test_num$oppTeam, levels=teams)
test_num$season<-factor(test_num$season, levels=seasons)

testmatchpredictions<-readRDS(file = paste0("test_", intermediateOutput, ".rds"))  
testmatchpredictions<-inner_join(testmatchpredictions, test_num, by=c("team", "oppTeam", "round", "gameindex", "season"))

new_data<-testmatchpredictions %>% select(-gameindex, -GS, -GC)        

#new_data<-new_data%>%select(team, oppTeam, round, team_promoted, oppTeam_promoted, starts_with("pG"), season)
new_data <-new_data %>% select(-starts_with("pois"))

new_data[is.na(new_data <- new_data)] <- -999
#new_data[is.na(new_data <- new_data)] <- 0
testMatrix<-sparse.model.matrix(~.-1, data = new_data)
testLabels<-testmatchpredictions %>% select(GS, GC)   

#parameters$eta<-0.1

for (j in 1:10){
  if(file.exists(newpredfilename)){
    predictions<-read.csv(newpredfilename)
  } else {
    predictions<-data.frame()
  }
  
  model3<-xgb.train(nrounds =1,  #early_stopping_rounds = 50, 
                    params = xgbparamsFactor, 
                    data = trainMatrix , 
                    objective="multi:softmax", eval_metric="mlogloss", maximize=FALSE, 
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
  
  trainMatrix<-xgb.DMatrix(xgbMatrix, label=as.integer(allScores$score)-1, missing = -999, weight=allScores$weights)
  testMatrix<-xgb.DMatrix(sparse.model.matrix(~.-1, data = new_data, missing = -999))

  for (jj in 1:100){
    ptrain <- predict(model3, trainMatrix, outputmargin=TRUE)
    ptest  <- predict(model3, testMatrix, outputmargin=TRUE)
    setinfo(trainMatrix, "base_margin", ptrain) #+ptrain*rnorm(n=length(ptrain), mean = 0.01, sd = 0.01)) # disturb train weights with random noise
    setinfo(testMatrix, "base_margin", ptest)
    
    model3<-xgb.train(nrounds =1, 
                      params = xgbparamsFactor,#, min_child_weight=0,
                      data = trainMatrix , 
                      objective="multi:softmax", eval_metric="mlogloss", maximize=FALSE, 
                      print_every_n=1, verbose=1, 
                      nthread = 3,  num_class=nlevels(gLev) 
                      , watchlist = watchlist
    )
    wpoints<-data.frame(
      train_wpoints=mean(calcWeightedPoints(model3, trainMatrix)),
      train_maxp=mean(calcSoftMaxProb(preds = ptrain)))
    wpoint_log<-rbind(wpoint_log, wpoints)
    
    trainResult<-evaluateXgbFactorSimple(model3, trainMatrix, ylabels, "train XGB objective function")
    trainGS<-trainResult$gspred
    trainGC<-trainResult$gcpred
    trainScore<-mean(calcPoints(labels$GS, labels$GC, trainGS, trainGC, home=TRUE))
    ev<-cbind(model3$evaluation_log, train_avgscore=trainScore)
    evaluation_log<-rbind(evaluation_log, ev)
    w1<-evaluation_log$train_avgpoints / wpoint_log$train_wpoints
    
    if (jj %% 5 == 0){
      print(cbind(j, jj, wpoints, ev))
    }
    if (jj %% 20 == 0){
      printSoftMax(model3, testMatrix, trainMatrix)
    }
    #xgb.plot.deepness(model = model3)
    #xgb.dump(model=model3, fname="model3.save.txt", with_stats = TRUE, dump_format = "text")
    if (jj >= 10 & trainScore>2.2) {
      #ensResult<-c(ensResult, list(evaluateXgbFactorSimple(model3, testMatrix, actualGoals, "test ensemble XGB objective function")))   

      testResult<-evaluateXgbFactorSimple(model3, testMatrix, testLabels, "New games")
      #print(cbind(new_data %>% select(team, oppTeam), testResult))
      #printSoftMax(model3, testMatrix, trainMatrix)
      
      #models[[j]]<-model3
      predictions<-rbind(predictions, cbind(new_data %>% select(team, oppTeam), testResult, j=j, jj=jj, wpoints, ev[,2], ev[,3]))
    }
  }

  #plotImportance(model3, n=100, cols=colnames(trainMatrix))
  varImp<-printImportance(model3, n=ncol(trainMatrix), cols=colnames(trainMatrix))

  write.csv(predictions, newpredfilename, row.names = FALSE)
  
}
predictions<-read.csv(newpredfilename)

predictions %>% mutate(game=paste(team, oppTeam)) %>% dplyr::select(gspred, gcpred, game) %>% table

#read.csv("newpredictions_20180122.csv")%>% mutate(game=paste(team, oppTeam)) %>% dplyr::select(gspred, gcpred, game) %>% table


# print(predictions %>% arrange(team, gspred, gcpred))
# 
# with(predictions %>% arrange(j),
#      plot(train_avgscore ~ jj, col=j, type="l"))
# with(predictions %>% arrange(j),
#      plot(train_mlogloss ~ jj, col=j, type="l"))
# 
# with(predictions %>% filter(oppTeam=="Bayern Munich") %>% arrange(jj),
#      plot(train_avgscore ~ jj, col=j, type="l"))
# 
# with(predictions %>% filter(team=="FC Koln") %>% arrange(jj),
#      plot(paste(gspred,gcpred,sep=":") ~ jj))


library(RColorBrewer)
colors <- rep(brewer.pal(n = 12, name = "Set1"), times=4)
get_color<-function(gs,gc) colors[as.integer(factor(paste(gs, gc, sep=":"), levels = gLev))]
for (t in unique(predictions$team)) {
  with(predictions %>% filter(team==t) %>% arrange(jj),
       plot(train_avgscore+rnorm(n=nrow(predictions), sd=0.001) ~ I(jj+rnorm(n=nrow(predictions), sd=1)), 
            #col=colors[gspred+5*gcpred], 
            col=get_color(gspred, gcpred),
            pch=gspred+5*gcpred,
            main = unique(paste(team, oppTeam, sep=" - "))))
  gs<-as.integer(substr(as.character(gLev),1,1))
  gc<-as.integer(substr(as.character(gLev),3,3))
  legend("topleft", horiz = FALSE, bty="n", text.width = rep(0, 25),# seq(0,24,1)/(0:24), 
         legend=gLev, ncol=5,
         text.col = get_color(gs, gc), pch=gs+5*gc, col = get_color(gs, gc), 
         cex=1.2, x.intersp=0.2) 
  legend("bottomright", horiz = FALSE, bty="n", text.width = rep(0, 25),# seq(0,24,1)/(0:24), 
         legend=gLev, ncol=5,
         text.col = get_color(gs, gc), pch=gs+5*gc, col = get_color(gs, gc), 
         cex=1.2, x.intersp=0.2) 
}




with(predictions %>% filter(team=="Hannover") %>% arrange(jj),
     plot(train_avgscore+rnorm(n=nrow(predictions), sd=0.001) ~ I(jj+rnorm(n=nrow(predictions), sd=1)), 
          col=gspred+5*gcpred+1, 
          pch=gspred+5*gcpred+1,
          main = unique(paste(team, oppTeam, sep=" - "))))
gs<-as.integer(substr(as.character(gLev),1,1))
gc<-as.integer(substr(as.character(gLev),3,3))
legend("topleft", horiz = FALSE, bty="n", text.width = rep(0, 25),# seq(0,24,1)/(0:24), 
       legend=gLev, ncol=5,
       text.col = gs+5*gc+1, pch=gs+5*gc+1, col = gs+5*gc+1, 
       cex=0.8, x.intersp=0.1) 


with(predictions,# %>% filter(team=="Augsburg") %>% arrange(jj),
     plot(train_mlogloss ~ jj, col=gspred+5*gcpred+1, pch=j))
gs<-as.integer(substr(as.character(gLev),1,1))
gc<-as.integer(substr(as.character(gLev),3,3))
legend("topleft", horiz = FALSE, bty="n", text.width = rep(0, 25),# seq(0,24,1)/(0:24), 
       legend=gLev, ncol=5,
       text.col = gs+5*gc+1, 
       cex=0.8, x.intersp=0) 

with(predictions,# %>% filter(team=="Augsburg") %>% arrange(jj),
     plot(train_wpoints ~ jj, col=gspred+5*gcpred+1, pch=j))
gs<-as.integer(substr(as.character(gLev),1,1))
gc<-as.integer(substr(as.character(gLev),3,3))
legend("topleft", horiz = FALSE, bty="n", text.width = rep(0, 25),# seq(0,24,1)/(0:24), 
       legend=gLev, ncol=5,
       text.col = gs+5*gc+1, 
       cex=0.8, x.intersp=0) 

predictions %>% filter(oppTeam=="Bayern Munich") %>% select(gspred, gcpred, team, oppTeam) %>% table
predictions %>% mutate(game=paste(team, oppTeam)) %>% select(gspred, gcpred, game) %>% table

predictions %>% select(gspred, gcpred) %>% table

with(predictions %>% filter(team=="FC Koln" | TRUE) %>% arrange(jj),     gspred+5*gcpred+1)

#############################################


i<-5; printSoftMaxGame(model3, testMatrix, trainMatrix, i)
i<-1; printSoftMaxGame(models[[4]], testMatrix, trainMatrix, i)

new_data %>% select(team, oppTeam, round, team_promoted, oppTeam_promoted, starts_with("pG"), season, gameindex)
new_data %>% select(gameindex)

rfpred<-melt(new_data %>% select(team, oppTeam, round, team_promoted, oppTeam_promoted, starts_with("pG"), season, -starts_with("pGDiff")))
rfpreddata<-data.frame()
for (z in c(paste0("x",1:10), paste0("y",1:10))){
  rfpreddata<-rbind(rfpreddata,
    inner_join(
      rfpred %>% filter(variable==paste0("pGS.",z)) %>% select(team, oppTeam, GS=value),
      rfpred %>% filter(variable==paste0("pGC.",z)) %>% select(team, oppTeam, GC=value),
      by = c("team", "oppTeam")
    ) %>% mutate(diff=GS-GC, variable=z, teamid=as.integer(team))
  )
}
rfpreddata<-inner_join(rfpreddata, testmatchpredictions %>% select(gameindex, team, oppTeam), by = c("team", "oppTeam"))
rfpreddata %>% arrange(team, variable)
rfpreddata %>% filter(team=="Bayern Munich") %>% select(GS, GC) %>% plot() #xlim=c(0,2), ylim=c(0,2))
abline(a = 0, b=1)

with(rfpreddata, {
  plot(GS~GC, col=gameindex, pch=gameindex) #, xlim=c(0,2), ylim=c(0,2))
  abline(a = 0, b=1)
  abline(a = mean(GS)+mean(GC), b=-1, lty=2)
  legend(1.8, 2.0,
         legend=unique(paste(team, oppTeam, sep=" - ")), 
         text.col = unique(gameindex), 
         col = unique(gameindex), 
         pch = unique(gameindex), 
         cex=0.8, y.intersp=0.5, seg.len = 0) 
         #ncol=3, xjust=1, cex=0.8)
  points(GS~GC, col=gameindex, pch=gameindex, cex=2, lwd=3,
         data=rfpreddata %>% group_by(team, oppTeam, gameindex) %>% summarize(GS=mean(GS), GC=mean(GC)))   #xlim=c(0,2), ylim=c(0,2))
})
  

w<-which(allScores$round==1)
testResult<-evaluateXgbFactorSimple(model3, trainMatrix[w,], testLabels[w,], "First Round")
cbind(testResult, allScores$score[w], allScores$round[w], allScores$season[w])


ptest  <- predict(model3, newdata=testMatrix, outputmargin=TRUE, reshape = TRUE)
colnames(ptest)<-gLev
ptest


xgbMatrixPois<-sparse.model.matrix(~.-1, data = allgamesFillNA %>% select(team, oppTeam, round, season))
oob<-which(allgamesFillNA$season=="2016_17")
oob
poisTrainX<-xgbMatrixPois[-oob,]
poisTestX <-xgbMatrixPois[oob,]
poisTrainY<-(allScores[-oob,]) %>% select(GS, GC)
poisTestY <-(allScores[oob,]) %>% select(GS, GC)

dim(poisTrainX)
dim(poisTestX)
dim(poisTrainY)
dim(poisTestY)

trainMatrixGS<-xgb.DMatrix(poisTrainX, label=poisTrainY$GS, missing = -999)
watchlistGS<-list(train=trainMatrixGS)
trainMatrixGC<-xgb.DMatrix(poisTrainX, label=poisTrainY$GC, missing = -999)
watchlistGC<-list(train=trainMatrixGC)

modelPoisGS<-xgb.train(nrounds =1000,  #early_stopping_rounds = 50, 
                  params = xgbparamsFactor, 
                  data = trainMatrixGS , objective="count:poisson",
                  print_every_n=20, verbose=1, 
                  nthread = 10,  
                  watchlist = watchlistGS
)
modelPoisGC<-xgb.train(nrounds =1000,  #early_stopping_rounds = 50, 
                       params = xgbparamsFactor, 
                       data = trainMatrixGC , objective="count:poisson",
                       print_every_n=20, verbose=1, 
                       nthread = 10,  
                       watchlist = watchlistGC
)
ptestGS  <- predict(modelPoisGS, newdata=testMatrix)
ptestGC  <- predict(modelPoisGC, newdata=testMatrix)
print(cbind(new_data %>% select(team, oppTeam), ptestGS, ptestGC, diff=ptestGS-ptestGC))

ptestGS  <- predict(modelPoisGS, newdata=poisTestX)
ptestGC  <- predict(modelPoisGC, newdata=poisTestX)

printPoisResult <- function(modelPoisGS, modelPoisGC, poisTrainX, poisTrainY, trainMatrixGS) {
  ptrainGS  <- predict(modelPoisGS, newdata=poisTrainX)
  ptrainGC  <- predict(modelPoisGC, newdata=poisTrainX)
  print(summary(ptrainGS))
  print(summary(ptrainGC))
  print(cor(ptrainGS, poisTrainY$GS))
  print(cor(ptrainGC, poisTrainY$GC))
  #plotImportance(modelPoisGS, n=50, cols=colnames(trainMatrixGS))
  #varImp<-printImportance(modelPoisGS, n=ncol(trainMatrixGS), cols=colnames(trainMatrixGS))
  
  print(table(sign(ptrainGS-ptrainGC), (poisTrainY$GS-poisTrainY$GC)))
  print(mean(calcPoints((sign(ptrainGS-ptrainGC)+1)/2, (sign(ptrainGC-ptrainGS)+1)/2, poisTrainY$GS, poisTrainY$GC, home=TRUE)))
}

printPoisResult(modelPoisGS, modelPoisGC, poisTrainX, poisTrainY, trainMatrixGS)
printPoisResult(modelPoisGS, modelPoisGC, poisTestX, poisTestY, trainMatrixGS)

ptestGS  <- predict(modelPoisGS, newdata=poisTestX)
ptestGC  <- predict(modelPoisGC, newdata=poisTestX)
testGS<-(sign(ptestGS-ptestGC)+1)/2
testGC<-(sign(ptestGC-ptestGS)+1)/2
data.frame(poisTestY, testGS, testGC, row.names = NULL)
print(table(sign(ptestGS-ptestGC), (poisTestY$GS-poisTestY$GC)))
print(table(sign(ptestGS-ptestGC), sign(poisTestY$GS-poisTestY$GC)))


allScores %>% select(GS, GC, season) %>% group_by(season) %>% table()
