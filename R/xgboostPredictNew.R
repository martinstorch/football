setwd("~/LearningR/Bundesliga")

#install.packages("dplyr")
#install.packages("party")
##install.packages("rattle",  dependencies=c("Depends", "Suggests"))

#require(devtools)
#install.packages('xgboost')

inputFeatures<-c("BLfeatures_2.csv")
outputFeatures<-"BLfeatures_2013.csv"
intermediateOutput<-"pred_2013_rf_features"

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



objective<-function(preds, dtrain)
{
  ylabels<-getinfo(dtrain, "label")+1
  labels<-levels(gLev)[ylabels]
  predlabels<-levels(gLev)[preds+1]
  GS<-as.integer(substr(as.character(labels), start = 1, stop = 1))
  GC<-as.integer(substr(as.character(labels), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))
  pL<-calcPoints(GS1 = GS, GC1 = GC, GS2 = pGS, GC2 = pGC, home = TRUE)
  
  lGS<-as.integer(substr(as.character(levels(gLev)), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(levels(gLev)), start = 3, stop = 3))
  p1<-expand.grid(1:nlevels(gLev), preds+1)
  prob<-predest[cbind(p1$Var2,p1$Var1)]
  
  probmatrix<-predest[preds+1,]
  points<-rowSums(probmatrix * maskm$home[ylabels,])
  c2<-expand.grid(lGS,GS)
  c3<-expand.grid(lGC,GC)
  
  c4<-calcPoints(GS1 = c2$Var1, GC1 = c3$Var1, GS2 = c2$Var2, GC2 = c3$Var2, home = TRUE)
  
  L <- expand.grid(lGS, pL)$Var2
  L <- expand.grid(lGS, points)$Var2
  
  grad<-prob*(c4-L)*parGrad #+runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
  #hess<-grad*(1-2*prob)/2
  hess<-(grad*(1-2*prob))/parGrad*rep(sample(c(-1,1), prob=c(parHessNegProb, 1-parHessNegProb), size = length(preds), replace=TRUE), nlevels(gLev))
  hess<-hess+rnorm(n=length(preds)*nlevels(gLev), mean=parHessShift*sd(hess), sd=parHessShiftSd*sd(hess))
  return(list(grad= -grad, hess=hess))
}

parGrad<-25
parHessNegProb<-0.7
parHessShift<- -3
parHessShiftSd<-2


################################################################################
# data preparation for learning


mask<-buildMask()
gLev<-sort(buildFactorScore(data.frame(expand.grid(GS=0:4, GC=0:4)))$score)
predest<-prediction_estimate()
maskm<-buildMaskMatrix()

xy<-data.frame()
for (f in inputFeatures){
  xyf<-read.csv(file=f, stringsAsFactors = TRUE, as.is = FALSE)
  xy<-rbind(xy, xyf)
}


test_num<-read.csv(file=outputFeatures, stringsAsFactors = TRUE, as.is = FALSE)
if ("GS" %in% colnames(test_num)) {
  test_num <- test_num %>% select(-GS, -GC)
}

teams<-levels(as.factor(c(as.character(xy$team), as.character(test_num$team))))
seasons<-levels(as.factor(c(as.character(xy$season), as.character(test_num$season))))

xy$team<-factor(xy$team, levels=teams)
xy$oppTeam<-factor(xy$oppTeam, levels=teams)
xy$season<-factor(xy$season, levels=seasons)
test_num$team<-factor(test_num$team, levels=teams)
test_num$oppTeam<-factor(test_num$oppTeam, levels=teams)
test_num$season<-factor(test_num$season, levels=seasons)


train_num<-xy

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
print (parameters)

# algo=8 
parGrad<-30
parHessNegProb<-0.5
parHessShift<- -2
parHessShiftSd<-1.5

xgbparamsFactor<-list(gamma=parameters$gamma, 
                      max_depth = parameters$max_depth, 
                      eta = parameters$eta, 
                      subsample=parameters$subsample, 
                      colsample_bytree=parameters$colsample_bytree)



labels<-data.frame(
  GS=as.integer(substr(as.character(allScores$score),1,1)),
  GC=as.integer(substr(as.character(allScores$score),3,3)))
allgamesFillNA<-allScores %>% select(-gameindex, -score, -GS, -GC)        # , -season, -round 
allgamesFillNA[is.na(allgamesFillNA <- allgamesFillNA)] <- -999


xgbMatrix<-sparse.model.matrix(~.-1, data = allgamesFillNA)
trainMatrix<-xgb.DMatrix(xgbMatrix, label=as.integer(allScores$score)-1, missing = -999)
watchlist<-list(train=trainMatrix)
      
model3<-xgb.train(nrounds =100,  #early_stopping_rounds = 50, 
                  params = xgbparamsFactor2, 
                  data = trainMatrix , 
                  obj=objective2, feval=evaltarget, maximize=TRUE, 
                  print_every_n=1, verbose=1, 
                  nthread = 10,  num_class=nlevels(gLev) 
                  , watchlist = watchlist
)


#plotImportance(model3, n=50, cols=colnames(trainMatrix))
#varImp<-printImportance(model3, n=ncol(trainMatrix), cols=colnames(trainMatrix))

trainResult<-evaluateXgbFactorSimple(model3, trainMatrix, labels, "train XGB objective function")
trainGS<-trainResult$gspred
trainGC<-trainResult$gcpred
trainScore<-mean(calcPoints(labels$GS, labels$GC, trainGS, trainGC, home=TRUE))
trainScore
summarizePredictionSimple(labels, trainGS, trainGC, label="Train", verbose=0)
plot(model3$evaluation_log$train_avgpoints[20:200], type="l")

##############################################
# Load new data

test_num<-read.csv(file=outputFeatures, stringsAsFactors = TRUE, as.is = FALSE)

test_num$team<-factor(test_num$team, levels=teams)
test_num$oppTeam<-factor(test_num$oppTeam, levels=teams)
test_num$season<-factor(test_num$season, levels=seasons)

testmatchpredictions<-readRDS(file = paste0("test_", intermediateOutput, ".rds"))  
testmatchpredictions<-inner_join(testmatchpredictions, test_num, by=c("team", "oppTeam", "round", "gameindex", "season"))

new_data<-testmatchpredictions %>% select(-gameindex, -GS, -GC)        
new_data[is.na(new_data <- new_data)] <- -999
testMatrix<-sparse.model.matrix(~.-1, data = new_data)
testLabels<-testmatchpredictions %>% select(GS, GC)   

testResult<-evaluateXgbFactorSimple(model3, testMatrix, testLabels, "New data")
table(testResult)
testScore<-mean(calcPoints(testLabels$GS, testLabels$GC, testResult$gspred, testResult$gcpred, home=TRUE))
testScore
summarizePredictionSimple(testLabels, testResult$gspred, testResult$gcpred, label="New data", verbose=1)

points_per_round<-sapply(1:max(new_data$round), function(r) 
  calcPoints(testLabels$GS[new_data$round==r], testLabels$GC[new_data$round==r], 
             testResult$gspred[new_data$round==r], testResult$gcpred[new_data$round==r], home=TRUE))
sum(points_per_round)
plot(colSums(points_per_round), type="b", main=paste("Points per round / Total =", sum(points_per_round)), ylab="Points from prediction")

points_per_tree<-sapply(1:model3$niter, function(r) {
  pred<-predict(object=model3, newdata = testMatrix, ntreelimit = r)
  pScore<-gLev[pred+1]
  return(calcPoints(testLabels$GS, testLabels$GC, 
                    as.integer(substr(as.character(pScore),1,1)), 
                    as.integer(substr(as.character(pScore),3,3)), 
                    home=TRUE))
})
colSums(points_per_tree)
plot(colSums(points_per_tree)[10:200], type="l")
points_per_tree
summary(rowMeans(points_per_tree[,20:100]))
sum(rowMeans(points_per_tree[,20:100]))
sum(apply(points_per_tree[,20:100], 1, max))
sum(apply(points_per_tree[,1:100], 1, max))
sum(apply(points_per_tree[,50:100], 1, max))
sum(apply(points_per_tree[,90:100], 1, max))
sum(apply(points_per_tree[,20:50], 1, max))
sum(apply(points_per_tree[,20:50], 1, max))


predictions_per_tree<-sapply(1:model3$niter, function(r) predict(object=model3, newdata = testMatrix, ntreelimit = r)) 
scores_per_tree<-gLev[predictions_per_tree+1]
dim(scores_per_tree)<-list(nrow=nrow(testLabels), ncol=model3$niter)

pred_matrix<-paste(scores_per_tree, points_per_tree, sep="/")
dim(pred_matrix)<-list(nrow=nrow(testLabels), ncol=model3$niter)

pred_matrix

cbind(new_data$team, new_data$oppTeam, testLabels, pred_matrix[,100])

table(pred_matrix[,10:100])
table(pred_matrix[,20:100])
table(pred_matrix[,100])

cbind(table(pred_matrix[3,20:100]), pred_matrix[3,100])
votes<-lapply(1:nrow(testLabels), function(i) table(pred_matrix[i,20:100]))
names(votes)<-paste(testLabels$GS, testLabels$GC, pred_matrix[,100], sep=":")
votes

median_prediction<-apply(scores_per_tree[,20:100], 1, median)
median_scores_per_tree<-gLev[median_prediction]

median_points<-calcPoints(testLabels$GS, testLabels$GC, 
           as.integer(substr(as.character(median_scores_per_tree),1,1)), 
           as.integer(substr(as.character(median_scores_per_tree),3,3)), 
           home=TRUE)
sum(median_points)
e<-cbind(p50=points_per_tree[,50], p100=points_per_tree[,100], med=median_points)
w<-which(e[,3]!=e[,2] | e[,3]!=e[,1] | e[,1]!=e[,2])
w<-which(e[,3]!=e[,2])
e[w,]
votes[w]

raisin_picking_pred<-scores_per_tree[,100]
raisin_picking_pred[apply(scores_per_tree[,20:100], 1, function (x) any(x==6))]<-"0:2"
raisin_picking_pred[apply(scores_per_tree[,20:100], 1, function (x) any(x==20))]<-"2:0"
raisin_picking_pred[apply(scores_per_tree[,20:100], 1, function (x) any(x==5))]<-"1:3"
raisin_picking_pred[apply(scores_per_tree[,20:100], 1, function (x) any(x==21))]<-"3:1"
table(raisin_picking_pred)
rp_points<-calcPoints(testLabels$GS, testLabels$GC, 
                          as.integer(substr(as.character(raisin_picking_pred),1,1)), 
                          as.integer(substr(as.character(raisin_picking_pred),3,3)), 
                          home=TRUE)
sum(rp_points)
summarizePredictionSimple(testLabels, as.integer(substr(as.character(raisin_picking_pred),1,1)), as.integer(substr(as.character(raisin_picking_pred),3,3)), label="Raisin Picking", verbose=1)

w<-which(rp_points!=points_per_tree[,100])
cbind(paste(raisin_picking_pred[w], rp_points[w]), testLabels$GS[w], testLabels$GC[w], pred_matrix[w,100])



pred<-predict(object=model3, newdata = testMatrix, ntreelimit = 50, reshape = TRUE, outputmargin = TRUE)
colnames(pred)<-gLev
summary(pred)
sort(colMeans(pred))
apply(pred, 1, max)
table(colnames(pred)[apply(pred, 1, which.max)])

maxPrediction<-colnames(pred)[apply(pred, 1, which.max)]
summarizePredictionSimple(testLabels, 
                          as.integer(substr(as.character(maxPrediction),1,1)), 
                          as.integer(substr(as.character(maxPrediction),3,3)), 
                          label="Max Prediction", verbose=1)

sec.fun <- function (x) {
  which.max( x[x!=max(x)] )
}

max2Prediction<-colnames(pred)[apply(pred, 1, sec.fun)]
summarizePredictionSimple(testLabels, 
                          as.integer(substr(as.character(max2Prediction),1,1)), 
                          as.integer(substr(as.character(max2Prediction),3,3)), 
                          label="2nd Max Prediction", verbose=1)


prediction_estimate<-function(){
  lGS<-as.integer(substr(as.character(gLev), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(gLev), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(gLev), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(gLev), start = 3, stop = 3))
  p1<-expand.grid(lGS, pGS)
  p2<-expand.grid(lGC, pGC)
  pp<-dpois(x=p1$Var1, lambda = (p1$Var2+ifelse(p1$Var2==0, 0.85, 0.45) )) * dpois(x=p2$Var1, lambda = (p2$Var2+ifelse(p2$Var2==0, 0.6, 0.3))) 
  
  p3<-ifelse(p1$Var1==p1$Var2 & p2$Var1==p2$Var2, 1, 0) # full hit
  p4<-abs((p1$Var1-p2$Var1) - (p1$Var2-p2$Var2))*-0.4 + 2 # goal diff score 
  p5<-ifelse(sign(p1$Var1-p2$Var1) == sign(p1$Var2-p2$Var2), 0.5, 0) # tendendy score 
  p6<-abs((p1$Var1+p2$Var1) - (p1$Var2+p2$Var2))*-0.05 + 0.7 # total goals  score 
  p7<-(2*p3+3*p4+p5+p6)
  p7<-(1*p3+3*p4+5*p5+8*p6)
  p7<-pp
  #p7<-p7-min(p7)

  predest<-matrix(p7, ncol=25, byrow = TRUE)
  colnames(predest)<-gLev
  rownames(predest)<-gLev
  predest<-predest-apply(predest, 1, min)
  predest<-predest/rowSums(predest)
}
predest<-prediction_estimate()
sort(colSums(predest))
sort(predest["0:0",])
sort(predest["1:1",])
sort(predest["2:2",])
sort(predest["0:1",])
sort(predest["0:2",])
sort(predest["1:2",])
sort(predest["1:0",])
sort(predest["2:0",])
sort(predest["2:1",])
sort(predest["0:4",])
sort(predest["3:4",])

xgb.plot.tree(feature_names = colnames(xgbMatrix), model = model3, n_first_tree = 30)
library(DiagrammeR)
install.package("DiagrammeR")

xgb.plot.multi.trees(feature_names = colnames(xgbMatrix), model = model3)

xgb.dump(model=model3, fname="model3.save.txt", with_stats = TRUE, dump_format = "text")

objective2<-function(preds, dtrain)
{
  ylabels<-getinfo(dtrain, "label")+1
  labels<-levels(gLev)[ylabels]
  predlabels<-levels(gLev)[preds+1]
  GS<-as.integer(substr(as.character(labels), start = 1, stop = 1))
  GC<-as.integer(substr(as.character(labels), start = 3, stop = 3))
  pGS<-as.integer(substr(as.character(predlabels), start = 1, stop = 1))
  pGC<-as.integer(substr(as.character(predlabels), start = 3, stop = 3))
  pL<-calcPoints(GS1 = GS, GC1 = GC, GS2 = pGS, GC2 = pGC, home = TRUE)
  
  lGS<-as.integer(substr(as.character(levels(gLev)), start = 1, stop = 1))
  lGC<-as.integer(substr(as.character(levels(gLev)), start = 3, stop = 3))
  p1<-expand.grid(1:nlevels(gLev), preds+1)
  prob<-predest[cbind(p1$Var2,p1$Var1)]
  
  probmatrix<-predest[preds+1,]
  points<-rowSums(probmatrix * maskm$home[ylabels,])
  c2<-expand.grid(lGS,GS)
  c3<-expand.grid(lGC,GC)
  
  c4<-calcPoints(GS1 = c2$Var1, GC1 = c3$Var1, GS2 = c2$Var2, GC2 = c3$Var2, home = TRUE)
  
  L <- expand.grid(lGS, pL)$Var2
  L <- expand.grid(lGS, points)$Var2
  
  grad<-prob*(c4-L) # *parGrad #+runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
  #hess<-grad*(1-2*prob)/2
  hess<-(grad*(1-2*prob)) # /parGrad*rep(sample(c(-1,1), prob=c(parHessNegProb, 1-parHessNegProb), size = length(preds), replace=TRUE), nlevels(gLev))
  hess<-(prob*(1-2*prob)) # /parGrad*rep(sample(c(-1,1), prob=c(parHessNegProb, 1-parHessNegProb), size = length(preds), replace=TRUE), nlevels(gLev))
  # hess<-hess+rnorm(n=length(preds)*nlevels(gLev), mean=parHessShift*sd(hess), sd=parHessShiftSd*sd(hess))
  #grad<-ifelse(p1$Var2==1 & c2$Var2==0 & c3$Var2==4 & (c2$Var1!=c3$Var1 | c2$Var2!=c3$Var2), -100, grad) # jump out of 0:4 
#  print(table(predlabels))
#  if(min(preds)==max(preds+1)){
#    grad<-runif(n = length(grad), min = -1, max=1)
#    hess<-rep(1, length(grad))
#  }
#  print(grad[1:50])
#  print(sprintf("sum(grad)=%f, sum(hess)=%f, sum(grad)^2/sum(hess)=%f", sum(grad), sum(hess), sum(grad)^2/sum(hess)))
#  print(hess[1:50])
#  print(sum(hess))
  return(list(grad= -grad, hess=hess))
}
xgbparamsFactor2<-xgbparamsFactor
xgbparamsFactor2$gamma=0
xgbparamsFactor2$min_child_weight=1
xgbparamsFactor2$max_delta_step=0
xgbparamsFactor2$lambda=0
xgbparamsFactor2$alpha=0

model_trial<-xgb.train(nrounds =100, #early_stopping_rounds = 50, 
                  params = xgbparamsFactor2, #base_score=0.5, scale_pos_weight = 1000000,
                  data = trainMatrix , 
                  obj=objective2, feval=evaltarget, maximize=TRUE, 
                  print_every_n=1, verbose=1, 
                  nthread = 10,  num_class=nlevels(gLev) 
                  , watchlist = watchlist
)
xgb.plot.tree(feature_names = colnames(xgbMatrix), model = model_trial, n_first_tree = 20)
xgb.plot.multi.trees(feature_names = colnames(xgbMatrix), model = model_trial)
xgb.plot.deepness(model = model_trial)
xgb.dump(model=model_trial, fname="model_trial.save.txt", with_stats = TRUE, dump_format = "text", feature_names = colnames(xgbMatrix))
colnames(xgbMatrix)[187]

pred_trial<-predict(object=model_trial, newdata = trainMatrix, reshape = TRUE, outputmargin = TRUE)
colnames(pred_trial)<-gLev
pred_trial[1:6,]
predict(object=model_trial, newdata = trainMatrix, reshape = FALSE, outputmargin = FALSE)[1:6]
