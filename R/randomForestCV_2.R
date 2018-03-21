setwd("~/LearningR/Bundesliga")

#install.packages("dplyr")
#install.packages("party")
##install.packages("rattle",  dependencies=c("Depends", "Suggests"))

#require(devtools)
#install.packages('xgboost')

library(dplyr)
#require(expm)
#library(car)
library(party)
#library(rattle)
# rattle()
require(Matrix)
#require(data.table)
library(xgboost)


#######################################################################################################################
# predict goals separately

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

evaluateMaxExpectation <- function(model, newdata) {
  predProbsList<-predict(model, newdata = newdata , type = "prob")
  predoutcomes<-t(sapply(predProbsList, cbind))
  colnames(predoutcomes)<-colnames(predProbsList[[1]])
  probmatrix<-predoutcomes
  predoutcomes<-predoutcomes[,order(colnames(predoutcomes))]  
  colnames(predoutcomes)<-substr(colnames(predoutcomes), 7,9)
  substr(colnames(predoutcomes), 2,2)<-":"

  mask<-mask[order(names(mask))]
  expectedValues<-sapply(1:25, function(i) predoutcomes %*% as.vector(t(mask[i][[1]])))
  colnames(expectedValues)<-names(mask)
  
  maxEV<-apply(expectedValues, 1, max)
  whichMaxEV<-apply(expectedValues, 1, which.max)
  
  pred<-colnames(expectedValues)[whichMaxEV]
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  
  ylabels<-newdata$score
  evpoints<-rowSums(probmatrix * maskm$home[ylabels,])
  result<-data.frame(pGS=gspred, pGC=gcpred, evPoints=evpoints)
  return(result)
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

#############################################################################################################
# xgboost functions


printLearningCurve<-function(model, maximise=FALSE, cutoff=1, info=list()){
  train_means<-model$evaluation_log[,2][[1]]
  test_means<-model$evaluation_log[,3][[1]]
  l<-length(train_means)
  ylim<-range(c(train_means[cutoff:l], test_means[cutoff:l]))
  plot(train_means, lwd=1, type="l", ylim=ylim, 
       main=paste(best_mean, best_iter), 
       #sub=printtext, 
       xlab="Iteration", ylab="Score")
  points(test_means, lwd=1, type="l", col="blue")
  if (maximise==TRUE) {
    best_mean<-max(test_means)
    best_iter<-which.max(test_means)
  } else {
    best_mean<-min(test_means)
    best_iter<-which.min(test_means)
  }
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
    print.data.frame (mat, nrows=n) 
  else
    print.data.frame (mat[1:n], nrows=n) 
  return(mat)
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


evaluateXgbFactorSimple<-function(model, newdata, newlabels, label="", levels1=gLev) {
  comparedata<-as.data.frame(as.matrix(newlabels))
  pred<-predict(model, newdata=newdata)
  pred<-levels1[pred+1]
  gspred<-as.integer(substr(pred,1,1))
  gcpred<-as.integer(substr(pred,3,3))
  # summarizePredictionSimple(data = comparedata, gspred, gcpred, label)
  return(data.frame(gspred, gcpred))
}

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
  #  print(table(sign(pGS-pGC), sign(GS-GC)))
  p1<-expand.grid(1:nlevels(gLev), preds+1)
  prob<-predest[cbind(p1$Var2,p1$Var1)]
  
  probmatrix<-predest[preds+1,]
  points<-rowSums(probmatrix * maskm$home[ylabels,])
  #print(summary(pL))
  #print(summary(points))
  #points<-points$points
  # print(points)
  
  c2<-expand.grid(lGS,GS)
  c3<-expand.grid(lGC,GC)
  
  c4<-calcPoints(GS1 = c2$Var1, GC1 = c3$Var1, GS2 = c2$Var2, GC2 = c3$Var2, home = TRUE)
  
  L <- expand.grid(lGS, pL)$Var2
  L <- expand.grid(lGS, points)$Var2
  
  #pc2<-expand.grid(lGS,pGS)
  #pc3<-expand.grid(lGC,pGC)
  
  #L<-calcPoints(GS1 = pc2$Var1, GC1 = pc3$Var1, GS2 = pc2$Var2, GC2 = pc3$Var2, home = c1$Var2==1)
  
  grad<-prob*(c4-L)*25 #+runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
  #grad<-prob*(c4-L)*25 +rnorm(n=length(preds)*nlevels(gLev), mean = 0, sd=10)
  #grad<-runif(n=length(preds)*nlevels(gLev), min=-0.15, max=0.15)
  hess<-grad*(1-2*prob)/2
  #hess<-rnorm(n=length(preds)*nlevels(gLev), mean = -0.5, sd=1)*grad*0.3
  #hess<-grad
  #hess<-runif(n=length(preds)*nlevels(gLev), min=-5.5, max=5.5)*grad*0.2
  
  #hess<-runif(n=length(preds)*nlevels(gLev), min=-0.04, max=0.04)*grad*0.2
  #hess<-rnorm(n=length(preds)*nlevels(gLev), mean = 0.01, sd = 0.04)*grad*0.2
  hess<-(grad*(1-2*prob))/20*rep(sample(c(-1,1), prob=c(0.7, 0.3), size = length(preds), replace=TRUE), nlevels(gLev))
  hess<-hess+rnorm(n=length(preds)*nlevels(gLev), mean=-3.0*sd(hess), sd=2*sd(hess))
  #hess<-grad*(1-2*prob)*0.18*rep(runif(n=length(preds), min=-0.05, max=0.05), nlevels(gLev))
  #+rnorm(n=length(preds)*nlevels(gLev), mean = -grad*(1-2*prob)/25, sd = 2*grad*(1-2*prob)/25)
  #hess<-rep(1.0, length(preds)*nlevels(gLev))
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
#  print(grad)
#  print(hess)
  return(list(grad= -grad, hess=hess))
}


################################################################################
# data preparation for learning

gLev<-sort(buildFactorScore(data.frame(expand.grid(GS=0:4, GC=0:4)))$score)
maskm<-buildMaskMatrix()
mask<-buildMask()
predest<-prediction_estimate()

xy<-read.csv(file="BLfeatures_2.csv", stringsAsFactors = TRUE, as.is = FALSE)

train_num<-xy
train_factor<-limitMaxScore(xy, 4) %>% buildFactorScore()
labels_num<-train_num %>% select(GS, GC)
labels_factor<-train_factor$score 
folds <-split(seq_along(xy$season), xy$season)


for (i in 1:1) {
  eff_seed <- sample(1:2^15, 1)
  print(sprintf("Seed for session: %s", eff_seed))
  set.seed(eff_seed)
  
  parameters<-data.frame(
    seed=eff_seed,
    ntree=sample(c(2000, 500, 1000),1),
    mtry=sample(c(1,2,3,4,5,6),1), # ,7,6
    minsplit = sample(c(10,12,15),1),
    nparallel = 1,
    algo=10
    #fraction = sample(c(0.632, 0.5, 0.3),1),
    #mincriterion = 0,
  )
  print (parameters)
  
  cforest_ctrl<-cforest_unbiased(
    ntree=parameters$ntree,
    mtry=parameters$mtry,
    minsplit = parameters$minsplit
  )
  
  for (fn in names(folds)){
    print(sprintf("Fold: %s", fn))
    f<-folds[[fn]]
    for (p in 1:parameters$nparallel){
      #print(sprintf("Loop repeat: %s", p))
      sub_seed <- sample(1:2^15, 1)
      #print(sprintf("Seed for session: %s", sub_seed))
      set.seed(sub_seed)

      model1 <- cforest(GS*GC ~ ., data=train_num[-f,], controls = cforest_ctrl)
      games <- predict_from_model1(model1, newdata=train_num[-f,])
      newgames <- predict_from_model1(model1, newdata=train_num[f,])
      
      if (parameters$algo==1) {
        model2<-cforest(GS*GC ~ . - (gameindex+season), data=games, controls = cforest_unbiased(mtry=4, ntree=100))
        train_pred2<-predict(model2, newdata=games)
        train_pred2<-matrix(data=unlist(train_pred2), ncol=2, byrow=TRUE, dimnames = list(NULL, c("pGS","pGC")))
        trainGS<-round(train_pred2[,1])
        trainGC<-round(train_pred2[,2])
        trainScore<-mean(calcPoints(games$GS, games$GC, trainGS, trainGC, home=TRUE))
        
        test_pred2<-predict(model2, newdata=newgames)
        test_pred2<-matrix(data=unlist(test_pred2), ncol=2, byrow=TRUE, dimnames = list(NULL, c("pGS","pGC")))
        testGS<-round(test_pred2[,1])
        testGC<-round(test_pred2[,2])
        testScore<-mean(calcPoints(newgames$GS, newgames$GC, testGS, testGC, home=TRUE))
        trainEV<-NA
        testEV<-NA
      }
      if (parameters$algo==2) {
        traingames<-games %>% limitMaxScore(4) %>% buildFactorScore()  
        model4<-cforest(score ~ . , data=traingames %>% select(-gameindex, -season), controls = cforest_unbiased(mtry=4, ntree=200))
        train_pred4<-evaluateMaxExpectation(model4, traingames)        
        trainGS<-train_pred4$pGS
        trainGC<-train_pred4$pGC
        trainScore<-mean(calcPoints(games$GS, games$GC, trainGS, trainGC, home=TRUE))
        trainEV<-mean(train_pred4$evPoints)        
        
        test_pred4<-evaluateMaxExpectation(model4, newgames %>% limitMaxScore(4) %>% buildFactorScore())
        testGS<-test_pred4$pGS
        testGC<-test_pred4$pGC
        testScore<-mean(calcPoints(newgames$GS, newgames$GC, testGS, testGC, home=TRUE))
        testEV<-mean(test_pred4$evPoints)        

      }                          

      if (parameters$algo==3 | parameters$algo==4 | parameters$algo==10) {
        traingames<-games %>% limitMaxScore(4) %>% buildFactorScore()  
        testgames<-newgames %>% limitMaxScore(4) %>% buildFactorScore()  
        
        # side_info<-train_num 
        # %>% 
        #   select(gameindex, round, season, team, oppTeam, where,
        #          mt_Shotstarget, mt_Shotstarget_where, t_diffRank, t_diffGoals_where, t_diffBothGoals_where,
        #          pois1_Shots, SVD_GDiff, t_diffGoals, mt_Shots, t_diffBothGoals, mt_Shots_where,
        #          t_diffBothGoals2H_where, SVD1_oppFouls, mt_Yellow_where, SVD_Shotstarget)
        
        sidehome<-train_num %>% filter(where=="Home") %>% select(-where)
        # sideaway<-side_info %>% filter(where=="Away") %>% mutate(z = team, team=oppTeam, oppTeam=z) %>% select(-where, -z)
        # side_info_joined<-inner_join(sidehome, sideaway, by=c("team", "oppTeam", "round", "gameindex", "season")) 
        
        #allgames<-rbind(games, newgames) %>% select(-GS, -GC, -gameindex, -season, -round)        
        allgames<-inner_join(rbind(games, newgames) , sidehome, by=c("team", "oppTeam", "round", "gameindex", "season", "GS", "GC")) %>%
          select(-GS, -GC, -gameindex, -season, -round)        
        
        allgamesFillNA<-allgames
        allgamesFillNA[is.na(allgamesFillNA <- allgames)] <- -999
        
        xgbMatrix<-sparse.model.matrix(~.-1, data = allgamesFillNA)
        trainMatrix<-xgb.DMatrix(xgbMatrix[1:nrow(games),], label=as.integer(traingames$score)-1, missing = -999)
        testMatrix <-xgb.DMatrix(xgbMatrix[(nrow(games)+1):(nrow(games)+nrow(newgames)),], label=as.integer(testgames$score)-1, missing = -999)

#        trainMatrix<-xgb.DMatrix(xgbMatrix[1:nrow(games),], label=games$GS-games$GC, missing = -999)
#        testMatrix <-xgb.DMatrix(xgbMatrix[(nrow(games)+1):(nrow(games)+nrow(newgames)),], label=label=newgames$GS-newgames$GC, missing = -999)
        
        watchlist<-list(train=trainMatrix, test=testMatrix)

        xgbparamsFactor<-list(gamma=sample(c(1),1), 
                              max_depth = sample(c(10),1), 
                              eta = 0.003, #sample(c(0.1,0.01,0.001,0.003,0.03, 0.0003),1), 
                              subsample=0.7,#sample(c(0.45,0.5),1), 
                              colsample_bytree=sample(c(0.8),1))
        
        model3<-xgb.train(nrounds =50,  #early_stopping_rounds = 50, 
                        params = xgbparamsFactor, 
                        data = trainMatrix , 
                        obj=objective, feval=evaltarget, maximize=TRUE, 
                        print_every_n=1, verbose=1, 
                        nthread = 10,  num_class=nlevels(gLev) 
                        , watchlist = watchlist
                        )
        
        #printLearningCurve(model3, maximise = TRUE)
  
        #plotImportance(model3, n=ncol(trainMatrix), cols=colnames(trainMatrix))
        #plotImportance(model3, n=50, cols=colnames(trainMatrix))
        #imp<-printImportance(model3, n=ncol(trainMatrix), cols=colnames(trainMatrix))

        trainResult<-evaluateXgbFactorSimple(model3, trainMatrix, games %>% select(GS, GC), "train XGB objective function")

        trainGS<-trainResult$gspred
        trainGC<-trainResult$gcpred
        trainScore<-mean(calcPoints(games$GS, games$GC, trainGS, trainGC, home=TRUE))
        trainEV<-NA
        
        testResult<-evaluateXgbFactorSimple(model3, testMatrix, newgames %>% select(GS, GC), "test XGB objective function")
        testGS<-testResult$gspred
        testGC<-testResult$gcpred
        testScore<-mean(calcPoints(newgames$GS, newgames$GC, testGS, testGC, home=TRUE))
        testEV<-NA
        
      }                          
      
            #result_rf_log<-data.frame()
      result_rf_log<-read.csv(file = "rf\\result_rf_log.csv")
      result<-data.frame(time=date(), parameters, sub_seed=sub_seed, fold=fn, k=p,
                         trainScore=trainScore,
                         testScore=testScore,
                         trainEV=trainEV, testEV=testEV)
      result_rf_log<-rbind(result_rf_log, result)
      write.csv(result_rf_log, file = "rf\\result_rf_log.csv", row.names = FALSE)

      pdf(paste0("rf\\plot", sub_seed, ".pdf"))
      summarizePredictionSimple(games, trainGS, trainGC, label=paste("Train", paste(names(result), result, sep="=", collapse = ", ")), verbose=0)
      summarizePredictionSimple(newgames, testGS, testGC, label=paste("Test", paste(names(result), result, sep="=", collapse = ", ")), verbose=0)
      dev.off()
      sink(paste0("rf\\cvdata", sub_seed, ".log"))
      cat("seed=", sub_seed, "\n")
      print(result)
      summarizePredictionSimple(games, trainGS, trainGC, label=paste("Train", paste(names(result), result, sep="=", collapse = ", ")), verbose=1)
      summarizePredictionSimple(newgames, testGS, testGC, label=paste("Test", paste(names(result), result, sep="=", collapse = ", ")), verbose=1)
      sink()
      print(result)
      
      #vi<-sort(varimp(model1), decreasing = TRUE)
      #plot(vi)
      
    }
  }
}    

result_rf_log %>% 
  group_by(seed, fold, ntree, mtry, minsplit, nparallel, algo) %>%
  summarize(train=mean(trainScore), trainsd=sd(trainScore), test=mean(testScore), testsd=sd(testScore)) %>% 
  arrange(-test) %>%
  print.data.frame()

result_rf_log %>% 
  group_by(seed, ntree, mtry, minsplit, nparallel, algo) %>%
  summarize(train=mean(trainScore), trainsd=sd(trainScore), test=mean(testScore), testsd=sd(testScore)) %>% 
  arrange(-test) %>%
  print.data.frame()

# varimp_model <- cforest(GS ~ ., data=train_num[-f,], controls = cforest_ctrl)
# vi<-sort(varimp(varimp_model), decreasing = TRUE)
# plot(vi)
# vi



aggregate(testScore~seed+fold+ntree+mtry+minsplit+nparallel, data=result_rf_log, FUN=mean)
  
plot(y=result_rf_log$testScore, x=result_rf_log$trainScore)

smoothScatter((result_rf_log %>% filter(algo==3))$trainScore, (result_rf_log %>% filter(algo==3))$testScore)
abline(lm(testScore~trainScore, data=result_rf_log %>% filter(algo==3)))

plot(testScore~trainScore, data=result_rf_log %>% filter(algo==3))
abline(lm(testScore~trainScore, data=result_rf_log %>% filter(algo==3)))

plot(lm(testScore~trainScore, data=result_rf_log))

plot(testScore~mtry, data=result_rf_log)
boxplot(testScore~mtry, data=result_rf_log %>% filter(algo==3))
boxplot(testScore~mtry*fold, data=result_rf_log %>% filter(algo==3 & ntree==2000))
boxplot(testScore~ntree, data=result_rf_log%>% filter(algo==3))
boxplot(testScore~ntree+fold, data=result_rf_log%>% filter(algo==3))
boxplot(testScore~(minsplit), data=result_rf_log%>% filter(algo==1))
boxplot(testScore~(minsplit*fold), data=result_rf_log%>% filter(algo==3 & mtry==3 & ntree==2000))

boxplot(testScore~mtry*minsplit*ntree*fold, data=result_rf_log %>% filter(algo==3 & mtry==2))
boxplot(testScore~mtry*minsplit*ntree, data=result_rf_log %>% filter(algo==3 & ntree==2000))
boxplot(testScore~mtry*minsplit*ntree, data=result_rf_log %>% filter(algo==3 & ntree==2000), plot=FALSE)

boxplot(testScore~mtry*fold, data=result_rf_log %>% filter(algo==2 & ntree==1000))

result_rf_log%>% filter(algo==2, ntree==1000) %>% group_by(minsplit) %>%
  summarize(train=mean(trainScore), trainsd=sd(trainScore), test=mean(testScore), testsd=sd(testScore)) %>% 
  arrange(test) %>%
  print.data.frame()

plot(testScore~(mtry), data=result_rf_log)
plot(testScore~(ntree), data=result_rf_log)
plot(testScore~(minsplit), data=result_rf_log)
summary(result_rf_log)

plot(testScore~trainScore, data=result_rf_log)
plot(testScore~trainScore, data=result_rf_log)

result_rf_log %>% filter(algo==3& ntree==2000) %>%
  group_by(mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, mtry)

result_rf_log %>% filter(algo==3 ) %>%
  group_by(ntree, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, ntree)


result_rf_log %>% filter(algo==3& ntree==2000 & mtry==2) %>%
  group_by(minsplit, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, minsplit)

result_rf_log %>% filter(algo==3 & mtry==2 & minsplit==15 & ntree==2000) %>%
  group_by(minsplit, ntree, mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  arrange(fold, ntree, minsplit, mtry) %>%
  print.data.frame()


result_rf_log %>% filter(algo==3) %>%
  group_by(minsplit, ntree, mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  mutate(meansd=mean-2*sd) %>%
  arrange(meansd, fold, ntree, minsplit, mtry) %>%
  print.data.frame()

result_rf_log %>% filter(algo==3 & ntree==2000 & mtry==2 & minsplit==15) %>%
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
  

result_rf_log %>% filter(algo==3) %>%
  group_by(minsplit, ntree, mtry, fold) %>% 
  summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
  ungroup() %>%
  select(sdTrain, sd) %>%
  plot() 
abline(lm(sd~sdTrain, data = result_rf_log %>% filter(algo==3) %>%
            group_by(minsplit, ntree, mtry, fold) %>% 
            summarise(mean = mean(testScore), sd=sd(testScore), n=n(), meanTrain=mean(trainScore), sdTrain=sd(trainScore)) %>%
            ungroup()))


t/sum(t)

str(summary(runif(100)))


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
  write.csv(resultlog, file = "result_rf_log.csv", row.names = FALSE)
}




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
##########################################################################################################################


summarizePrediction(data = test, test$GS, test$GC, "100% correct") # 1618
summarizePrediction(data = test, ifelse(test$where=="Home", 2, 1), ifelse(test$where=="Home", 1, 2), "2:1 home win") # 411
summarizePrediction(data = test, ifelse(test$where=="Home", 1, 2), ifelse(test$where=="Home", 2, 1), "1:2 away win") # 407

homegames<-train %>% filter(where=="Home") %>% select(GS, GC)
randidx<-sample(nrow(homegames), size = nrow(test), replace=TRUE)
summarizePrediction(data = test, 
                    ifelse(test$where=="Home", homegames[randidx,1], homegames[randidx,2]), 
                    ifelse(test$where=="Home", homegames[randidx,2], homegames[randidx,1]), "random draw from distribution") # 340


summarizePrediction(data = test, test$GS, test$GC, "random guess")



evaluateNumeric<-function(model, newdata, label="") {
  pred<-predict(model, newdata=newdata)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  summarizePrediction(data = newdata, round(gspred), round(gcpred), label)
  return(data.frame(gspred, gcpred))
}


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


varimp_model <- cforest(GS ~ ., data=train_num[-f,], controls = cforest_ctrl)
vi<-sort(varimp(varimp_model), decreasing = TRUE)
plot(vi)
vi


vi<-sort(varimp(model4), decreasing = TRUE)
plot(vi)
vi

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

