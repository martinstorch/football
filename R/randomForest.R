download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
download.file("http://www.football-data.co.uk/mmz4281/1415/D1.csv", "BL2014.csv")

library(dplyr)
require(expm)
library(car)
library(party)

data1<-read.csv("BL2016.csv")
data1$season<-2016
data2<-read.csv("BL2015.csv")
data2$season<-2015
data3<-read.csv("BL2014.csv")
data3$season<-2014
data<-rbind(data3[,colnames(data1)], data2, data1)
teams <- unique(data$HomeTeam)
results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'Date', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR' )]
results$season<-as.factor(results$season)
table(results$FTR , results$season)
results$spieltag <- floor((9:(nrow(results)+8))/9)
results$round <- ((results$spieltag-1) %% 34) +1
results$Date<-as.Date(results$Date, "%d/%m/%y")
results$dayofweek<-weekdays(results$Date)
results$gameindex<-(0:(nrow(results)-1))%%9+1
teamresults <- data.frame(team=results$HomeTeam, oppTeam=results$AwayTeam, 
                          GS=results$FTHG, GC=results$FTAG, where="Home", round=results$round, season=results$season, 
                          GS1H=results$HTHG, GC1H=results$HTAG, GS2H=results$FTHG-results$HTHG, GC2H=results$FTAG-results$HTAG, 
                          dow=results$dayofweek, gameindex=results$gameindex)
teamresults <- rbind(data.frame(team=results$AwayTeam, oppTeam=results$HomeTeam, 
                                GS=results$FTAG, GC=results$FTHG, where="Away", round=results$round, season=results$season,
                                GS1H=results$HTAG, GC1H=results$HTHG, GS2H=results$FTAG-results$HTAG, GC2H=results$FTHG-results$HTHG, 
                                dow=results$dayofweek, gameindex=results$gameindex),
                     teamresults)

teamresults<-teamresults[order(teamresults$season, teamresults$round, teamresults$team),]

teamresults$points<-ifelse(
  sign(teamresults$GS - teamresults$GC)==1, 3, 
  ifelse(teamresults$GS == teamresults$GC, 1, 0))

teamresults$oppPoints<-ifelse(
  sign(teamresults$GS - teamresults$GC)==1, 0, 
  ifelse(teamresults$GS == teamresults$GC, 1, 3))


teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) %>%
  mutate(t_Points = cumsum(points)-points) %>%
  mutate(t_GS = cumsum(GS)-GS) %>%
  mutate(t_GC = cumsum(GC)-GC) %>%
  mutate(t_GS1H = cumsum(GS1H)-GS1H) %>%
  mutate(t_GC1H = cumsum(GC1H)-GC1H) %>%
  mutate(t_GS2H = cumsum(GS2H)-GS2H) %>%
  mutate(t_GC2H = cumsum(GC2H)-GC2H) %>%
  mutate(t_Matches_total = ifelse(round>1, round-1, 1)) %>%
  group_by(season, where, team) %>%
  arrange(round) %>%
  mutate(t_Points_where = cumsum(points)-points) %>%
  mutate(t_GS_where = cumsum(GS)-GS) %>%
  mutate(t_GC_where = cumsum(GC)-GC) %>%
  mutate(t_GS1H_where = cumsum(GS1H)-GS1H) %>%
  mutate(t_GC1H_where = cumsum(GC1H)-GC1H) %>%
  mutate(t_GS2H_where = cumsum(GS2H)-GS2H) %>%
  mutate(t_GC2H_where = cumsum(GC2H)-GC2H) %>%
  mutate(t_Matches_where = ifelse(row_number()>1, row_number()-1, 1)) %>%
  group_by(season, oppTeam) %>%
  arrange(round) %>%
  mutate(t_oppPoints = cumsum(oppPoints)-oppPoints) %>% 
  mutate(t_oppGS = cumsum(GC)-GC) %>%
  mutate(t_oppGC = cumsum(GS)-GS) %>%
  mutate(t_oppGS1H = cumsum(GC1H)-GC1H) %>%
  mutate(t_oppGC1H = cumsum(GS1H)-GS1H) %>%
  mutate(t_oppGS2H = cumsum(GC2H)-GC2H) %>%
  mutate(t_oppGC2H = cumsum(GS2H)-GS2H) %>%
  group_by(season, where, oppTeam) %>%
  arrange(round) %>%
  mutate(t_oppPoints_where = cumsum(oppPoints)-oppPoints) %>% 
  mutate(t_oppGS_where = cumsum(GC)-GC) %>%
  mutate(t_oppGC_where = cumsum(GS)-GS) %>%
  mutate(t_oppGS1H_where = cumsum(GC1H)-GC1H) %>%
  mutate(t_oppGC1H_where = cumsum(GS1H)-GS1H) %>%
  mutate(t_oppGS2H_where = cumsum(GC2H)-GC2H) %>%
  mutate(t_oppGC2H_where = cumsum(GS2H)-GS2H) %>%
  mutate(t_oppMatches_where = ifelse(row_number()>1, row_number()-1, 1)) %>%
  ungroup()

teamresults<-
  teamresults %>%
  mutate(mt_Points = t_Points/t_Matches_total) %>%
  mutate(mt_GS = t_GS/t_Matches_total) %>%
  mutate(mt_GC = t_GC/t_Matches_total) %>%
  mutate(mt_GS1H = t_GS1H/t_Matches_total) %>%
  mutate(mt_GC1H = t_GC1H/t_Matches_total) %>%
  mutate(mt_GS2H = t_GS2H/t_Matches_total) %>%
  mutate(mt_GC2H = t_GC2H/t_Matches_total) %>%
  mutate(mt_oppPoints = t_oppPoints/t_Matches_total) %>%
  mutate(mt_oppGS = t_oppGS/t_Matches_total) %>%
  mutate(mt_oppGC = t_oppGC/t_Matches_total) %>%
  mutate(mt_oppGS1H = t_oppGS1H/t_Matches_total) %>%
  mutate(mt_oppGC1H = t_oppGC1H/t_Matches_total) %>%
  mutate(mt_oppGS2H = t_oppGS2H/t_Matches_total) %>%
  mutate(mt_oppGC2H = t_oppGC2H/t_Matches_total) %>%
  mutate(mt_Points_where = t_Points/t_Matches_where) %>%
  mutate(mt_GS_where = t_GS_where/t_Matches_where) %>%
  mutate(mt_GC_where = t_GC_where/t_Matches_where) %>%
  mutate(mt_GS1H_where = t_GS1H_where/t_Matches_where) %>%
  mutate(mt_GC1H_where = t_GC1H_where/t_Matches_where) %>%
  mutate(mt_GS2H_where = t_GS2H_where/t_Matches_where) %>%
  mutate(mt_GC2H_where = t_GC2H_where/t_Matches_where) %>%
  mutate(mt_oppPoints_where = t_oppPoints/t_oppMatches_where) %>%
  mutate(mt_oppGS_where = t_oppGS_where/t_oppMatches_where) %>%
  mutate(mt_oppGC_where = t_oppGC_where/t_oppMatches_where) %>%
  mutate(mt_oppGS1H_where = t_oppGS1H_where/t_oppMatches_where) %>%
  mutate(mt_oppGC1H_where = t_oppGC1H_where/t_oppMatches_where) %>%
  mutate(mt_oppGS2H_where = t_oppGS2H_where/t_oppMatches_where) %>%
  mutate(mt_oppGC2H_where = t_oppGC2H_where/t_oppMatches_where) 

#teamresults %>%
#  arrange(season, team, where, round) %>% 
#  select (team, oppTeam, round, season, where, GS, GC, t_Points:t_oppMatches_where)
  
teamresults<-
  teamresults %>%
  group_by(season, round) %>%
  mutate(t_Rank = rank(-t_Points, ties.method="min"),
         t_oppRank = rank(-t_oppPoints, ties.method="min")) %>% 
  arrange(season, round, team) %>%
  ungroup() %>%  
  mutate(t_diffRank = -t_Rank+t_oppRank,
         t_diffPoints = t_Points-t_oppPoints,
         t_diffGoals = t_GS - t_GC,
         t_diffGoals1H = t_GS1H - t_GC1H,
         t_diffGoals2H = t_GS2H - t_GC2H,
         t_diffOppGoals = t_oppGS - t_oppGC,
         t_diffOppGoals1H = t_oppGS1H - t_oppGC1H,
         t_diffOppGoals2H = t_oppGS2H - t_oppGC2H,
         t_diffGoals_where = t_GS_where - t_GC_where,
         t_diffGoals1H_where = t_GS1H_where - t_GC1H_where,
         t_diffGoals2H_where = t_GS2H_where - t_GC2H_where,
         t_diffOppGoals_where = t_oppGS_where - t_oppGC_where,
         t_diffOppGoals1H_where = t_oppGS1H_where - t_oppGC1H_where,
         t_diffOppGoals2H_where = t_oppGS2H_where - t_oppGC2H_where,
         t_diffBothGoals = t_diffGoals - t_diffOppGoals,
         t_diffBothGoals1H = t_diffGoals1H - t_diffOppGoals1H,
         t_diffBothGoals2H = t_diffGoals2H - t_diffOppGoals2H,
         t_diffBothGoals_where = t_diffGoals_where - t_diffOppGoals_where,
         t_diffBothGoals1H_where = t_diffGoals1H_where - t_diffOppGoals1H_where,
         t_diffBothGoals2H_where = t_diffGoals2H_where - t_diffOppGoals2H_where
  )

lag5<-function(x, round) {
  return(ifelse(round>5, x-lag(x, 5), x))
}

# enrich with last 5 games
teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) %>%
  mutate(l5_Points = lag5(t_Points, round)) %>%  
  mutate(l5_diffGoals = lag5(t_diffGoals, round)) %>%  
  mutate(l5_GS = lag5(t_GS, round)) %>%  
  mutate(l5_GC = lag5(t_GC, round)) %>%  
  group_by(season, oppTeam) %>%
  arrange(round) %>%
  mutate(l5_oppPoints = lag5(t_oppPoints, round)) %>%  
  mutate(l5_oppGS = lag5(t_oppGS, round)) %>%  
  mutate(l5_oppGC = lag5(t_oppGC, round)) %>%  
  mutate(l5_diffOppGoals = lag5(t_diffOppGoals, round)) %>%  
  ungroup() %>%  
  mutate(l5_diffBothPoints = l5_Points-l5_oppPoints) %>%  
  mutate(l5_diffBothGoals = l5_diffGoals - l5_diffOppGoals) 


print.data.frame(tail(teamresults, 30))

x<-teamresults %>% select(-points, -oppPoints, -starts_with("GC"),-starts_with("GS"))
y<-teamresults %>% transmute(y=GS-GC)
y<-as.factor(y$y)
str(y)

#y<-as.factor(paste(outputs$FTHG,outputs$FTAG, sep=":"))
#y<-outputs$FTHG - outputs$FTAG
#model_rf <- cforest(GS-GC ~ . - GS -GC -GC1H -GC1H, data=teamresults, controls = cforest_unbiased())
model_rf <- cforest(y ~ ., data=data.frame(y,x), controls = cforest_unbiased())
#model_rf <- cforest(y ~ ., data=data.frame(y,x), controls = cforest_unbiased(ntree = 1500, mtry = 8))
model_rf
summary(model_rf)
sort(varimp(model_rf), decreasing = TRUE)

#model_rf$importance[order(model_rf$importance[,1], decreasing = TRUE), ]


xy<-data.frame(y, x)

predictFactor<-function(xy, model_rf) {
  
  pred<-predict(model_rf, newdata = xy)
  pred<-as.integer(as.character(pred))
  act<-as.integer(as.character(xy$y))
  print(summary(pred))
  hist(pred)
  print(table(pred))
  print(table(pred, act))
  #plot(pred, act)
  #qqplot(pred, act)
  #densityPlot(pred, act)

  print(table(pred - act))
  print(table(sign(pred), sign(act)))
  print(table(sign(pred) - sign(act)))
  print(data.frame(cov=cov(pred, act), 
                   cor=cor(pred, act), 
                   hit_diff = sum(pred == act)/length(act)*100, 
                   hit_tendency = sum(sign(pred) == sign(act))/length(act)*100) )
}

predictFactor(xy, model_rf)
predictFactor(xy %>% filter(where=="Home"), model_rf)
predictFactor(xy %>% filter(where=="Away"), model_rf)
predictFactor(xy %>% filter(round<=17), model_rf)
predictFactor(xy %>% filter(round>17), model_rf)
predictFactor(xy %>% filter(t_Rank>=12), model_rf)
predictFactor(xy %>% filter(dow=="Friday"), model_rf)
predictFactor(xy %>% filter(dow=="Sunday"), model_rf)
predictFactor(xy %>% filter(dow=="Wednesday"), model_rf)
predictFactor(xy %>% filter(dow=="Tuesday"), model_rf)
predictFactor(xy %>% filter(team=="Bayern Munich"), model_rf)
predictFactor(xy %>% filter(team=="Dortmund"), model_rf)
predictFactor(xy %>% filter(team=="Leverkusen"), model_rf)
predictFactor(xy %>% filter(team=="Schalke 04"), model_rf)
predictFactor(xy %>% filter(team=="Hamburg"), model_rf)
predictFactor(xy %>% filter(team=="Wolfsburg"), model_rf)
predictFactor(xy %>% filter(team=="Darmstadt"), model_rf)


train<-xy %>% filter(season %in% c(2014,2015) | round <= 17)

model_cv <- cforest(y ~ ., data=train, controls = cforest_unbiased())
#model_rf <- cforest(y ~ ., data=data.frame(y,x), controls = cforest_unbiased(ntree = 1500, mtry = 8))
model_cv
summary(model_cv)
sort(varimp(model_cv), decreasing = TRUE)


test<-xy%>% filter(season == 2016 & round > 17)

nrow(train)
nrow(test)
nrow(xy)

predictFactor(test, model_cv)



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



predictGoalsNumeric<-function(xy, model) {
  
  pred<-predict(model, newdata = xy)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  print(summary(gspred))
  print(summary(gcpred))
  print(summary(gspred-gcpred))
  hist(gspred)
  hist(gcpred)
  hist(gspred-gcpred)
  plot(gspred, xy$GS)
  plot(gcpred, xy$GC)
  plot(gspred-gcpred, xy$GS - xy$GC)
  print(data.frame(covGS=cov(gspred, xy$GS), covGC=cov(gcpred, xy$GC), covDiff=cov(gspred-gcpred, xy$GS-xy$GC),
                   corGS=cor(gspred, xy$GS), corGC=cor(gcpred, xy$GC), corDiff=cor(gspred-gcpred, xy$GS-xy$GC)))
  print(table(round(gspred), xy$GS))
  print(table(round(gcpred), xy$GC))
  pred_diff1<-round(gspred)-round(gcpred)
  pred_diff2<-round(gspred-gcpred)
  act_diff<-xy$GS-xy$GC
  pred_tend1<-sign(pred_diff1)
  pred_tend2<-sign(pred_diff2)
  act_tend<-sign(act_diff)
  
  print(table(pred_diff1))
  print(table(pred_diff1, act_diff))
  print(table(pred_diff2))
  print(table(pred_diff2, act_diff))
  print(table(pred_tend1, act_tend))
  print(table(pred_tend2, act_tend))
  print(table(pred_tend2, pred_tend1))

  print(data.frame(hit_diff1 = sum(pred_diff1 == act_diff)/length(act_diff)*100, 
                   hit_tendency1 = sum(pred_tend1 == act_tend)/length(act_tend)*100,
                  hit_diff2 = sum(pred_diff2 == act_diff)/length(act_diff)*100, 
                  hit_tendency2 = sum(pred_tend2 == act_tend)/length(act_tend)*100,
                  exact_match = sum(round(gspred)==xy$GS & round(gcpred)==xy$GC)/length(act_diff)*100))

}  

predictGoalsNumeric2<-function(xy, model, model2) {
  
  pred<-predict(model, newdata = xy)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])

  data2<-data.frame(gspred, gcpred, xy)
  
  pred2<-predict(model2, newdata = data2)
  gspred<-sapply(pred2, function(x) x[,'GS'])
  gcpred<-sapply(pred2, function(x) x[,'GC'])
  
  print(summary(gspred))
  print(summary(gcpred))
  print(summary(gspred-gcpred))
  hist(gspred)
  hist(gcpred)
  hist(gspred-gcpred)
  plot(gspred, xy$GS)
  plot(gcpred, xy$GC)
  plot(gspred-gcpred, xy$GS - xy$GC)
  print(data.frame(covGS=cov(gspred, xy$GS), covGC=cov(gcpred, xy$GC), covDiff=cov(gspred-gcpred, xy$GS-xy$GC),
                   corGS=cor(gspred, xy$GS), corGC=cor(gcpred, xy$GC), corDiff=cor(gspred-gcpred, xy$GS-xy$GC)))
  print(table(round(gspred), xy$GS))
  print(table(round(gcpred), xy$GC))
  pred_diff1<-round(gspred)-round(gcpred)
  pred_diff2<-round(gspred-gcpred)
  act_diff<-xy$GS-xy$GC
  pred_tend1<-sign(pred_diff1)
  pred_tend2<-sign(pred_diff2)
  act_tend<-sign(act_diff)
  
  print(table(pred_diff1))
  print(table(pred_diff1, act_diff))
  print(table(pred_diff2))
  print(table(pred_diff2, act_diff))
  print(table(pred_tend1, act_tend))
  print(table(pred_tend2, act_tend))
  print(table(pred_tend2, pred_tend1))
  
  print(data.frame(hit_diff1 = sum(pred_diff1 == act_diff)/length(act_diff)*100, 
                   hit_tendency1 = sum(pred_tend1 == act_tend)/length(act_tend)*100,
                   hit_diff2 = sum(pred_diff2 == act_diff)/length(act_diff)*100, 
                   hit_tendency2 = sum(pred_tend2 == act_tend)/length(act_tend)*100,
                   exact_match = sum(round(gspred)==xy$GS & round(gcpred)==xy$GC)/length(act_diff)*100))
  
}  

predictGoalsNumeric3<-function(xy, model, model3) {
  
  pred<-predict(model, newdata = xy)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  
  data3<-data.frame(gspred, gcpred, xy)
  
  pred3<-predict(model3, newdata = data3 )
  gspred<-sapply(pred3, function(x) which.max(x[1,1:8])-1)
  gcpred<-sapply(pred3, function(x) which.max(x[1,9:16])-1)
  gspred<-as.integer(as.character(gspred))
  gcpred<-as.integer(as.character(gcpred))
  print(summary(gspred))
  print(summary(gcpred))
  print(summary(gspred-gcpred))
  hist(gspred)
  hist(gcpred)
  hist(gspred-gcpred)
  plot(gspred, xy$GS)
  plot(gcpred, xy$GC)
  plot(gspred-gcpred, xy$GS - xy$GC)
  print(data.frame(covGS=cov(gspred, xy$GS), covGC=cov(gcpred, xy$GC), covDiff=cov(gspred-gcpred, xy$GS-xy$GC),
                   corGS=cor(gspred, xy$GS), corGC=cor(gcpred, xy$GC), corDiff=cor(gspred-gcpred, xy$GS-xy$GC)))
  print(table(round(gspred), xy$GS))
  print(table(round(gcpred), xy$GC))
  pred_diff1<-round(gspred)-round(gcpred)
  act_diff<-xy$GS-xy$GC
  pred_tend1<-sign(pred_diff1)
  act_tend<-sign(act_diff)
  
  print(table(pred_diff1))
  print(table(pred_diff1, act_diff))
  print(table(pred_tend1, act_tend))

  print(table(gspred, gcpred))
  predScore<-data.frame(score=paste(gspred, gcpred, sep = ":"), 
                        fullhit=(gspred==xy$GS & gcpred==xy$GC), 
                        gdhit=(gspred-gcpred == xy$GS-xy$GC), 
                        tendhit=sign(gspred-gcpred) == sign(xy$GS-xy$GC))
  predScore$bidres="None"
  predScore=within(predScore,{
    bidres[tendhit]="Tendency"
    bidres[gdhit]="GoalDiff"
    bidres[fullhit]="Full"
  })
  predScore$bidres<-factor(predScore$bidres, levels=c("None", "Tendency", "GoalDiff", "Full"), ordered=TRUE)
  predScore$score<-factor(predScore$score, levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                                                    "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                                                    "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE)
  scoreTable<-table(predScore$score, predScore$bidres)
  
  print(scoreTable)
  print(scoreTable/rowSums(scoreTable)*100)
  print(colSums(scoreTable)/sum(scoreTable)*100)
  
  plot(predScore$score, predScore$bidres)
  
  print(data.frame(hit_diff1 = sum(pred_diff1 == act_diff)/length(act_diff)*100, 
                   hit_tendency1 = sum(pred_tend1 == act_tend)/length(act_tend)*100,
                   exact_match = sum(round(gspred)==xy$GS & round(gcpred)==xy$GC)/length(act_diff)*100))

  return(predScore)
}  

predictGoalsNumeric4<-function(xy, model, model4) {
  
  pred<-predict(model, newdata = xy)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  
  data4<-data.frame(gspred, gcpred, xy)
  
  pred4<-predict(model4, newdata = data4 )
  gspred<-as.integer(substr(pred4,1,1))
  gcpred<-as.integer(substr(pred4,3,3))
  print(summary(gspred))
  print(summary(gcpred))
  print(summary(gspred-gcpred))
  hist(gspred)
  hist(gcpred)
  hist(gspred-gcpred)
  plot(gspred, xy$GS)
  plot(gcpred, xy$GC)
  plot(gspred-gcpred, xy$GS - xy$GC)
  print(data.frame(covGS=cov(gspred, xy$GS), covGC=cov(gcpred, xy$GC), covDiff=cov(gspred-gcpred, xy$GS-xy$GC),
                   corGS=cor(gspred, xy$GS), corGC=cor(gcpred, xy$GC), corDiff=cor(gspred-gcpred, xy$GS-xy$GC)))
  print(table(round(gspred), xy$GS))
  print(table(round(gcpred), xy$GC))
  pred_diff1<-round(gspred)-round(gcpred)
  act_diff<-xy$GS-xy$GC
  pred_tend1<-sign(pred_diff1)
  act_tend<-sign(act_diff)
  
  print(table(pred_diff1))
  print(table(pred_diff1, act_diff))
  print(table(pred_tend1, act_tend))
  
  print(table(gspred, gcpred))
  predScore<-data.frame(score=paste(gspred, gcpred, sep = ":"), 
                        fullhit=(gspred==xy$GS & gcpred==xy$GC), 
                        gdhit=(gspred-gcpred == xy$GS-xy$GC), 
                        tendhit=sign(gspred-gcpred) == sign(xy$GS-xy$GC),
                        where=xy$where,
                        xy$GS, xy$GC)
  predScore$bidres="None"
  predScore=within(predScore,{
    bidres[tendhit]="Tendency"
    bidres[gdhit]="GoalDiff"
    bidres[fullhit]="Full"
  })
  predScore$bidres<-factor(predScore$bidres, levels=c("None", "Tendency", "GoalDiff", "Full"), ordered=TRUE)
  predScore$score<-factor(predScore$score, levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                                                    "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                                                    "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE)
  
  predScore$bidpoints=0
  predScore=within(predScore,{
    bidpoints[tendhit & where=='Home' & xy$GS>xy$GC]=2
    bidpoints[tendhit & where=='Home' & xy$GS<xy$GC]=4
    bidpoints[gdhit & where=='Home' & xy$GS>xy$GC]=3
    bidpoints[gdhit & where=='Home' & xy$GS==xy$GC]=2
    bidpoints[gdhit & where=='Home' & xy$GS<xy$GC]=5
    bidpoints[fullhit & where=='Home' & xy$GS>xy$GC]=4
    bidpoints[fullhit & where=='Home' & xy$GS==xy$GC]=6
    bidpoints[fullhit & where=='Home' & xy$GS<xy$GC]=7
    bidpoints[tendhit & where=='Away' & xy$GS>xy$GC]=4
    bidpoints[tendhit & where=='Away' & xy$GS<xy$GC]=2
    bidpoints[gdhit & where=='Away' & xy$GS>xy$GC]=5
    bidpoints[gdhit & where=='Away' & xy$GS==xy$GC]=2
    bidpoints[gdhit & where=='Away' & xy$GS<xy$GC]=3
    bidpoints[fullhit & where=='Away' & xy$GS>xy$GC]=7
    bidpoints[fullhit & where=='Away' & xy$GS==xy$GC]=6
    bidpoints[fullhit & where=='Away' & xy$GS<xy$GC]=4
  })
  scoreTable<-table(predScore$score, predScore$bidres)
  
  print(scoreTable)
  print(scoreTable/rowSums(scoreTable)*100)
  print(colSums(scoreTable)/sum(scoreTable)*100)
  
  plot(predScore$score, predScore$bidres)
  
  print(data.frame(hit_diff1 = sum(pred_diff1 == act_diff)/length(act_diff)*100, 
                   hit_tendency1 = sum(pred_tend1 == act_tend)/length(act_tend)*100,
                   exact_match = sum(round(gspred)==xy$GS & round(gcpred)==xy$GC)/length(act_diff)*100,
                  average_points = mean(predScore$bidpoints), total_points = sum(predScore$bidpoints)/2))
  
  return(predScore)
}  

predictGoalsNumeric5<-function(xy, model, model4, mask) {
  
  xy<-xy[xy$where=="Home",]
  pred<-predict(model, newdata = xy)
  gspred<-sapply(pred, function(x) x[,'GS'])
  gcpred<-sapply(pred, function(x) x[,'GC'])
  
  data4<-data.frame(gspred, gcpred, xy)
  
  predProbsList<-predict(model4, newdata = data4 , type = "prob")
  predoutcomes<-t(sapply(predProbsList, cbind))
  colnames(predoutcomes)<-colnames(predProbsList[[1]])
  predoutcomes<-predoutcomes[,order(colnames(predoutcomes))]  
  
  mask<-mask[order(names(mask))]
  expectedValues<-sapply(1:25, function(i) predoutcomes %*% as.vector(t(mask[i][[1]])))
  colnames(expectedValues)<-names(mask)
  
  maxEV<-apply(expectedValues, 1, max)
  whichMaxEV<-apply(expectedValues, 1, which.max)
  
  pred5<-colnames(expectedValues)[whichMaxEV]
  
  gspred<-as.integer(substr(pred5,1,1))
  gcpred<-as.integer(substr(pred5,3,3))
  #print(summary(gspred))
  #print(summary(gcpred))
  #print(summary(gspred-gcpred))
  #hist(gspred)
  #hist(gcpred)
  #hist(gspred-gcpred)
  #plot(gspred, xy$GS)
  #plot(gcpred, xy$GC)
  #plot(gspred-gcpred, xy$GS - xy$GC)
  print(data.frame(covGS=cov(gspred, xy$GS), covGC=cov(gcpred, xy$GC), covDiff=cov(gspred-gcpred, xy$GS-xy$GC),
                   corGS=cor(gspred, xy$GS), corGC=cor(gcpred, xy$GC), corDiff=cor(gspred-gcpred, xy$GS-xy$GC)))
#  print(table(round(gspred), xy$GS))
#  print(table(round(gcpred), xy$GC))
  pred_diff1<-round(gspred)-round(gcpred)
  act_diff<-xy$GS-xy$GC
  pred_tend1<-sign(pred_diff1)
  act_tend<-sign(act_diff)
  
  print(table(pred_diff1))
  print(table(pred_diff1, act_diff))
  print(table(pred_tend1, act_tend))
  
  print(table(gspred, gcpred))
  predScore<-data.frame(score=paste(gspred, gcpred, sep = ":"), 
                        fullhit=(gspred==xy$GS & gcpred==xy$GC), 
                        gdhit=(gspred-gcpred == xy$GS-xy$GC), 
                        tendhit=sign(gspred-gcpred) == sign(xy$GS-xy$GC),
                        where=xy$where,
                        xy$GS, xy$GC)
  predScore$bidres="None"
  predScore=within(predScore,{
    bidres[tendhit]="Tendency"
    bidres[gdhit]="GoalDiff"
    bidres[fullhit]="Full"
  })
  predScore$bidres<-factor(predScore$bidres, levels=c("None", "Tendency", "GoalDiff", "Full"), ordered=TRUE)
  predScore$score<-factor(predScore$score, levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                                                    "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                                                    "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE)
  
  predScore$bidpoints=0
  predScore=within(predScore,{
    bidpoints[tendhit & where=='Home' & xy$GS>xy$GC]=2
    bidpoints[tendhit & where=='Home' & xy$GS<xy$GC]=4
    bidpoints[gdhit & where=='Home' & xy$GS>xy$GC]=3
    bidpoints[gdhit & where=='Home' & xy$GS==xy$GC]=2
    bidpoints[gdhit & where=='Home' & xy$GS<xy$GC]=5
    bidpoints[fullhit & where=='Home' & xy$GS>xy$GC]=4
    bidpoints[fullhit & where=='Home' & xy$GS==xy$GC]=6
    bidpoints[fullhit & where=='Home' & xy$GS<xy$GC]=7
    bidpoints[tendhit & where=='Away' & xy$GS>xy$GC]=4
    bidpoints[tendhit & where=='Away' & xy$GS<xy$GC]=2
    bidpoints[gdhit & where=='Away' & xy$GS>xy$GC]=5
    bidpoints[gdhit & where=='Away' & xy$GS==xy$GC]=2
    bidpoints[gdhit & where=='Away' & xy$GS<xy$GC]=3
    bidpoints[fullhit & where=='Away' & xy$GS>xy$GC]=7
    bidpoints[fullhit & where=='Away' & xy$GS==xy$GC]=6
    bidpoints[fullhit & where=='Away' & xy$GS<xy$GC]=4
  })
  scoreTable<-table(predScore$score, predScore$bidres)
  
  print(scoreTable)
  print(scoreTable/rowSums(scoreTable)*100)
  print(colSums(scoreTable)/sum(scoreTable)*100)
  
  plot(predScore$score, predScore$bidres)
  
  print(data.frame(hit_diff1 = sum(pred_diff1 == act_diff)/length(act_diff)*100, 
                   hit_tendency1 = sum(pred_tend1 == act_tend)/length(act_tend)*100,
                   exact_match = sum(round(gspred)==xy$GS & round(gcpred)==xy$GC)/length(act_diff)*100,
                   average_points = mean(predScore$bidpoints), total_points = sum(predScore$bidpoints)))
  
  return(predScore)
}  


x<-teamresults %>% select(-points, -oppPoints, -starts_with("GC"),-starts_with("GS"))
y<-teamresults %>% select(GS,GC)

xy<-data.frame(y, x)
mask<-buildMask()

# y<-teamresults %>% transmute(y=GS)

#y<-as.factor(y$y)
str(y)

#y<-as.factor(paste(outputs$FTHG,outputs$FTAG, sep=":"))
#y<-outputs$FTHG - outputs$FTAG
#model_rf <- cforest(GS-GC ~ . - GS -GC -GC1H -GC1H, data=teamresults, controls = cforest_unbiased())
model_rf_goals <- cforest(GS+GC ~ ., data=data.frame(y,x), controls = cforest_unbiased())
#model_rf <- cforest(y ~ ., data=data.frame(y,x), controls = cforest_unbiased(ntree = 1500, mtry = 8))
model_rf_goals
summary(model_rf_goals)
# sort(varimp(model_rf_goals), decreasing = TRUE)

predictGoalsNumeric(xy = xy, model = model_rf_goals)

train<-xy %>% filter(season %in% c(2015,2014) )

train<-train %>% select(-starts_with("l5"))
train<-train %>% select(-starts_with("mt"))

model_cv_goals <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased())
model_cv_goals2 <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree = 1500, mtry = 8))
#model_rf <- cforest(y ~ ., data=data.frame(y,x), controls = cforest_unbiased(ntree = 1500, mtry = 8))
model_cv_goals

train<-xy %>% filter(season %in% c(2015,2014) )
test<-xy%>% filter(season == 2016)

model_cv_goals <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased())
pred<-predict(model_cv_goals, newdata = train)
gspred<-sapply(pred, function(x) x[,'GS'])
gcpred<-sapply(pred, function(x) x[,'GC'])
data_raw<-data.frame(gspred, gcpred, train)
model2<-cforest(GS*GC ~ ., data=data_raw %>% select(GS, GC, gspred, gcpred), controls = cforest_unbiased(ntree=50, mtry=2))
data_factor<-data_raw %>%mutate(GS=as.factor(GS), GC=as.factor(GC))%>% select(GS, GC, gspred, gcpred, team, oppTeam, where)
model3<-cforest(GS*GC ~ ., data=data_factor, controls = cforest_unbiased(ntree=500, mtry=3))

data_multifactor<-data_raw %>% 
  mutate(eg1=ifelse(GS>4,GS-4,0), eg2=ifelse(GC>4,GC-4,0)) %>% 
  mutate(eg3=ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0), eg4=ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)) %>%
  mutate(score=factor(paste(eg3, eg4, sep = ":"), 
                       levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                                "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                                "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE))%>% 
  select(score, gspred, gcpred, team, oppTeam, where) # 


model4<-cforest(score ~ ., data=data_multifactor, controls = cforest_unbiased(ntree=500, mtry=3))

predictGoalsNumeric(xy = test, model = model_cv_goals)
predictGoalsNumeric2(xy = test, model = model_cv_goals, model2)
result3<-predictGoalsNumeric3(xy = test, model = model_cv_goals, model3)
result4<-predictGoalsNumeric4(xy = test, model = model_cv_goals, model4)
result5<-predictGoalsNumeric5(xy = test, model = model_cv_goals, model4, mask)


hist(test[!result4$tendhit,]$round )
plot(sort(test[!result4$tendhit,]$round ))





predictGoalsNumeric(xy = test, model = model_cv_goals2)
predictGoalsNumeric(xy = train, model = model_cv_goals2)

predictGoalsNumeric(xy = test %>% filter(dow=="Saturday"), model = model_cv_goals)
predictGoalsNumeric(xy = test %>% filter(dow!="Saturday"), model = model_cv_goals)
predictGoalsNumeric(xy = test %>% filter(team=="Bayern Munich"), model = model_cv_goals)
predictGoalsNumeric(xy = test %>% filter(team=="Schalke 04"), model = model_cv_goals)

#########################################################
cbind(
xy %>% filter(team=="Schalke 04" & season==2016) %>% arrange(round) %>% select (mt_GS)
,
xy %>% filter(team=="Bayern Munich" & season==2016) %>% arrange(round) %>% select (mt_GS)
)

test %>% filter(dow=="Saturday") %>% select (team) %>% table


                    
xy %>% select(-GC)
varimp(model_cv_goals)
importance(model_cv_goals)
varimpAUC(model_cv_goals)
library(caret)
model_varimp <- cforest(GS ~ ., data=xy %>% select(-GC), controls = cforest_unbiased())
varImp(model_varimp)
sort(varimp(model_varimp), decreasing = TRUE)





    pred<-predict(model, newdata = xy)
  pred<-as.integer(as.character(pred))
  act<-as.integer(as.character(xy$y))
  print(summary(pred))
  hist(pred)
  print(table(pred))
  print(table(pred, act))
  #plot(pred, act)
  #qqplot(pred, act)
  #densityPlot(pred, act)
  
  print(table(pred - act))
  print(table(sign(pred), sign(act)))
  print(table(sign(pred) - sign(act)))
  print(data.frame(cov=cov(pred, act), 
                   cor=cor(pred, act), 
                   hit_diff = sum(pred == act)/length(act)*100, 
                   hit_tendency = sum(sign(pred) == sign(act))/length(act)*100) )
}


xy<-data.frame(pred=as.integer(as.character(pred)), act=as.integer(as.character(act)), x)
xy %>% filter(where=="Away") %>% select (pred, act) %>% table
xy %>% filter(where=="Home") %>% select (pred, act) %>% table


xy %>% 
  filter(where=="Home") %>% 
  left_join(xy %>% filter(where=="Away"), by = c("season", "round", "gameindex")) %>%
  select(pred.home=pred.x, pred.away=pred.y, act.x, act.y) %>%
  mutate(predsum=pred.home+pred.away, actsum=act.x+act.y) %>% 
  filter(predsum != 0)


str(xy)

table(xy$pred)

densityPlot(pred, act)


nrow(x)
x
tail(x[1:801,])

train_idx<-1:801
cv_idx<-801:nrow(x)



model_rf <- randomForest(x = x[train_idx,], y=y[train_idx], importance=TRUE)
model_rf <- cforest(y ~ ., data=data.frame(y=y[train_idx],x[train_idx,]), controls = cforest_unbiased(ntree = 1500, mtry = 8))
x<-x[cv_idx,]
y<-y[cv_idx]
model_rf
summary(model_rf)
sort(varimp(model_rf), decreasing = TRUE)
#varImpPlot(model_rf)
data.frame(predict(model_rf, newdata = x), y)
cov(predict(model_rf, newdata = x), y)
cor(predict(model_rf, newdata = x), y)
plot(predict(model_rf, newdata = x), y)
qqplot(predict(model_rf, newdata = x), y)
table(round(predict(model_rf, newdata = x)), y)
table(round(predict(model_rf, newdata = x)) - y)
table(sign(round(predict(model_rf, newdata = x))), sign(y))
table(sign(round(predict(model_rf, newdata = x))) - sign(y))
table(sign(round(predict(model_rf, newdata = x))) == sign(y))

sum(sign(round(predict(model_rf, newdata = x))) == sign(y))/length(y)*100 


m.team<-glm(formula = goals ~ (team+otherTeam)+where+tablepoints+rank+season+round+pointdiff+rankdiff, data=teamresults, family = poisson)
summary(m.team)

ranking <- 
  teamresults %>% 
  group_by(season, round, team) %>% 
  select (tablepoints, rank, scoredgoals, receivedgoals) %>% 
  ungroup()

results <-
  results %>% 
  select(-HTR, -FTR) %>%
  rename(team=HomeTeam) %>% left_join(ranking, by = c("season", "round", "team")) %>% rename(HomeTeam=team, HRank=rank, HP=tablepoints, HSG=scoredgoals, HRG=receivedgoals)%>% 
  rename(team=AwayTeam) %>% left_join(ranking, by = c("season", "round", "team")) %>% rename(AwayTeam=team, ARank=rank, AP=tablepoints, ASG=scoredgoals, ARG=receivedgoals)%>%
  mutate(PointDiff=HP-AP, RankDiff=ARank-HRank,  FTR=sign(FTHG-FTAG), HTR=sign(HTHG-HTAG))


outputs <- results %>%
  select(FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR )

inputs <- results %>%
  select(HomeTeam, AwayTeam, season, round, dayofweek, HP, AP  , PointDiff, HRank, ARank ,  RankDiff, HSG, HRG, ASG, ARG)

x2<-scale(outputs, scale=FALSE, center = TRUE)
str(x2)
summary(x2)
sigma<-cov(x2)
sigma_inv<-solve(sigma)
sigma_inv_sqrt<-sqrtm(sigma_inv)
white<-t(sigma_inv_sqrt %*% t(x2))
cov(white)
colnames(white)<-colnames(outputs)



km<-kmeans(white, centers = 10)

km<-kmeans(x3, centers = 10)


outputs2 <- results %>%
  select(FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR )%>%
  mutate(FTHG2 = FTHG-HTHG, FTAG2 = FTAG-HTAG, FTGDiff=FTHG-FTAG, HT2GDiff=FTHG2-FTAG2, HTGDiff=HTHG-HTAG) 

factanal(x = outputs, factors=9)

factanal(x = white, factors=5)

cov(outputs2)


library(RANN)
x3<-x2
x3[,1]<-outputs[,1]*3
x3[,2]<-outputs[,2]*3
x3[,3]<-outputs[,3]*5
nn<-nn2(x3)
nn<-nn2(white)
summary(nn$nn.dists[, 2:10])

rbind(outputs[nn$nn.idx[,1],], outputs[nn$nn.idx[,2],], outputs[nn$nn.idx[,3],])[ c(0, nrow(outputs), 2*nrow(outputs))+rep(1:nrow(outputs), each=3), ]

matrix(ncol=10, data=outputs[nn$nn.idx[,1:10],], byrow = FALSE)

matrix(ncol=10, data=km$cluster[nn$nn.idx[,1:10]], byrow = FALSE)
km$cluster[nn$nn.idx[1,1:10]]
nn$nn.idx[2,1:10]


table(outputs$FTHG,outputs$FTAG, km$cluster)

sum(km$withinss)+km$betweenss
km
km$size

km$centers
summary(dist(km$centers))
boxplot(dist(km$centers))
hist(dist(km$centers))

shift <- attr(x2,"scaled:center")
t(t(x2)+shift)

unwhiten<-function(x) t(sigma %*% sigma_inv_sqrt %*% t(x)+shift)
unwhiten(white)

cmeans<-t(t(km$centers)+shift)
colnames(cmeans)<-names(shift)
cmeans
dist(cmeans)


unwhiten(km$centers)

outputs[km$cluster==10,]

neglinks <- matrix(1, ncol = length(km$size), nrow=length(km$size), byrow = TRUE)
diag(neglinks)<-0

library(dummies)
inputs_df <- dummy.data.frame(inputs, names=c("HomeTeam", "AwayTeam", "season", "dayofweek"), sep="_")

inputs_df2 <- dummy.data.frame(data = data.frame(Team=inputs[, "HomeTeam"]), names="Team", sep="_")
inputs_df3 <- dummy.data.frame(data = data.frame(Team=inputs[, "AwayTeam"]), names="Team", sep="_")

inputs_df4 <- dummy.data.frame( data.frame(inputs_df2-inputs_df3, inputs[,-c(1,2)]), names=c("season", "dayofweek"), sep="_")


str(inputs_df)

dcaData <- dca(data = inputs_df4, chunks = km$cluster, neglinks=neglinks)
dcaData <- dca(data = inputs_df, chunks = km$cluster, neglinks=neglinks)
str(dcaData)
dcaData$newData
dcamatrix<-dcaData$DCA
colnames(dcamatrix)<-colnames(inputs_df)
dcamatrix

distmatrix<-as.matrix(dist(dcaData$newData))
str(distmatrix)
hist(distmatrix, prob = T)

hist(1/distmatrix)
hist(exp(-distmatrix))
summary(exp(-distmatrix))

plot(sort(exp(-distmatrix[1,2:909])))

i<-302
plot(sort(exp(-distmatrix[i,-i]^4.5)))
s1<-aggregate(exp(-distmatrix[i,-i]^1.5), by=list(outputs$FTHG[-i], outputs$FTAG[-i]), sum)
s1$x<-s1$x/sum(s1$x)*100
s1
paste(outputs$FTHG[i], outputs$FTAG[i])


library(class)
knnclass <- knn(train = dcaData$newData, test = dcaData$newData, km$cluster, k = 10, l = 0, prob = TRUE, use.all = TRUE)
table(knnclass, km$cluster)

summary(attr(knnclass, "prob"))

library(RANN)
nn<-nn2(dcaData$newData)
summary(nn$nn.dists[,2:10])

matrix(ncol=10, data=km$cluster[nn$nn.idx[,1:10]], byrow = FALSE)
km$cluster[nn$nn.idx[1,1:10]]
nn$nn.idx[2,1:10]

str(knnclass)


## rf
library(randomForest)
library(party)
x<-inputs
x$dayofweek<-as.factor(x$dayofweek)
x$HPQ<-x$HP/x$round
x$APQ<-x$AP/x$round
x$HSGQ<-x$HSG/x$round
x$HRGQ<-x$HRG/x$round
x$ASGQ<-x$ASG/x$round
x$ARGQ<-x$ARG/x$round




str(inputs)
varimp(model_rf)

as.factor(paste(outputs$FTHG,outputs$FTAG, sep=":"))

table(outputs$FTHG,outputs$FTAG, km$cluster)


pred<-sapply(1:nrow(outputs), function(i) 
  aggregate(exp(-distmatrix[i,]^1.5), by=list(outputs$FTHG, outputs$FTAG), sum)$x
  , simplify = TRUE)
pred<-t(pred)
str(pred)

scores<-aggregate(exp(-distmatrix[1,]), by=list(outputs$FTHG, outputs$FTAG), sum)[,1:2]
ownscore<-t(sapply(1:nrow(outputs), function(i) outputs$FTHG[i]==scores[,1] & outputs$FTAG[i]==scores[,2]))
pred2<-pred-ownscore
pred3<-pred2/rowSums(pred2)
colnames(pred3)<-paste(scores[,1], scores[,2])
pred3

summary(pred3[ownscore])

scores_0<-table(outputs$FTHG, outputs$FTAG)/nrow(outputs)

data.frame(meanhitprob=mean(pred3[ownscore]), standardhitprob=sum(scores_0^2))

#####################

pred<-sapply(1:nrow(outputs), function(i) 
  aggregate(exp(-distmatrix[i,]^1.5), by=list(outputs$FTR), sum)$x
  , simplify = TRUE)
pred<-t(pred)
str(pred)

scores<-aggregate(exp(-distmatrix[1,]), by=list(outputs$FTR), sum)[,1:2]
ownscore<-t(sapply(1:nrow(outputs), function(i) outputs$FTR[i]==scores[,1]))
pred2<-pred-ownscore
pred3<-pred2/rowSums(pred2)
colnames(pred3)<-paste(scores[,1])
pred3

summary(pred3[ownscore])

scores_0<-table(outputs$FTR)/nrow(outputs)

data.frame(meanhitprob=mean(pred3[ownscore]), standardhitprob=sum(scores_0^2))

#####################

pred<-sapply(1:nrow(outputs), function(i) 
  aggregate(exp(-distmatrix[i,]^4.5), by=list(outputs$FTHG-outputs$FTAG), sum)$x
  , simplify = TRUE)
pred<-t(pred)
str(pred)

scores<-aggregate(exp(-distmatrix[1,]), by=list(outputs$FTHG-outputs$FTAG), sum)[,1:2]
ownscore<-t(sapply(1:nrow(outputs), function(i) outputs$FTHG[i]-outputs$FTAG[i]==scores[,1]))
pred2<-pred-ownscore
pred3<-pred2/rowSums(pred2)
colnames(pred3)<-paste(scores[,1])
pred3

summary(pred3[ownscore])

scores_0<-table(outputs$FTHG-outputs$FTAG)/nrow(outputs)

data.frame(meanhitprob=mean(pred3[ownscore]), standardhitprob=sum(scores_0^2))

#####################






paste(scores[,1], scores[,2])

pred[2,]
pred2[2,]
scores[ownscore[2,],]

  aggregate(exp(-distmatrix[i,]), by=list(outputs$FTHG, outputs$FTAG), sum)$x
  , simplify = TRUE)


pred2<-matrix(nrow=nrow(outputs), byrow = TRUE, data = unlist(pred))

sapply(pred, length)

str(pred[[2]])

unlist(pred, recursive = FALSE)

table(outputs$FTHG, outputs$FTAG)

table(paste(outputs$FTHG, outputs$FTAG))


dist(km$centers)
dist(unwhiten(km$centers)[,"FTR"])

str(df)

inputs

require(mclust)
mcl<-Mclust(scale((outputs), scale=FALSE))
summary(mcl)
plot(mcl)


mahalanobis()


require(graphics)

ma <- cbind(1:6, 1:3)
(S <-  var(ma))
mahalanobis(c(0, 0), 1:2, S)

x <- matrix(rnorm(100*3), ncol = 3)
stopifnot(mahalanobis(x, 0, diag(ncol(x))) == rowSums(x*x))
##- Here, D^2 = usual squared Euclidean distances

Sx <- cov(x)
D2 <- mahalanobis(x, colMeans(x), Sx)
plot(density(D2, bw = 0.5),
     main="Squared Mahalanobis distances, n=100, p=3") ; rug(D2)
qqplot(qchisq(ppoints(100), df = 3), D2,
       main = expression("Q-Q plot of Mahalanobis" * ~D^2 *
                           " vs. quantiles of" * ~ chi[3]^2))
abline(0, 1, col = 'gray')




outputs %>% mutate(cluster=km$cluster) %>% arrange(cluster) %>% 
  group_by(cluster) %>% 
  select(table(FTHG,FTAG))
  print.data.frame()


X <- matrix(nrow = 2, ncol=5, data=rnorm(10))
t(X)%*%X
plot(X)
X%*%t(X)
i<-1
j<-5
e <- matrix(nrow = 5, ncol=1, data=c(1,0,0,0,-1))
t(e) %*% t(X) %*%X %*% e
t(X[,1]-X[,5]) %*% (X[,1]-X[,5])

require (dml)

## Not run:
set.seed(123)
require(MASS) # generate synthetic Gaussian data
k = 100 # sample size of each class
n = 3 # specify how many class
N = k * n # total sample number
x1 = mvrnorm(k, mu = c(-2, 1), matrix(c(10, 4, 4, 10), ncol = 2))
x2 = mvrnorm(k, mu = c(0, 0), matrix(c(10, 4, 4, 10), ncol = 2))
x3 = mvrnorm(k, mu = c(2, -1), matrix(c(10, 4, 4, 10), ncol = 2))
data = as.data.frame(rbind(x1, x2, x3))
# The fully labeled data set with 3 classes
plot(data$V1, data$V2, bg = c("#E41A1C", "#377EB8", "#4DAF4A")[gl(n, k)],
     pch = c(rep(22, k), rep(21, k), rep(25, k)))
Sys.sleep(3)
# Same data unlabeled; clearly the classes' structure is less evident
plot(data$V1, data$V2)
Sys.sleep(3)
chunk1 = sample(1:100, 5)
chunk2 = sample(setdiff(1:100, chunk1), 5)
chunk3 = sample(101:200, 5)
chunk4 = sample(setdiff(101:200, chunk3), 5)
chunk5 = sample(201:300, 5)
chks = list(chunk1, chunk2, chunk3, chunk4, chunk5)
chunks = rep(-1, 300)
# positive samples in the chunks
for (i in 1:5) {
  for (j in chks[[i]]) {
    chunks[j] = i
  }
}
# define the negative constrains between chunks
neglinks = matrix(c(
  0, 0, 1, 1, 1,
  0, 0, 1, 1, 1,
  1, 1, 0, 0, 0,
  1, 1, 0, 0, 1,
  1, 1, 1, 1, 0),
  ncol = 5, byrow = TRUE)
# 4 GdmDiag
dcaData = dca(data = data, chunks = chunks, neglinks = neglinks)$newData
# plot DCA transformed data
plot(dcaData[, 1], dcaData[, 2], bg = c("#E41A1C", "#377EB8", "#4DAF4A")[gl(n, k)],
     pch = c(rep(22, k), rep(21, k), rep(25, k))
     )
#,     xlim = c(-15, 15), ylim = c(-15, 15))
## End(Not run)








## Not run:
set.seed(602)
library(MASS)
library(scatterplot3d)
# generate simulated Gaussian data
k = 100
m <- matrix(c(1, 0.5, 1, 0.5, 2, -1, 1, -1, 3), nrow =3, byrow = T)
x1 <- mvrnorm(k, mu = c(1, 1, 1), Sigma = m)
x2 <- mvrnorm(k, mu = c(-1, 0, 0), Sigma = m)
data <- rbind(x1, x2)
# define similar constrains
simi <- rbind(t(combn(1:k, 2)), t(combn((k+1):(2*k), 2)))
temp <- as.data.frame(t(simi))
tol <- as.data.frame(combn(1:(2*k), 2))
# define disimilar constrains
dism <- t(as.matrix(tol[!tol %in% simi]))
# transform data using GdmDiag
result <- GdmDiag(data, simi, dism)
newData <- result$newData
# plot original data
color <- gl(2, k, labels = c("red", "blue"))
par(mfrow = c(2, 1), mar = rep(0, 4) + 0.1)
scatterplot3d(data, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Original Data")
# plot GdmDiag transformed data
scatterplot3d(newData, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Transformed Data")
## End(Not run)



## Not run:
set.seed(123)
library(MASS)
library(scatterplot3d)
# generate simulated Gaussian data
k = 100
m <- matrix(c(1, 0.5, 1, 0.5, 2, -1, 1, -1, 3), nrow =3, byrow = T)
x1 <- mvrnorm(k, mu = c(1, 1, 1), Sigma = m)
x2 <- mvrnorm(k, mu = c(-1, 0, 0), Sigma = m)
data <- rbind(x1, x2)
# define similar constrains
simi <- rbind(t(combn(1:k, 2)), t(combn((k+1):(2*k), 2)))
temp <- as.data.frame(t(simi))
tol <- as.data.frame(combn(1:(2*k), 2))
# define disimilar constrains
dism <- t(as.matrix(tol[!tol %in% simi]))
# transform data using GdmFull
result <- GdmFull(data, simi, dism)
newData <- result$newData
# plot original data
color <- gl(2, k, labels = c("red", "blue"))
par(mfrow = c(2, 1), mar = rep(0, 4) + 0.1)
scatterplot3d(data, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Original Data")
# plot GdmFull transformed data
scatterplot3d(newData, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Transformed Data")
## End(Not run)



par(mfrow = c(1, 1))



set.seed(1234)
require(MASS) # generate synthetic Gaussian data
k = 100 # sample size of each class
n = 3 # specify how many class
N = k * n # total sample number
x1 = mvrnorm(k, mu = c(-10, 6), matrix(c(10, 4, 4, 10), ncol = 2))
x2 = mvrnorm(k, mu = c(0, 0), matrix(c(10, 4, 4, 10), ncol = 2))
x3 = mvrnorm(k, mu = c(10, -6), matrix(c(10, 4, 4, 10), ncol = 2))
x = as.data.frame(rbind(x1, x2, x3))
x$V3 = gl(n, k)
# The fully labeled data set with 3 classes
plot(x$V1, x$V2, bg = c("#E41A1C", "#377EB8", "#4DAF4A")[x$V3],
     pch = c(rep(22, k), rep(21, k), rep(25, k)))
Sys.sleep(3)
# Same data unlabeled; clearly the classes' structure is less evident
plot(x$V1, x$V2)
Sys.sleep(3)
chunk1 = sample(1:100, 5)
chunk2 = sample(setdiff(1:100, chunk1), 5)
chunk3 = sample(101:200, 5)
chunk4 = sample(setdiff(101:200, chunk3), 5)
chunk5 = sample(201:300, 5)
chks = x[c(chunk1, chunk2, chunk3, chunk4, chunk5), ]
chunks = list(chunk1, chunk2, chunk3, chunk4, chunk5)
# The chunklets provided to the RCA algorithm
plot(chks$V1, chks$V2, col = rep(c("#E41A1C", "#377EB8",
                                   "#4DAF4A", "#984EA3", "#FF7F00"), each = 5),
     pch = rep(0:4, each = 5), ylim = c(-15, 15))
Sys.sleep(3)
# Whitening transformation applied to the chunklets
chkTransformed = as.matrix(chks[ , 1:2]) %*% rca(x[ , 1:2], chunks)$A
plot(chkTransformed[ , 1], chkTransformed[ , 2], col = rep(c(
  "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"), each = 5),
  pch = rep(0:4, each = 5), ylim = c(-15, 15))
Sys.sleep(3)
# The origin data after applying the RCA transformation
plot(rca(x[ , 1:2], chunks)$newX[, 1], rca(x[ , 1:2], chunks)$newX[, 2],
     bg = c("#E41A1C", "#377EB8", "#4DAF4A")[gl(n, k)],
     pch = c(rep(22, k), rep(21, k), rep(25, k)))
# The RCA suggested transformation of the data, dimensionality reduced
rca(x[ , 1:2], chunks)$A
# The RCA suggested Mahalanobis matrix
rca(x[ , 1:2], chunks)$B
## End(Not run)




