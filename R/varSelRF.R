setwd("~/LearningR/Bundesliga")

download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
download.file("http://www.football-data.co.uk/mmz4281/1415/D1.csv", "BL2014.csv")

#install.packages("dplyr")
#install.packages("party")
##install.packages("rattle",  dependencies=c("Depends", "Suggests"))
install.packages("varSelRF")
install.packages("FSelector")
install.packages("RWekajars")
install.packages("infotheo")

library(dplyr)
#require(expm)
#library(car)
library(party)
#library(rattle)
# rattle()
library(varSelRF)
library(FSelector)
library(infotheo)

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
                          dow=results$dayofweek, gameindex=results$gameindex,
                          Shots=results$HS, Shotstarget=results$HST, Fouls=results$HF, Corners=results$HC, Yellow=results$HY, Red=results$HR)
teamresults <- rbind(data.frame(team=results$AwayTeam, oppTeam=results$HomeTeam, 
                                GS=results$FTAG, GC=results$FTHG, where="Away", round=results$round, season=results$season,
                                GS1H=results$HTAG, GC1H=results$HTHG, GS2H=results$FTAG-results$HTAG, GC2H=results$FTHG-results$HTHG, 
                                dow=results$dayofweek, gameindex=results$gameindex,
                                Shots=results$AS, Shotstarget=results$AST, Fouls=results$AF, Corners=results$AC, Yellow=results$AY, Red=results$AR),
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
  mutate(t_Shots = cumsum(Shots)-Shots) %>%
  mutate(t_Shotstarget = cumsum(Shotstarget)-Shotstarget) %>%
  mutate(t_Fouls = cumsum(Fouls)-Fouls) %>%
  mutate(t_Corners = cumsum(Corners)-Corners) %>%
  mutate(t_Yellow = cumsum(Yellow)-Yellow) %>%
  mutate(t_Red = cumsum(Red)-Red) %>%
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
  mutate(t_Shots_where = cumsum(Shots)-Shots) %>%
  mutate(t_Shotstarget_where = cumsum(Shotstarget)-Shotstarget) %>%
  mutate(t_Fouls_where = cumsum(Fouls)-Fouls) %>%
  mutate(t_Corners_where = cumsum(Corners)-Corners) %>%
  mutate(t_Yellow_where = cumsum(Yellow)-Yellow) %>%
  mutate(t_Red_where = cumsum(Red)-Red) %>%
  group_by(season, oppTeam) %>%
  arrange(round) %>%
  mutate(t_oppPoints = cumsum(oppPoints)-oppPoints) %>% 
  mutate(t_oppGS = cumsum(GC)-GC) %>%
  mutate(t_oppGC = cumsum(GS)-GS) %>%
  mutate(t_oppGS1H = cumsum(GC1H)-GC1H) %>%
  mutate(t_oppGC1H = cumsum(GS1H)-GS1H) %>%
  mutate(t_oppGS2H = cumsum(GC2H)-GC2H) %>%
  mutate(t_oppGC2H = cumsum(GS2H)-GS2H) %>%
  mutate(t_oppShots = cumsum(Shots)-Shots) %>%
  mutate(t_oppShotstarget = cumsum(Shotstarget)-Shotstarget) %>%
  mutate(t_oppFouls = cumsum(Fouls)-Fouls) %>%
  mutate(t_oppCorners = cumsum(Corners)-Corners) %>%
  mutate(t_oppYellow = cumsum(Yellow)-Yellow) %>%
  mutate(t_oppRed = cumsum(Red)-Red) %>%
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
  mutate(t_oppShots_where = cumsum(Shots)-Shots) %>%
  mutate(t_oppShotstarget_where = cumsum(Shotstarget)-Shotstarget) %>%
  mutate(t_oppFouls_where = cumsum(Fouls)-Fouls) %>%
  mutate(t_oppCorners_where = cumsum(Corners)-Corners) %>%
  mutate(t_oppYellow_where = cumsum(Yellow)-Yellow) %>%
  mutate(t_oppRed_where = cumsum(Red)-Red) %>%
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
  mutate(mt_Shots = t_Shots/t_Matches_total) %>%
  mutate(mt_Shotstarget = t_Shotstarget/t_Matches_total) %>%
  mutate(mt_goal_efficiency = t_GS/t_Shotstarget) %>%
  mutate(mt_Shot_efficiency = t_Shotstarget/t_Shots) %>%
  mutate(mt_Fouls = t_Fouls/t_Matches_total) %>%
  mutate(mt_Corners = t_Corners/t_Matches_total) %>%
  mutate(mt_Yellow = t_Yellow/t_Matches_total) %>%
  mutate(mt_Red = t_Red/t_Matches_total) %>%
  mutate(mt_oppPoints = t_oppPoints/t_Matches_total) %>%
  mutate(mt_oppGS = t_oppGS/t_Matches_total) %>%
  mutate(mt_oppGC = t_oppGC/t_Matches_total) %>%
  mutate(mt_oppGS1H = t_oppGS1H/t_Matches_total) %>%
  mutate(mt_oppGC1H = t_oppGC1H/t_Matches_total) %>%
  mutate(mt_oppGS2H = t_oppGS2H/t_Matches_total) %>%
  mutate(mt_oppGC2H = t_oppGC2H/t_Matches_total) %>%
  mutate(mt_oppShots = t_oppShots/t_Matches_total) %>%
  mutate(mt_oppShotstarget = t_oppShotstarget/t_Matches_total) %>%
  mutate(mt_oppgoal_efficiency = t_GC/t_Shotstarget) %>%
  mutate(mt_oppShot_efficiency = t_oppShotstarget/t_Shots) %>%
  mutate(mt_oppFouls = t_oppFouls/t_Matches_total) %>%
  mutate(mt_oppCorners = t_oppCorners/t_Matches_total) %>%
  mutate(mt_oppYellow = t_oppYellow/t_Matches_total) %>%
  mutate(mt_oppRed = t_oppRed/t_Matches_total) %>%
  mutate(mt_Points_where = t_Points/t_Matches_where) %>%
  mutate(mt_GS_where = t_GS_where/t_Matches_where) %>%
  mutate(mt_GC_where = t_GC_where/t_Matches_where) %>%
  mutate(mt_GS1H_where = t_GS1H_where/t_Matches_where) %>%
  mutate(mt_GC1H_where = t_GC1H_where/t_Matches_where) %>%
  mutate(mt_GS2H_where = t_GS2H_where/t_Matches_where) %>%
  mutate(mt_GC2H_where = t_GC2H_where/t_Matches_where) %>%
  mutate(mt_Shots_where = t_Shots/t_Matches_total) %>%
  mutate(mt_Shotstarget_where = t_Shotstarget_where/t_Matches_where) %>%
  mutate(mt_goal_efficiency_where = t_GS_where/t_Shotstarget_where) %>%
  mutate(mt_Shot_efficiency_where = t_Shotstarget_where/t_Shots_where) %>%
  mutate(mt_Fouls_where = t_Fouls_where/t_Matches_where) %>%
  mutate(mt_Corners_where = t_Corners_where/t_Matches_where) %>%
  mutate(mt_Yellow_where = t_Yellow_where/t_Matches_where) %>%
  mutate(mt_Red_where = t_Red_where/t_Matches_where) %>%
  mutate(mt_oppPoints_where = t_oppPoints/t_oppMatches_where) %>%
  mutate(mt_oppGS_where = t_oppGS_where/t_oppMatches_where) %>%
  mutate(mt_oppGC_where = t_oppGC_where/t_oppMatches_where) %>%
  mutate(mt_oppGS1H_where = t_oppGS1H_where/t_oppMatches_where) %>%
  mutate(mt_oppGC1H_where = t_oppGC1H_where/t_oppMatches_where) %>%
  mutate(mt_oppGS2H_where = t_oppGS2H_where/t_oppMatches_where) %>%
  mutate(mt_oppGC2H_where = t_oppGC2H_where/t_oppMatches_where)  %>%
  mutate(mt_oppShots_where = t_oppShots_where/t_oppMatches_where) %>%
  mutate(mt_oppShotstarget_where = t_oppShotstarget_where/t_oppMatches_where) %>%
  mutate(mt_oppgoal_efficiency_where = t_GC_where/t_Shotstarget) %>%
  mutate(mt_oppShot_efficiency_where = t_oppShotstarget_where/t_Shots) %>%
  mutate(mt_oppFouls_where = t_oppFouls_where/t_oppMatches_where) %>%
  mutate(mt_oppCorners_where = t_oppCorners_where/t_oppMatches_where) %>%
  mutate(mt_oppYellow_where = t_oppYellow_where/t_oppMatches_where) %>%
  mutate(mt_oppRed_where = t_oppRed_where/t_oppMatches_where)

teamresults<-
  within(teamresults,{
  mt_goal_efficiency[is.na(mt_goal_efficiency)]<-0
  mt_Shot_efficiency[is.na(mt_Shot_efficiency)]<-0
  mt_oppgoal_efficiency[is.na(mt_oppgoal_efficiency)]<-0
  mt_oppShot_efficiency[is.na(mt_oppShot_efficiency)]<-0
  mt_goal_efficiency_where[is.na(mt_goal_efficiency_where)]<-0
  mt_Shot_efficiency_where[is.na(mt_Shot_efficiency_where)]<-0
  mt_oppgoal_efficiency_where[is.na(mt_oppgoal_efficiency_where)]<-0
  mt_oppShot_efficiency_where[is.na(mt_oppShot_efficiency_where)]<-0
}) 



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

summarizePrediction<-function(xy, gspred, gcpred) {
  
  pred_diff1<-gspred-gcpred
  act_diff<-xy$GS-xy$GC
  pred_tend1<-sign(pred_diff1)
  act_tend<-sign(act_diff)
  
  print(table(gspred, gcpred))
  print(table(pred_tend1, act_tend))
  print(table(pred_diff1))
  print(table(pred_diff1, act_diff))
  
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
  
  predScore$score[predScore$where=='Away']<-paste(gcpred, gspred, sep = ":")[predScore$where=='Away']
  
  scoreTable<-table(predScore$score, predScore$bidres)
  
  print(scoreTable)
  print(colSums(scoreTable)/sum(scoreTable)*100)
  print(scoreTable/rowSums(scoreTable)*100)
  
  plot(predScore$score[predScore$where=='Home'], predScore$bidres[predScore$where=='Home'], xlab="Home")
  plot(predScore$score[predScore$where=='Away'], predScore$bidres[predScore$where=='Away'], xlab="Away")
  plot(predScore$score, predScore$bidres, xlab="All")
  
  print(data.frame(hit_tendency1 = sum(pred_tend1 == act_tend)/length(act_tend)*100,
                   hit_diff1 = sum(pred_diff1 == act_diff)/length(act_diff)*100, 
                   exact_match = sum(gspred==xy$GS & gcpred==xy$GC)/length(act_diff)*100,
                   average_points = mean(predScore$bidpoints), 
                   total_points_home = sum(predScore$bidpoints[predScore$where=='Home']),
                   total_points_away = sum(predScore$bidpoints[predScore$where=='Away'])
                   ))
  
  return(predScore)
}  


mask<-buildMask()
x<-teamresults %>% select(-points, -oppPoints, -Shots, -Shotstarget, -Fouls, -Corners, -Yellow, -Red, -starts_with("GC"),-starts_with("GS"))
#x<-x %>% select(-starts_with("t_"))
y<-teamresults %>% select(GS,GC)
xy<-data.frame(y, x)

yWin<-as.factor(sign(xy$GS-xy$GC))
vs<-varSelRF(x, yWin)

is.na(x)


vs<-varSelRF(x, yWin)
selectedModelTendency<-vs$selected.model
selectedModelTendency<-"mt_oppCorners_where + mt_oppgoal_efficiency_where + mt_oppShots_where + mt_Shotstarget_where + oppTeam + t_diffBothGoals + t_diffBothGoals_where + t_diffBothGoals2H + t_diffBothGoals2H_where + t_diffGoals_where + t_diffOppGoals + t_diffOppGoals_where + team + where"

selectedVarsTendency<-vs$selected.vars


plot(vs)
plot(vs$selec.history)
dput(vs, file="varSelRF_tendency")

vs$firstForest$confusion
plot(vs$firstForest$err.rate[,1], ylab=colnames(vs$firstForest$err.rate)[1])
plot(vs$firstForest$err.rate[,2], ylab=colnames(vs$firstForest$err.rate)[2])
plot(vs$firstForest$err.rate[,3], ylab=colnames(vs$firstForest$err.rate)[3])
plot(vs$firstForest$err.rate[,4], ylab=colnames(vs$firstForest$err.rate)[4])


yDraw<-as.factor(as.integer(xy$GS==xy$GC))
vsDraw<-varSelRF(x, yDraw, verbose = TRUE, ntree=2000)

dput(vsDraw, file="varSelRF_draw")
plot(vsDraw)

selectedModelDraw<-vsDraw$selected.model
selectedModelDraw<-"oppTeam + t_Corners + t_diffBothGoals + t_diffBothGoals_where + t_diffBothGoals2H + t_Fouls + t_Fouls_where + t_oppCorners + t_oppFouls + t_oppFouls_where + t_oppShots + t_oppShots_where + t_oppShotstarget + t_Shots + t_Shots_where + t_Shotstarget + team"
#"mt_oppCorners_where + mt_oppgoal_efficiency_where + mt_oppShots_where + mt_Shotstarget_where + oppTeam + t_diffBothGoals + t_diffBothGoals_where + t_diffBothGoals2H + t_diffBothGoals2H_where + t_diffGoals_where + t_diffOppGoals + t_diffOppGoals_where + team + where"

selectedVarsDraw<-vsDraw$selected.vars

vsDraw$firstForest$confusion
plot(vsDraw$firstForest$err.rate[,1], ylab=colnames(vsDraw$firstForest$err.rate)[1])
plot(vsDraw$firstForest$err.rate[,2], ylab=colnames(vsDraw$firstForest$err.rate)[2])
plot(vsDraw$firstForest$err.rate[,3], ylab=colnames(vsDraw$firstForest$err.rate)[3])

table(yWin)/length(yWin)
vs$selec.history %>% select(-Vars.in.Forest)

table(yDraw)/length(yDraw)
vsDraw$selec.history %>% select(-Vars.in.Forest)

paste(selectedVarsDraw, collapse = ", ")

mutinformation(yDraw, yDraw)
mutinformation(yDraw, x %>% select(oppTeam, t_Corners, t_diffBothGoals, t_diffBothGoals_where, t_diffBothGoals2H, t_Fouls, t_Fouls_where, t_oppCorners, t_oppFouls, t_oppFouls_where, t_oppShots, t_oppShots_where, t_oppShotstarget, t_Shots, t_Shots_where, t_Shotstarget, team))
mutinformation(yDraw, x %>% select(oppTeam, t_oppShots_where, t_oppShotstarget, team))

mutinformation(yDraw, yDraw, method="mm")
mutinformation(yDraw, x %>% select(oppTeam, t_Corners, t_diffBothGoals, t_diffBothGoals_where, t_diffBothGoals2H, t_Fouls, t_Fouls_where, t_oppCorners, t_oppFouls, t_oppFouls_where, t_oppShots, t_oppShots_where, t_oppShotstarget, t_Shots, t_Shots_where, t_Shotstarget, team), method="mm")
mutinformation(yDraw, x %>% select(oppTeam, t_oppShots_where, t_oppShotstarget, team), method="mm")

mutinformation(yDraw, yDraw, method="shrink")
mutinformation(yDraw, x %>% select(oppTeam, t_Corners, t_diffBothGoals, t_diffBothGoals_where, t_diffBothGoals2H, t_Fouls, t_Fouls_where, t_oppCorners, t_oppFouls, t_oppFouls_where, t_oppShots, t_oppShots_where, t_oppShotstarget, t_Shots, t_Shots_where, t_Shotstarget, team), method="shrink")
mutinformation(yDraw, x %>% select(oppTeam, t_oppShots_where, t_oppShotstarget, team), method="shrink")

mutinformation(yDraw, yDraw, method="sg")
mutinformation(yDraw, x %>% select(oppTeam, t_Corners, t_diffBothGoals, t_diffBothGoals_where, t_diffBothGoals2H, t_Fouls, t_Fouls_where, t_oppCorners, t_oppFouls, t_oppFouls_where, t_oppShots, t_oppShots_where, t_oppShotstarget, t_Shots, t_Shots_where, t_Shotstarget, team), method="sg")
mutinformation(yDraw, x %>% select(oppTeam, t_oppShots_where, t_oppShotstarget, team), method="sg")

str(sapply(selectedVarsDraw, paste, sep=","))

multiinformation(x %>% select(oppTeam, t_Corners, t_diffBothGoals, t_diffBothGoals_where, t_diffBothGoals2H, t_Fouls, t_Fouls_where, t_oppCorners, t_oppFouls, t_oppFouls_where, t_oppShots, t_oppShots_where, t_oppShotstarget, t_Shots, t_Shots_where, t_Shotstarget, team))
#interinformation(x %>% select(oppTeam, t_Corners, t_diffBothGoals, t_diffBothGoals_where, t_diffBothGoals2H, t_Fouls, t_Fouls_where, t_oppCorners, t_oppFouls, t_oppFouls_where, t_oppShots, t_oppShots_where, t_oppShotstarget, t_Shots, t_Shots_where, t_Shotstarget, team))

condinformation(yDraw, x %>% select(oppTeam, t_oppShots_where, t_oppShotstarget, team), teamresults$GS)



#train<-train %>% select(-starts_with("l5"))
#train<-train %>% select(-starts_with("mt"))

train<-xy %>% filter(season %in% c(2015,2014) ) 
test<-xy%>% filter(season == 2016)

################################################################################
# build models and evaluate training data scores

model_cv_goals <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree=1000, mtry=20))
pred<-predict(model_cv_goals, newdata = train)
gspred<-sapply(pred, function(x) x[,'GS'])
gcpred<-sapply(pred, function(x) x[,'GC'])
result<-summarizePrediction(xy = train, round(gspred), round(gcpred))

data_raw<-data.frame(gspred, gcpred, train)
model2<-cforest(GS*GC ~ ., data=data_raw %>% select(GS, GC, gspred, gcpred), controls = cforest_unbiased(ntree=50, mtry=2))

pred2<-predict(model2, newdata = data_raw)
gspred2<-sapply(pred2, function(x) x[,'GS'])
gcpred2<-sapply(pred2, function(x) x[,'GC'])
result<-summarizePrediction(xy = train, round(gspred2), round(gcpred2))

data3<-train %>% mutate(isDraw=as.factor(1-abs(sign(GS-GC)))) %>% select(-GS, -GC)
model3<-cforest(isDraw ~ ., data=data3, controls = cforest_unbiased(ntree=1000, mtry=3))
#sort(varimp(model3), decreasing = TRUE)
drawPred<-predict(model3, newdata=data3, type="prob")
predDraws<-t(sapply(drawPred, cbind))
summary(predDraws[,2])
boxplot(predDraws[,2]  ~ data3$isDraw)
hist(predDraws[,2])
plot(predDraws[,2], data3$isDraw)

table(predDraws[,2], data3$isDraw)


data_multifactor<-data_raw %>% 
  mutate(eg1=ifelse(GS>4,GS-4,0), eg2=ifelse(GC>4,GC-4,0)) %>% 
  mutate(eg3=ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0), eg4=ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)) %>%
  mutate(score=factor(paste(eg3, eg4, sep = ":"), 
                       levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                                "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                                "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE))%>% 
  select(score, gspred, gcpred, team, oppTeam, where, starts_with("mt_Yellow"), starts_with("mt_Red"), starts_with("mt_Fouls")) # , starts_with("t_diffBothGoals")


model4<-cforest(score ~ ., data=data_multifactor, controls = cforest_unbiased(ntree=1000, mtry=5))
#sort(varimp(model4), decreasing = TRUE)

pred4<-predict(model4, newdata = data_multifactor )
gspred4<-as.integer(substr(pred4,1,1))
gcpred4<-as.integer(substr(pred4,3,3))
result<-summarizePrediction(xy = train, gspred4, gcpred4)

data_all_multifactor<-cbind(data_raw, score=data_multifactor$score) %>% select(-GS, -GC)

model6<-cforest(score ~ ., data=data_all_multifactor, controls = cforest_unbiased(ntree=500, mtry=5))

pred6<-predict(model6, newdata = data_all_multifactor )
gspred6<-as.integer(substr(pred6,1,1))
gcpred6<-as.integer(substr(pred6,3,3))
result<-summarizePrediction(xy = train, gspred6, gcpred6)


yDiff<-train$GS-train$GC
yDiff[yDiff>3]<-3
yDiff[yDiff< -3]<- -3
yDiff<-as.factor(yDiff)
xDiff <- data_raw %>% select(-GS, -GC)
vsModel6<-varSelRF(xDiff, yDiff, verbose = TRUE, ntree=1000)

dput(vsModel6, file="varSelRF_vsModel6")
plot(vsModel6)

selectedModel6<-vsModel6$selected.model
selectedModel6<-"gcpred + gspred + mt_Shotstarget_where + t_diffBothGoals_where + t_diffOppGoals + t_diffOppGoals_where"
selectedModel6<-"gspred + gcpred + t_diffOppGoals_where + mt_Shotstarget_where + t_diffOppGoals + t_diffBothGoals_where"
altxxxxxxxxxxx<-"gspred + gcpred + t_diffOppGoals_where + mt_Shotstarget_where + t_diffOppGoals + t_diffBothGoals_where + t_diffBothGoals + t_diffGoals_where + t_diffGoals + team + t_diffPoints + t_diffOppGoals2H_where + mt_oppShots_where + t_diffOppGoals2H + mt_oppPoints + mt_Points + t_Corners_where + t_Shotstarget_where + mt_oppShot_efficiency_where + t_oppShots + mt_oppgoal_efficiency_where"
selectedVars6<-vsModel6$selected.vars

vsModel6$firstForest$confusion
plot(vsModel6$firstForest$err.rate[,1], ylab=colnames(vsModel6$firstForest$err.rate)[1])
plot(vsModel6$firstForest$err.rate[,2], ylab=colnames(vsModel6$firstForest$err.rate)[2])
plot(vsModel6$firstForest$err.rate[,3], ylab=colnames(vsModel6$firstForest$err.rate)[3])
plot(vsModel6$firstForest$err.rate[,4], ylab=colnames(vsModel6$firstForest$err.rate)[4])
plot(vsModel6$firstForest$err.rate[,5], ylab=colnames(vsModel6$firstForest$err.rate)[5])
plot(vsModel6$firstForest$err.rate[,6], ylab=colnames(vsModel6$firstForest$err.rate)[6])
plot(vsModel6$firstForest$err.rate[,7], ylab=colnames(vsModel6$firstForest$err.rate)[7])
plot(vsModel6$firstForest$err.rate[,8], ylab=colnames(vsModel6$firstForest$err.rate)[8])

vsModel6$selec.history %>% select (-Vars.in.Forest)

vsModel6$selec.history$Vars.in.Forest

        )

        


model7<-cforest(score ~ ., data=data_multifactor %>% select(score, gspred, gcpred), 
                controls = cforest_unbiased(ntree=500, mtry=2))
pred7<-predict(model7, newdata = data_multifactor )
gspred7<-as.integer(substr(pred7,1,1))
gcpred7<-as.integer(substr(pred7,3,3))
result<-summarizePrediction(xy = train, gspred7, gcpred7)


model_cv_goals1 <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree=1000, mtry=20))
model_cv_goals2 <- cforest(GS*GC ~ ., data=train%>%filter(GS!=GC), controls = cforest_unbiased(ntree=1000, mtry=10))
model_cv_goals3 <- cforest(GS*GC ~ ., data=train%>%select(GS, GC, team, oppTeam, where) , controls = cforest_unbiased(ntree=1000, mtry=2))
model_cv_goals4 <- cforest(GS*GC ~ ., data=train, controls = cforest_unbiased(ntree=500, mtry=5))

pred1<-predict(model_cv_goals1, newdata = train)
gspred1<-sapply(pred1, function(x) x[,'GS'])
gcpred1<-sapply(pred1, function(x) x[,'GC'])
result<-summarizePrediction(xy = train, round(gspred1), round(gcpred1))
pred2<-predict(model_cv_goals2, newdata = train)
gspred2<-sapply(pred2, function(x) x[,'GS'])
gcpred2<-sapply(pred2, function(x) x[,'GC'])
result<-summarizePrediction(xy = train, round(gspred2), round(gcpred2))
pred3<-predict(model_cv_goals3, newdata = train)
gspred3<-sapply(pred3, function(x) x[,'GS'])
gcpred3<-sapply(pred3, function(x) x[,'GC'])
result<-summarizePrediction(xy = train, round(gspred3), round(gcpred3))
pred4<-predict(model_cv_goals4, newdata = train)
gspred4<-sapply(pred4, function(x) x[,'GS'])
gcpred4<-sapply(pred4, function(x) x[,'GC'])
result<-summarizePrediction(xy = train, round(gspred4), round(gcpred4))

data_is_draw<-train %>% mutate(isDraw=as.factor(1-abs(sign(GS-GC)))) %>% select(-GS, -GC)
model_is_draw<-cforest(isDraw ~ ., data=data_is_draw, controls = cforest_unbiased(ntree=1000, mtry=3))
drawPred<-predict(model_is_draw, newdata=data_is_draw, type="prob")
drawProb<-t(sapply(drawPred, cbind))[,2]

data_train8<-train %>% 
  mutate(eg1=ifelse(GS>4,GS-4,0), eg2=ifelse(GC>4,GC-4,0)) %>% 
  mutate(eg3=ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0), eg4=ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)) %>%
  mutate(score=factor(paste(eg3, eg4, sep = ":"), 
                      levels=c("0:4", "1:4", "0:3", "2:4", "1:3", "0:2", "3:4", "2:3", "1:2", "0:1",
                               "0:0", "1:1", "2:2", "3:3", "4:4", "1:0", "2:1", "3:2", "4:3", 
                               "2:0", "3:1", "4:2", "3:0", "4:1", "4:0" ), ordered=TRUE))%>% 
  select(score, team, oppTeam, where)
data_train8<-data.frame(data_train8, gspred1, gcpred1, gspred2, gcpred2, gspred3, gcpred3, gspred4, gcpred4
                        #, drawProb
                        )

model8<-cforest(score ~ ., data=data_train8, controls = cforest_unbiased(ntree=1000, mtry=5))
#sort(varimp(model8), decreasing = TRUE)

pred8<-predict(model8, newdata = data_train8 )
gspred8<-as.integer(substr(pred8,1,1))
gcpred8<-as.integer(substr(pred8,3,3))
result<-summarizePrediction(xy = train, gspred8, gcpred8)

gc(TRUE)

################################################################################
# evaluate oob test data scores


pred<-predict(model_cv_goals, newdata = test)
gspred<-sapply(pred, function(x) x[,'GS'])
gcpred<-sapply(pred, function(x) x[,'GC'])
result<-summarizePrediction(xy = test, round(gspred), round(gcpred))

data2<-data.frame(gspred, gcpred, test)
pred2<-predict(model2, newdata = data2)
gspred2<-sapply(pred2, function(x) x[,'GS'])
gcpred2<-sapply(pred2, function(x) x[,'GC'])
result<-summarizePrediction(xy = test, round(gspred2), round(gcpred2))

pred3<-predict(model3, newdata=test, type="prob")
pred3Draws<-t(sapply(pred3, cbind))
summary(pred3Draws[,2])
boxplot(pred3Draws[,2]  ~ test$GS==test$GC)
hist(pred3Draws[,2])
smoothScatter(pred3Draws[,2], test$GS==test$GC)


data4<-data.frame(gspred, gcpred, test)
pred4<-predict(model4, newdata = data4 )
pred4[pred4=="4:0"]<-"3:0"
pred4[pred4=="0:4"]<-"0:3"
gspred4<-as.integer(substr(pred4,1,1))
gcpred4<-as.integer(substr(pred4,3,3))
result<-summarizePrediction(xy = test, gspred4, gcpred4)

predProbsList<-predict(model4, newdata = data4 , type = "prob")
predoutcomes<-t(sapply(predProbsList, cbind))
colnames(predoutcomes)<-colnames(predProbsList[[1]])
predoutcomes<-predoutcomes[,order(colnames(predoutcomes))]  
colnames(predoutcomes)<-substr(colnames(predoutcomes), 7,9)
substr(colnames(predoutcomes), 2,2)<-":"

matchidx<-match(paste(test$GS, test$GC, sep = ":"), colnames(predoutcomes))
matchedprobs<-predoutcomes[cbind(1:length(matchidx), matchidx)]
summary(matchedprobs)

matchidx4<-match(paste(gspred4, gcpred4, sep = ":"), colnames(predoutcomes))
matchedprobs4<-predoutcomes[cbind(1:length(matchidx4), matchidx4)]
summary(matchedprobs4)

mask<-mask[order(names(mask))]
expectedValues<-sapply(1:25, function(i) predoutcomes %*% as.vector(t(mask[i][[1]])))
colnames(expectedValues)<-names(mask)

maxEV<-apply(expectedValues, 1, max)
whichMaxEV<-apply(expectedValues, 1, which.max)

pred5<-colnames(expectedValues)[whichMaxEV]
pred5[pred5=="4 0"]<-"3 0"
pred5[pred5=="0 4"]<-"0 3"
gspred5<-as.integer(substr(pred5,1,1))
gcpred5<-as.integer(substr(pred5,3,3))
result<-summarizePrediction(xy = test, gspred5, gcpred5)

pred6<-predict(model6, newdata = data4 )
gspred6<-as.integer(substr(pred6,1,1))
gcpred6<-as.integer(substr(pred6,3,3))
result<-summarizePrediction(xy = test, gspred6, gcpred6)

pred7<-predict(model7, newdata = data4 )
pred7[pred7=="4:0"]<-"3:0"
pred7[pred7=="0:4"]<-"0:3"
gspred7<-as.integer(substr(pred7,1,1))
gcpred7<-as.integer(substr(pred7,3,3))
result<-summarizePrediction(xy = test, gspred7, gcpred7)

#result<-summarizePrediction(xy = test, rep(2, nrow(test)), rep(1, nrow(test)))
#dummy<-data.frame(where=test$where, gspred=2, gcpred=1)
#dummy$gspred[dummy$where=="Away"]<-1
#dummy$gcpred[dummy$where=="Away"]<-2
#result<-summarizePrediction(xy = test, dummy$gspred, dummy$gcpred)


pred1<-predict(model_cv_goals1, newdata = test)
gspred1<-sapply(pred1, function(x) x[,'GS'])
gcpred1<-sapply(pred1, function(x) x[,'GC'])
result<-summarizePrediction(xy = test, round(gspred1), round(gcpred1))
pred2<-predict(model_cv_goals2, newdata = test)
gspred2<-sapply(pred2, function(x) x[,'GS'])
gcpred2<-sapply(pred2, function(x) x[,'GC'])
result<-summarizePrediction(xy = test, round(gspred2), round(gcpred2))
pred3<-predict(model_cv_goals3, newdata = test)
gspred3<-sapply(pred3, function(x) x[,'GS'])
gcpred3<-sapply(pred3, function(x) x[,'GC'])
result<-summarizePrediction(xy = test, round(gspred3), round(gcpred3))
pred4<-predict(model_cv_goals4, newdata = test)
gspred4<-sapply(pred4, function(x) x[,'GS'])
gcpred4<-sapply(pred4, function(x) x[,'GC'])
result<-summarizePrediction(xy = test, round(gspred4), round(gcpred4))

cv_data_is_draw<-test %>% mutate(isDraw=as.factor(1-abs(sign(GS-GC)))) %>% select(-GS, -GC)
cv_drawPred<-predict(model_is_draw, newdata=cv_data_is_draw, type="prob")
cv_drawProb<-t(sapply(cv_drawPred, cbind))[,2]

data_test8<-data.frame(test, gspred1, gcpred1, gspred2, gcpred2, gspred3, gcpred3, gspred4, gcpred4
                       #, drawProb=cv_drawProb
                       )

pred8<-predict(model8, newdata = data_test8 )
gspred8<-as.integer(substr(pred8,1,1))
gcpred8<-as.integer(substr(pred8,3,3))
result<-summarizePrediction(xy = test, gspred8, gcpred8)

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
result<-summarizePrediction(xy = test, gspred9, gcpred9)


varimp_model <- cforest(I(GS-GC) ~ ., data=train, controls = cforest_unbiased(ntree=1000, mtry=20))
vi<-sort(varimp(varimp_model), decreasing = TRUE)
plot(vi)




