setwd("~/LearningR/Bundesliga")

download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
download.file("http://www.football-data.co.uk/mmz4281/1415/D1.csv", "BL2014.csv")

#install.packages("dplyr")

library(dplyr)
library(Matrix)
library(recommenderlab)
library(pscl)
library(lazyeval)

data1<-read.csv("BL2016.csv")
data1$season<-"2016_17"
data2<-read.csv("BL2015.csv")
data2$season<-"2015_16"
data3<-read.csv("BL2014.csv")
data3$season<-"2014_15"
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
                          Shots=results$HS, Shotstarget=results$HST, Fouls=results$HF, Corners=results$HC, Yellow=results$HY, Red=results$HR,
                          oppShots=results$AS, oppShotstarget=results$AST, oppFouls=results$AF, oppCorners=results$AC, oppYellow=results$AY, oppRed=results$AR)
teamresults <- rbind(data.frame(team=results$AwayTeam, oppTeam=results$HomeTeam, 
                                GS=results$FTAG, GC=results$FTHG, where="Away", round=results$round, season=results$season,
                                GS1H=results$HTAG, GC1H=results$HTHG, GS2H=results$FTAG-results$HTAG, GC2H=results$FTHG-results$HTHG, 
                                dow=results$dayofweek, gameindex=results$gameindex,
                                Shots=results$AS, Shotstarget=results$AST, Fouls=results$AF, Corners=results$AC, Yellow=results$AY, Red=results$AR,
                                oppShots=results$HS, oppShotstarget=results$HST, oppFouls=results$HF, oppCorners=results$HC, oppYellow=results$HY, oppRed=results$HR),
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



# approxSVD<-function(data, targetround, rank){
#   m<-with(data %>% 
#             transmute(i=as.integer(team), j=as.integer(oppTeam), x=svdx_input),
#           as.matrix(sparseMatrix(i=i, j=j, x=x)))
#   m[rowSums(m)==0,]<-NA
#   m[,colSums(m, na.rm = TRUE)==0]<-NA
# #  print(m)
#   
#   futureGames<-with(data %>% filter(round>=targetround) %>% 
#                       transmute(i=as.integer(team), j=as.integer(oppTeam), x=NA),
#                     as.matrix(sparseMatrix(i=i, j=j, x=x, dims = dim(m))))
#   
#   maskedMatrix<-m+futureGames
# }

approxSVD<-function(data, maskedMatrix, targetround, rank){
    m0<-mean(maskedMatrix, na.rm=TRUE)
#m0<-0
    #  print(Matrix(maskedMatrix))
  m.svd<-funkSVD(maskedMatrix-m0, k = rank)
  predMatrix<-predict(m.svd, newdata=maskedMatrix)+m0
#  print(Matrix(predMatrix))
  retData<-data %>% mutate(svdx_output=
                             ifelse(round==targetround, 
                                    predMatrix[cbind(as.integer(team), as.integer(oppTeam))],
                                    svdx_output)
                             )
  return(retData)    
}

enrichSVD<-function(data, rank=5) {
  svddata<-data %>% 
    select(gameindex, season, where, round, team, oppTeam, svdx_input) %>% 
    mutate(svdx_output=NA)
  svd_outdata<-svddata %>% filter(FALSE)
  for (iseason in levels(data$season)){
    for (iwhere in levels(data$where))  {
      idata<-svddata %>% filter(season==iseason & where==iwhere)
      l<-nlevels(data$team)
      m<-matrix(NA, ncol=l, nrow=l)
      for (iround in 2:(idata %>% summarise(maxround=max(round)))$maxround) {
        d<-idata %>% filter(round==iround-1)
        m[cbind(as.integer(d$team), as.integer(d$oppTeam))]<-d$svdx_input
        idata <- approxSVD(data = idata, m, targetround = iround, rank = rank)
        #print(iround)
        #print(idata %>% filter(round==iround) %>% select (team, oppTeam, svdx_input, svdx_output, round, season, where)) 
      }
      svd_outdata <- rbind(svd_outdata, idata)
    }
  }
  data<-left_join(data, y = svd_outdata)
  
  qqplot(data$svdx_output, data$svdx_input)
  data %>% select(svdx_output, svdx_input) %>% smoothScatter(main=(data %>% select(svdx_output, svdx_input) %>% cor(use = "complete.obs"))[1,2])
  abline(lm(svdx_input ~ svdx_output, data=data), col="red")
  print(data %>% select(svdx_output, svdx_input) %>% cor(use = "complete.obs"))
  return(data)
}

teamresults<-teamresults %>% mutate(svdx_input=Shots) %>% enrichSVD(rank=6) %>% mutate(SVD_Shots=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Shotstarget) %>% enrichSVD(rank=6) %>% mutate(SVD_Shotstarget=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Corners) %>% enrichSVD(rank=6) %>% mutate(SVD_Corners=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Fouls) %>% enrichSVD(rank=3) %>% mutate(SVD_Fouls=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Yellow) %>% enrichSVD(rank=2) %>% mutate(SVD_Yellow=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Red) %>% enrichSVD(rank=2) %>% mutate(SVD_Red=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS) %>% enrichSVD(rank=6) %>% mutate(SVD_GS=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GC) %>% enrichSVD(rank=6) %>% mutate(SVD_GC=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS-GC) %>% enrichSVD(rank=6) %>% mutate(SVD_GDiff=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS1H) %>% enrichSVD(rank=4) %>% mutate(SVD_GS1H=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GC1H) %>% enrichSVD(rank=4) %>% mutate(SVD_GC1H=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS1H-GC1H) %>% enrichSVD(rank=4) %>% mutate(SVD1H_GDiff=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS2H) %>% enrichSVD(rank=4) %>% mutate(SVD_GS2H=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GC2H) %>% enrichSVD(rank=4) %>% mutate(SVD_GC2H=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS2H-GC2H) %>% enrichSVD(rank=4) %>% mutate(SVD2H_GDiff=svdx_output) %>% select(-svdx_input, -svdx_output) 

teamresults<-teamresults %>% mutate(svdx_input=Shots) %>% enrichSVD(rank=1) %>% mutate(SVD1_Shots=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Shotstarget) %>% enrichSVD(rank=1) %>% mutate(SVD1_Shotstarget=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Corners) %>% enrichSVD(rank=1) %>% mutate(SVD1_Corners=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Fouls) %>% enrichSVD(rank=1) %>% mutate(SVD1_Fouls=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Yellow) %>% enrichSVD(rank=1) %>% mutate(SVD1_Yellow=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=Red) %>% enrichSVD(rank=1) %>% mutate(SVD1_Red=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS) %>% enrichSVD(rank=1) %>% mutate(SVD1_GS=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GC) %>% enrichSVD(rank=1) %>% mutate(SVD1_GC=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=GS-GC) %>% enrichSVD(rank=1) %>% mutate(SVD1_GDiff=svdx_output) %>% select(-svdx_input, -svdx_output) 

teamresults<-teamresults %>% mutate(svdx_input=oppShots) %>% enrichSVD(rank=6) %>% mutate(SVD_oppShots=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppShotstarget) %>% enrichSVD(rank=6) %>% mutate(SVD_oppShotstarget=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppCorners) %>% enrichSVD(rank=6) %>% mutate(SVD_oppCorners=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppFouls) %>% enrichSVD(rank=3) %>% mutate(SVD_oppFouls=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppYellow) %>% enrichSVD(rank=2) %>% mutate(SVD_oppYellow=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppRed) %>% enrichSVD(rank=2) %>% mutate(SVD_oppRed=svdx_output) %>% select(-svdx_input, -svdx_output) 

teamresults<-teamresults %>% mutate(svdx_input=oppShots) %>% enrichSVD(rank=1) %>% mutate(SVD1_oppShots=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppShotstarget) %>% enrichSVD(rank=1) %>% mutate(SVD1_oppShotstarget=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppCorners) %>% enrichSVD(rank=1) %>% mutate(SVD1_oppCorners=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppFouls) %>% enrichSVD(rank=1) %>% mutate(SVD1_oppFouls=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppYellow) %>% enrichSVD(rank=1) %>% mutate(SVD1_oppYellow=svdx_output) %>% select(-svdx_input, -svdx_output) 
teamresults<-teamresults %>% mutate(svdx_input=oppRed) %>% enrichSVD(rank=1) %>% mutate(SVD1_oppRed=svdx_output) %>% select(-svdx_input, -svdx_output) 

enrichPoisson<-function(input_data) {
  poiss_data<-input_data %>% 
    dplyr::select(poisx_input, gameindex, season, where, round, team, oppTeam, 
                  mt_Points, mt_GS, mt_GC, mt_GS1H, mt_GC1H, mt_GS2H, mt_GC2H, mt_Shots, mt_Shotstarget, 
                  mt_Fouls, mt_Corners, mt_Yellow, 
                  mt_oppPoints, mt_oppGS, mt_oppGC, mt_oppGS1H, mt_oppGC1H, mt_oppGS2H, mt_oppGC2H, mt_oppShots, mt_oppShotstarget, 
                  mt_oppFouls, mt_oppCorners, mt_oppYellow,
                  mt_Shots_where, mt_oppShots_where
                  # mt_goal_efficiency, mt_Shot_efficiency, mt_oppgoal_efficiency, mt_oppShot_efficiency, mt_Red,  mt_oppRed
                  ) %>% 
    mutate(poisx_output=NA, simple_poisx_output=NA)
  pois_outdata<-poiss_data %>% filter(FALSE)
  for (iseason in levels(input_data$season)){
      idata<-poiss_data %>% filter(season==iseason) 
      for (iround in 5:(idata %>% summarise(maxround=max(round)))$maxround) {
        traindata_poiss<-idata %>% filter(round < iround)
        newdata_poiss<-idata %>% filter(round == iround)
        limit<-max(newdata_poiss$poisx_input)*2
        model<-glm(formula = poisx_input ~ .+(team+oppTeam+.)*where,#. + (team+oppTeam+.)*where , 
                   data=traindata_poiss %>% dplyr::select(-season, -poisx_output, -simple_poisx_output), 
                   family = poisson,
                   weights = 1-exp(-traindata_poiss$round/34)) # + (team+oppTeam)*where, , weights = 1-exp(-traindata_poiss$round/34)
        # print(model)
        preddata <- predict(object=model, type = "response", newdata=newdata_poiss)
        preddata[preddata>limit] <- NA # prevents bad fit
        simplemodel<-glm(formula = poisx_input ~ (team+oppTeam)*where,
                   data=traindata_poiss %>% dplyr::select(-season, -poisx_output, -simple_poisx_output), 
                   family = poisson
                   ) #, weights = 1-exp(-traindata_poiss$round/34)) # + (team+oppTeam)*where, , weights = 1-exp(-traindata_poiss$round/34)
        simple_preddata <- predict(object=simplemodel, type = "response", newdata=newdata_poiss)
        simple_preddata[simple_preddata>limit] <- NA # prevents bad fit
        
        idata<-left_join(by = c("gameindex", "where", "round"), x = idata, y=data.frame(
          poisx_output2=preddata, simple_poisx_output2=simple_preddata, round=iround, gameindex=newdata_poiss$gameindex, where=newdata_poiss$where)) %>% 
          mutate(poisx_output=ifelse(round==iround, poisx_output2, poisx_output),
                 simple_poisx_output=ifelse(round==iround, simple_poisx_output2, simple_poisx_output)) %>%
          dplyr::select(-poisx_output2, -simple_poisx_output2)
        
        #print(iround)
        #print(idata %>% filter(round==iround) %>% dplyr::select (team, oppTeam, poisx_input, poisx_output, simple_poisx_output, round, season, where)) 
      }
      pois_outdata <- rbind(pois_outdata, idata)
  }
  input_data<-left_join(input_data, y = pois_outdata)
  
  qqplot(input_data$poisx_output, input_data$poisx_input)
  input_data %>% dplyr::select(poisx_output, poisx_input) %>% smoothScatter(main=(input_data %>% dplyr::select(poisx_output, poisx_input) %>% cor(use = "complete.obs"))[1,2])
  abline(lm(poisx_input ~ poisx_output, data=input_data), col="red")
  print(input_data %>% dplyr::select(poisx_output, poisx_input) %>% cor(use = "complete.obs"))
  qqplot(input_data$simple_poisx_output, input_data$poisx_input)
  input_data %>% dplyr::select(simple_poisx_output, poisx_input) %>% smoothScatter(main=(input_data %>% dplyr::select(simple_poisx_output, poisx_input) %>% cor(use = "complete.obs"))[1,2])
  abline(lm(poisx_input ~ simple_poisx_output, data=input_data), col="red")
  print(input_data %>% dplyr::select(simple_poisx_output, poisx_input) %>% cor(use = "complete.obs"))
  return(input_data)
}

teamresults<-teamresults %>% mutate(poisx_input = Shots) %>% enrichPoisson() %>% mutate(pois1_Shots=simple_poisx_output, pois2_Shots=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = Shotstarget) %>% enrichPoisson() %>% mutate(pois1_Shotstarget=simple_poisx_output, pois2_Shotstarget=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = Yellow) %>% enrichPoisson() %>% mutate(pois1_Yellow=simple_poisx_output, pois2_Yellow=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = Fouls) %>% enrichPoisson() %>% mutate(pois1_Fouls=simple_poisx_output, pois2_Fouls=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = Corners) %>% enrichPoisson() %>% mutate(pois1_Corners=simple_poisx_output, pois2_Corners=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
#teamresults<-teamresults %>% mutate(poisx_input = GS) %>% enrichPoisson() %>% mutate(pois1_GS=simple_poisx_output, pois2_GS=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)

teamresults<-teamresults %>% mutate(poisx_input = oppShots) %>% enrichPoisson() %>% mutate(pois1_oppShots=simple_poisx_output, pois2_oppShots=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = oppShotstarget) %>% enrichPoisson() %>% mutate(pois1_oppShotstarget=simple_poisx_output, pois2_oppShotstarget=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
#teamresults<-teamresults %>% mutate(poisx_input = oppYellow) %>% enrichPoisson() %>% mutate(pois1_oppYellow=simple_poisx_output, pois2_oppYellow=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = oppFouls) %>% enrichPoisson() %>% mutate(pois1_oppFouls=simple_poisx_output, pois2_oppFouls=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = oppCorners) %>% enrichPoisson() %>% mutate(pois1_oppCorners=simple_poisx_output, pois2_oppCorners=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
#teamresults<-teamresults %>% mutate(poisx_input = GC) %>% enrichPoisson() %>% mutate(pois1_GC=simple_poisx_output, pois2_GC=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)


######################################################################################################################
# count scores from past games in same season

limitMaxScore<-function(dataframe, limit) {
  GS<-dataframe$GS
  GC<-dataframe$GC
  eg1<-ifelse(GS>limit,GS-limit,0)
  eg2<-ifelse(GC>limit,GC-limit,0)
  eg3<-ifelse((GS-eg1-eg2)>0,GS-eg1-eg2,0)
  eg4<-ifelse((GC-eg1-eg2)>0,GC-eg1-eg2,0)
  dataframe$GS<-ifelse(GS==GC & GS>=limit, limit, eg3)
  dataframe$GC<-ifelse(GS==GC & GS>=limit, limit, eg4)
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

scorecountsTeam<-sparse.model.matrix(~.-1, data = teamresults %>% select(GS, GC) %>% limitMaxScore(3) %>% buildFactorScore(), drop.unused.levels = TRUE)
scorecountsOppTeam<-sparse.model.matrix(~.-1, data = teamresults %>% select(GS=GC, GC=GS) %>% limitMaxScore(3) %>% buildFactorScore(), drop.unused.levels = TRUE)

scorecounttable<-
  cbind(teamresults %>% select(team, oppTeam, season, round, gameindex, where), 
      matrix(scorecountsTeam, nrow=nrow(teamresults), dimnames=list(NULL, paste0("sc_team_", colnames(scorecountsTeam)))), 
      matrix(scorecountsOppTeam, nrow=nrow(teamresults), dimnames=list(NULL, paste0("sc_oppTeam_", colnames(scorecountsOppTeam))))) 

scorecounttable<-scorecounttable %>%
  group_by(season, team) %>%
  arrange(round) 
  
for(cn in paste0("sc_team_", colnames(scorecountsTeam))){
  scorecounttable <- scorecounttable %>%
    mutate_(.dots= setNames(list(interp(~ as.integer(cumsum(from)-from), from=as.name(cn))), paste0("t_",cn)))
}

scorecounttable<-scorecounttable %>%
  group_by(season, oppTeam) %>%
  arrange(round)
  
for(cn in paste0("sc_oppTeam_", colnames(scorecountsOppTeam))){
  scorecounttable <- scorecounttable %>%
    mutate_(.dots= setNames(list(interp(~ as.integer(cumsum(from)-from), from=as.name(cn))), paste0("t_",cn)))
}

scorecounttable<-scorecounttable %>%
  group_by(season, where, team) %>%
  arrange(round)
  
for(cn in paste0("sc_team_", colnames(scorecountsTeam))){
  scorecounttable <- scorecounttable %>%
    mutate_(.dots= setNames(list(interp(~ as.integer(cumsum(from)-from), from=as.name(cn))), paste0("t_",cn, "_where")))
}

scorecounttable<-scorecounttable %>%
  group_by(season, where, oppTeam) %>%
  arrange(round)

for(cn in paste0("sc_oppTeam_", colnames(scorecountsTeam))){
  scorecounttable <- scorecounttable %>%
    mutate_(.dots= setNames(list(interp(~ as.integer(cumsum(from)-from), from=as.name(cn))), paste0("t_",cn, "_where")))
}

teamresults<-inner_join(teamresults, scorecounttable %>% select(-starts_with("sc_")), 
                        by=c("team", "oppTeam", "season", "round", "where", "gameindex"))


######################################################################################################################
# result from previous game
teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) %>%
  mutate(l1_GS = lag(GS, 1)) %>%  
  mutate(l1_GC = lag(GC, 1)) %>%  
  group_by(season, oppTeam) %>%
  arrange(round) %>%
  mutate(l1_oppGS = lag(GC, 1)) %>%  
  mutate(l1_oppGC = lag(GS, 1)) %>%  
  ungroup()

teamresults<-
  teamresults %>%
  mutate(mt_teamWin = (t_sc_team_score1.0+t_sc_team_score2.0+t_sc_team_score3.0+t_sc_team_score2.1+t_sc_team_score3.1+t_sc_team_score3.2)/t_Matches_total) %>%
  mutate(mt_teamLoss = (t_sc_team_score0.1+t_sc_team_score0.2+t_sc_team_score0.3+t_sc_team_score1.2+t_sc_team_score1.3+t_sc_team_score2.3)/t_Matches_total) %>%
  mutate(mt_teamDraw = (t_sc_team_score0.0+t_sc_team_score1.1+t_sc_team_score2.2+t_sc_team_score3.3)/t_Matches_total) %>%
  mutate(mt_oppTeamWin = (t_sc_oppTeam_score1.0+t_sc_oppTeam_score2.0+t_sc_oppTeam_score3.0+t_sc_oppTeam_score2.1+t_sc_oppTeam_score3.1+t_sc_oppTeam_score3.2)/t_Matches_total) %>%
  mutate(mt_oppTeamLoss = (t_sc_oppTeam_score0.1+t_sc_oppTeam_score0.2+t_sc_oppTeam_score0.3+t_sc_oppTeam_score1.2+t_sc_oppTeam_score1.3+t_sc_oppTeam_score2.3)/t_Matches_total) %>%
  mutate(mt_oppTeamDraw = (t_sc_oppTeam_score0.0+t_sc_oppTeam_score1.1+t_sc_oppTeam_score2.2+t_sc_oppTeam_score3.3)/t_Matches_total) %>%
  mutate(mt_teamWin_where = (t_sc_team_score1.0_where+t_sc_team_score2.0_where+t_sc_team_score3.0_where+t_sc_team_score2.1_where+t_sc_team_score3.1_where+t_sc_team_score3.2_where)/t_Matches_where) %>%
  mutate(mt_teamLoss_where = (t_sc_team_score0.1_where+t_sc_team_score0.2_where+t_sc_team_score0.3_where+t_sc_team_score1.2_where+t_sc_team_score1.3_where+t_sc_team_score2.3_where)/t_Matches_where) %>%
  mutate(mt_teamDraw_where = (t_sc_team_score0.0_where+t_sc_team_score1.1_where+t_sc_team_score2.2_where+t_sc_team_score3.3_where)/t_Matches_where) %>%
  mutate(mt_oppTeamWin_where = (t_sc_oppTeam_score1.0_where+t_sc_oppTeam_score2.0_where+t_sc_oppTeam_score3.0_where+t_sc_oppTeam_score2.1_where+t_sc_oppTeam_score3.1_where+t_sc_oppTeam_score3.2_where)/t_oppMatches_where) %>%
  mutate(mt_oppTeamLoss_where = (t_sc_oppTeam_score0.1_where+t_sc_oppTeam_score0.2_where+t_sc_oppTeam_score0.3_where+t_sc_oppTeam_score1.2_where+t_sc_oppTeam_score1.3_where+t_sc_oppTeam_score2.3_where)/t_oppMatches_where) %>%
  mutate(mt_oppTeamDraw_where = (t_sc_oppTeam_score0.0_where+t_sc_oppTeam_score1.1_where+t_sc_oppTeam_score2.2_where+t_sc_oppTeam_score3.3_where)/t_oppMatches_where) 

write.csv(x = teamresults, file="teamresults.csv", quote = TRUE, row.names = FALSE)

#teamresults<-read.csv(file="teamresults.csv")

x<-teamresults %>% dplyr::select(-points, -oppPoints, -Shots, -Shotstarget, -Fouls, -Corners, -Yellow, -Red, 
                          -oppShots, -oppShotstarget, -oppFouls, -oppCorners, -oppYellow, -oppRed, 
                          -starts_with("GC"),-starts_with("GS"))
y<-teamresults %>% dplyr::select(GS,GC)
xy<-data.frame(y, x)

write.csv(x = xy, file="BLfeatures.csv", quote = TRUE, row.names = FALSE)


