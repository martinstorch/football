setwd("~/LearningR/Bundesliga/Analysis")
setwd("c:/users/marti")

library(metR)
library(ggExtra)
library(ggplot2)
library(ggpmisc)

seasons<-c("0001", "0102", "0203", "0304", "0405","0506","0607", "0708","0809","0910","1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819")
seasons<-c("0405","0506","0607", "0708","0809","0910","1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819")

library(dplyr)
library(reshape2)
library(lfda)
library(caret)
library(MASS)
library(tidyimpute)
library(na.tools)

fetch_data<-function(season){
  url <- paste0("http://www.football-data.co.uk/mmz4281/", season, "/D1.csv")
  inputFile <- paste0("BL",season,".csv")
  download.file(url, inputFile, method = "libcurl")
  data<-read.csv(inputFile)
  data[is.na(data<-data)]<-0
  data$season<-season
  print(str(data))
  if ("HFKC" %in% colnames(data))
  {
    data$HF<-data$HFKC
    data$AF<-data$AFKC
  }
  if (!"BWH" %in% colnames(data))
  {
    data$BWH<-NA
    data$BWD<-NA
    data$BWA<-NA
  }
  if (!"B365H" %in% colnames(data))
  {
    data$B365H<-NA
    data$B365D<-NA
    data$B365A<-NA
  }
  #results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'Date')]
  if (!"HST" %in% colnames(data)){
    data <- data%>%mutate(HST=HS*0.376, AST=AS*0.376)
  }
  results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'Date', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR' , 'BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')]
  results$season<-as.factor(results$season)
  with(results[!is.na(results$FTR),],{
    print(cbind ( table(FTR , season), table(FTR , season)/length(FTR)*100))
    print(table(FTHG , FTAG))
    print(cbind(meanHG = mean(FTHG), varHG = var(FTHG), meanAG = mean(FTAG),varAG = var(FTAG)))
    print(cor(FTHG, FTAG))
  })
  results$spieltag <- floor((9:(nrow(results)+8))/9)
  results$round <- ((results$spieltag-1) %% 34) +1
  results$subround<-as.factor(floor(results$round/6))
  results$Date<-as.Date(results$Date, "%d/%m/%y")
  results$dayofweek<-weekdays(results$Date)
  results$gameindex<-(0:(nrow(results)-1))%%9+1
  #print(tail(results[,c(1:10, 23:26)], 18))
  return(results)
}

alldata<-data.frame()
for (s in seasons) {
  alldata<-rbind(alldata, fetch_data(s))
}
str(alldata)
#mean(alldata$HST)/mean(alldata$HS)
#mean(alldata$AST)/mean(alldata$AS)

quote_names<-c('BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')
quote_names<-c('BWH', 'BWD', 'BWA')
normalized_quote_names<-paste0("p", quote_names)

alldata[,normalized_quote_names]<-1/alldata[,quote_names]
alldata[,normalized_quote_names[1:3]]<-alldata[,normalized_quote_names[1:3]]/rowSums(alldata[,normalized_quote_names[1:3]])
#alldata[,normalized_quote_names[4:6]]<-alldata[,normalized_quote_names[4:6]]/rowSums(alldata[,normalized_quote_names[4:6]])



#########################################################################################################################

str(alldata)
tail(alldata)


#data<-rbind(data3[,colnames(data1)], data2, data1)
teams <- unique(alldata$HomeTeam)
results <- alldata[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'round', 'Date', 'dayofweek', 'HS', 'AS', 'HST', 'AST', 
                      'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR', normalized_quote_names )]
results$season<-as.factor(results$season)
table(results$FTR , results$season)
#results$spieltag <- floor((9:(nrow(results)+8))/9)
#results$round <- ((results$spieltag-1) %% 34) +1
#results$Date<-as.Date(results$Date, "%d/%m/%y")
#results$dayofweek<-weekdays(results$Date)
results$gameindex<-(0:(nrow(results)-1))%%9+1
teamresults <- data.frame(team=results$HomeTeam, oppTeam=results$AwayTeam, 
                          GS=results$FTHG, GC=results$FTAG, where="Home", round=results$round, season=results$season, 
                          GS1H=results$HTHG, GC1H=results$HTAG, GS2H=results$FTHG-results$HTHG, GC2H=results$FTAG-results$HTAG, 
                          dow=results$dayofweek, gameindex=results$gameindex,
                          Shots=results$HS, Shotstarget=results$HST, 
                          Fouls=results$HF, Corners=results$HC, Yellow=results$HY, Red=results$HR,
                          oppShots=results$AS, oppShotstarget=results$AST, 
                          oppFouls=results$AF, oppCorners=results$AC, oppYellow=results$AY, oppRed=results$AR,
                          bwinWin=results$pBWH, bwinLoss=results$pBWA, bwinDraw=results$pBWD)
teamresults <- rbind(data.frame(team=results$AwayTeam, oppTeam=results$HomeTeam, 
                                GS=results$FTAG, GC=results$FTHG, where="Away", round=results$round, season=results$season,
                                GS1H=results$HTAG, GC1H=results$HTHG, GS2H=results$FTAG-results$HTAG, GC2H=results$FTHG-results$HTHG, 
                                dow=results$dayofweek, gameindex=results$gameindex,
                                Shots=results$AS, Shotstarget=results$AST, 
                                Fouls=results$AF, Corners=results$AC, Yellow=results$AY, Red=results$AR,
                                oppShots=results$HS, oppShotstarget=results$HST, 
                                oppFouls=results$HF, oppCorners=results$HC, oppYellow=results$HY, oppRed=results$HR,
                                bwinWin=results$pBWA, bwinLoss=results$pBWH, bwinDraw=results$pBWD),
                     teamresults)

teamresults<-teamresults[order(teamresults$season, teamresults$round, teamresults$team),]
#unique(teamresults$season)
teamresults$points<-ifelse(
  sign(teamresults$GS - teamresults$GC)==1, 3, 
  ifelse(teamresults$GS == teamresults$GC, 1, 0))

teamresults$oppPoints<-ifelse(
  sign(teamresults$GS - teamresults$GC)==1, 0, 
  ifelse(teamresults$GS == teamresults$GC, 1, 3))

library(TTR)
#install.packages("TTR")

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
  mutate(ema_Cards = lag(EMA(Red+0.25*Yellow, n=1, ratio=0.2), 1)) %>%
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
  mutate(ema_oppCards = lag(EMA(Red+0.25*Yellow, n=1, ratio=0.2), 1)) %>%
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
  ungroup()%>%
  mutate(ema_Cards = ifelse(is.na(ema_Cards), 0.5, ema_Cards),
         ema_oppCards = ifelse(is.na(ema_oppCards), 0.5, ema_oppCards))

#teamresults%>%arrange(season, team)%>%dplyr::select(season, team, Yellow, Red, t_Yellow, t_Red, ema_Cards)%>%print.data.frame()
#teamresults%>%arrange(season, oppTeam)%>%dplyr::select(season, oppTeam, Yellow, Red, t_oppYellow, t_oppRed, ema_oppCards)%>%print.data.frame()

# teamresults<-
#   teamresults %>%
#   group_by(season, round) %>%
#   mutate(t_Rank = rank(-t_Points, ties.method="min"),
#          t_oppRank = rank(-t_oppPoints, ties.method="min")) %>% 
#   arrange(season, round, team) %>%
#   ungroup() %>%  
#   mutate(t_diffRank = -t_Rank+t_oppRank,
#          t_diffPoints = t_Points-t_oppPoints,
#          t_diffGoals = t_GS - t_GC,
#          t_diffGoals1H = t_GS1H - t_GC1H,
#          t_diffGoals2H = t_GS2H - t_GC2H,
#          t_diffOppGoals = t_oppGS - t_oppGC,
#          t_diffOppGoals1H = t_oppGS1H - t_oppGC1H,
#          t_diffOppGoals2H = t_oppGS2H - t_oppGC2H,
#          t_diffGoals_where = t_GS_where - t_GC_where,
#          t_diffGoals1H_where = t_GS1H_where - t_GC1H_where,
#          t_diffGoals2H_where = t_GS2H_where - t_GC2H_where,
#          t_diffOppGoals_where = t_oppGS_where - t_oppGC_where,
#          t_diffOppGoals1H_where = t_oppGS1H_where - t_oppGC1H_where,
#          t_diffOppGoals2H_where = t_oppGS2H_where - t_oppGC2H_where,
#          t_diffBothGoals = t_diffGoals - t_diffOppGoals,
#          t_diffBothGoals1H = t_diffGoals1H - t_diffOppGoals1H,
#          t_diffBothGoals2H = t_diffGoals2H - t_diffOppGoals2H,
#          t_diffBothGoals_where = t_diffGoals_where - t_diffOppGoals_where,
#          t_diffBothGoals1H_where = t_diffGoals1H_where - t_diffOppGoals1H_where,
#          t_diffBothGoals2H_where = t_diffGoals2H_where - t_diffOppGoals2H_where
#   )

teamresults<-
  teamresults %>%
  mutate(t_Points = t_Points/t_Matches_total,
         t_GS = t_GS/t_Matches_total,
         t_GC = t_GC/t_Matches_total,
         t_GS1H = t_GS1H/t_Matches_total,
         t_GC1H = t_GC1H/t_Matches_total,
         t_GS2H = t_GS2H/t_Matches_total,
         t_GC2H = t_GC2H/t_Matches_total,
         t_Shots = t_Shots/t_Matches_total,
         t_Shotstarget = t_Shotstarget/t_Matches_total,
         t_Fouls = t_Fouls/t_Matches_total,
         t_Corners = t_Corners/t_Matches_total,
         t_Yellow = t_Yellow/t_Matches_total,
         t_Red = t_Red/t_Matches_total,
         
         
         t_oppPoints = t_oppPoints/t_Matches_total,
         t_oppGS = t_oppGS/t_Matches_total,
         t_oppGC = t_oppGC/t_Matches_total,
         t_oppGS1H = t_oppGS1H/t_Matches_total,
         t_oppGC1H = t_oppGC1H/t_Matches_total,
         t_oppGS2H = t_oppGS2H/t_Matches_total,
         t_oppGC2H = t_oppGC2H/t_Matches_total,
         t_oppShots = t_oppShots/t_Matches_total,
         t_oppShotstarget = t_oppShotstarget/t_Matches_total,
         t_oppFouls = t_oppFouls/t_Matches_total,
         t_oppCorners = t_oppCorners/t_Matches_total,
         t_oppYellow = t_oppYellow/t_Matches_total,
         t_oppRed = t_oppRed/t_Matches_total,
         
         t_Points_where = t_Points_where/t_Matches_where,
         t_GS_where = t_GS_where/t_Matches_where,
         t_GC_where = t_GC_where/t_Matches_where,
         t_GS1H_where = t_GS1H_where/t_Matches_where,
         t_GC1H_where = t_GC1H_where/t_Matches_where,
         t_GS2H_where = t_GS2H_where/t_Matches_where,
         t_GC2H_where = t_GC2H_where/t_Matches_where,
         t_Shots_where = t_Shots_where/t_Matches_where,
         t_Shotstarget_where = t_Shotstarget_where/t_Matches_where,
         t_Fouls_where = t_Fouls_where/t_Matches_where,
         t_Corners_where = t_Corners_where/t_Matches_where,
         t_Yellow_where = t_Yellow_where/t_Matches_where,
         t_Red_where = t_Red_where/t_Matches_where,
         
         
         t_oppPoints_where = t_oppPoints_where/t_oppMatches_where,
         t_oppGS_where = t_oppGS_where/t_oppMatches_where,
         t_oppGC_where = t_oppGC_where/t_oppMatches_where,
         t_oppGS1H_where = t_oppGS1H_where/t_oppMatches_where,
         t_oppGC1H_where = t_oppGC1H_where/t_oppMatches_where,
         t_oppGS2H_where = t_oppGS2H_where/t_oppMatches_where,
         t_oppGC2H_where = t_oppGC2H_where/t_oppMatches_where,
         t_oppShots_where = t_oppShots_where/t_oppMatches_where,
         t_oppShotstarget_where = t_oppShotstarget_where/t_oppMatches_where,
         t_oppFouls_where = t_oppFouls_where/t_oppMatches_where,
         t_oppCorners_where = t_oppCorners_where/t_oppMatches_where,
         t_oppYellow_where = t_oppYellow_where/t_oppMatches_where,
         t_oppRed_where = t_oppRed_where/t_oppMatches_where
         
  ) 

# teamresults$diffpoints<-teamresults$t_Points*teamresults$t_Matches_total-teamresults$t_oppPoints*teamresults$t_Matches_total
# library(e1071)
# 
# teamresults%>%filter(where=="Home")%>%
#   mutate(s=paste(season, as.integer((round-1)/17)))%>%
#   group_by(s)%>%summarise(skew=skewness(t_Points), kurtosis=kurtosis(t_Points), dp=sd(diffpoints)-IQR(abs(diffpoints)), c=cor(GS, GC), FTHG=mean(GS), FTAG=mean(GC), draw=mean(GS==GC), win=mean(GS>GC), loss=mean(GS<GC))%>%dplyr::select(-s)%>%cor()
# 
# teamresults%>%filter(where=="Home")%>%
#   mutate(s=paste(season, as.integer((round-1)/17)))%>%
#   group_by(s)%>%summarise(dp=max(t_Points*t_Matches_total), dpmin=min(t_Points*t_Matches_total), c=cor(GS, GC), FTHG=mean(GS), FTAG=mean(GC), draw=mean(GS==GC), win=mean(GS>GC), loss=mean(GS<GC))%>%dplyr::select(-s)%>%cor()
# 
# teamresults%>%filter(where=="Home")%>%
#   mutate(s=paste(season, as.integer((round-1)/6)))%>%
#   group_by(s)%>%summarise(dp=sd(diffpoints)-IQR(abs(diffpoints)), c=cor(GS, GC), FTHG=mean(GS), FTAG=mean(GC), draw=mean(GS==GC))%>%dplyr::select(c, draw)%>%plot()
# 
# t2<-teamresults%>%filter(where=="Home")%>%
#   mutate(s=paste(season, as.integer((round-1)/6)))%>%
#   group_by(s)%>%summarise(dp=sd(diffpoints)-IQR(abs(diffpoints)), c=cor(GS, GC), FTHG=mean(GS), FTAG=mean(GC), draw=mean(GS==GC))%>%dplyr::select(c, draw)
# 
# plot(t2$c, type="l")
# plot(t2$draw, type="l")
# 
# 

teamresults<-teamresults[order(teamresults$season, teamresults$round, teamresults$gameindex, teamresults$where),]
colnames(teamresults)
(teamresults[9080:9090,c("team", "oppTeam", "where", "t_Rank", "t_oppRank", "t_diffRank" )])

p_points<-function(pHG, pAG, FTHG, FTAG){
  hw<-FTHG>FTAG
  aw<-FTHG<FTAG
  draw<-FTHG==FTAG
  
  tendency<-sign(pHG-pAG)==sign(FTHG-FTAG)
  gdiff<-(pHG-pAG)==(FTHG-FTAG)
  hit<-gdiff & (pHG==FTHG)
  pistor <- tendency+gdiff+hit
  sky <- 2*tendency+3*hit
  return (list(pistor=pistor, sky=sky))
}

rankX <- teamresults%>%filter(where=="Home")%>%dplyr::select(team, oppTeam, GS, GC, round, season, gameindex, t_Rank, t_oppRank, t_diffRank)

rankX$pGS<-ifelse(rankX$t_diffRank>0, 2, 1)
rankX$pGC<-ifelse(rankX$t_diffRank<0, 2, 1)

tail(rankX, 18)



qmix_lfda<-function(pwin, ploss, thedata, quotes){
  #ord <- order(quotes[,1], quotes[,2], quotes[,3])
  ord <- do.call(order, quotes) # varargs ... uses lexicagraphic sorting
  thedata<-thedata[ord,]
  quotes<-quotes[ord,]
  n<-nrow(thedata)
  thedata$i <- 1:nrow(thedata)
  
  cutoffs <- quantile(thedata$i, c(ploss, 1-pwin), names=F) 
  #print(c(pwin, ploss, cutoffs))
  
  qqpred<-cut(thedata$i, c(-1e10, cutoffs+c(0.0, 0.00001), 1e10), labels=c("A", "D", "H"))
  qpred<-as.integer(qqpred)-2
  
  hw<-thedata$FTHG>thedata$FTAG
  aw<-thedata$FTHG<thedata$FTAG
  draw<-thedata$FTHG==thedata$FTAG
  
  tendency<-qpred==sign(thedata$FTHG-thedata$FTAG)
  gdiff<-tendency & ((abs(thedata$FTHG-thedata$FTAG)==1) | draw)
  hit<-gdiff & ((thedata$FTHG+thedata$FTAG) %in% 2:3)
  pistor <- (sum(tendency)+sum(gdiff)+sum(hit))/length(qpred)
  sky <- (2*sum(tendency)+0*sum(gdiff)+3*sum(hit))/length(qpred)
  tcs <- (2*sum(tendency&hw)+1*sum(gdiff&hw)+1*sum(hit&hw)+4*sum(tendency&aw)+1*sum(gdiff&aw)+2*sum(hit&aw)+2*sum(tendency&draw)+4*sum(hit&draw))/length(qpred)
  return (data.frame(pistor=pistor, sky=sky, tcs=tcs, hwin=sum(qpred==1)/length(qpred), draw=sum(qpred==0)/length(qpred), away=sum(qpred==-1)/length(qpred)))
}

prepare_plot_data_lfda<-function(thedata, step=0.01){
  features <- thedata %>% dplyr::select(dplyr::matches("X[0-9]"))
  #print(str(features))
  q<-c()  
  for (pwin in seq(0.3,0.9,step)) {
    for (ploss in seq(0.0, 1.00-pwin, step)) {
      q<-rbind(q, qmix_lfda(pwin, ploss, thedata, features))  
    }
  }
  q <- data.frame(q)
  #print(str(q))
  q<-unique(q)
  return(q)
}
#q<-prepare_plot_data_lfda(alldata%>%filter(season=="1819"))


eval_partition_points<-function(p, thedata){
  pwin <- p[1]
  ploss <- p[2]
  thedata$i <- 1:nrow(thedata)
  cutoffs <- quantile(thedata$i, c(ploss, 1-pwin), names=F) 
  #print(c(pwin, ploss, cutoffs))
  qqpred<-cut(thedata$i, c(-1e10, cutoffs+c(0.0, 0.00001), 1e10), labels=c("A", "D", "H"))
  qpred<-as.integer(qqpred)-2
  #gof<-cbind(qpred, sign(thedata$FTHG-thedata$FTAG))
  # cm <- table(sign(thedata$FTHG-thedata$FTAG), qpred)
  hw<-thedata$FTHG>thedata$FTAG
  aw<-thedata$FTHG<thedata$FTAG
  draw<-thedata$FTHG==thedata$FTAG
  
  tendency<-qpred==sign(thedata$FTHG-thedata$FTAG)
  gdiff<-tendency & ((abs(thedata$FTHG-thedata$FTAG)==1) | draw)
  hit<-gdiff & ((thedata$FTHG+thedata$FTAG) %in% 2:3)
  # table(tendency)
  # table(gdiff)
  # table(hit)
  pistor <- (sum(tendency)+sum(gdiff)+sum(hit))/length(qpred)
  sky <- (2*sum(tendency)+0*sum(gdiff)+3*sum(hit))/length(qpred)
  tcs <- (2*sum(tendency&hw)+1*sum(gdiff&hw)+1*sum(hit&hw)+4*sum(tendency&aw)+1*sum(gdiff&aw)+2*sum(hit&aw)+2*sum(tendency&draw)+4*sum(hit&draw))/length(qpred)
  #print(c(pistor, sky, tcs))
  #tt<-table(qpred)/length(qpred)
  return (c(cutoffs, pistor=pistor, sky=sky, tcs=tcs, hwin=sum(qpred==1)/length(qpred), draw=sum(qpred==0)/length(qpred), away=sum(qpred==-1)/length(qpred)))
}

decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  if (!is.null(model$means)) points(as.data.frame(model$means), pch=1, cex=3, lwd=2, col=2:(nlevels(p)+1L))
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}


train_seasons<-c("0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718")
test_seasons<-c("1819")

train_seasons<-c("0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819")
test_seasons<-c("0405", "0506", "0607", "0708")

train_seasons<-c("0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314")
test_seasons<-c( "1415", "1516", "1617", "1718", "1819")

train_seasons<-sample(levels(alldata$season), 8)
test_seasons<-setdiff(levels(alldata$season), train_seasons)

quote_names<-c('BWH', 'BWD', 'BWA')
#quote_names<-c('B365H', 'B365D', 'B365A')
#quote_names<-c('BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')

gridscope<-levels(alldata$season)#[-2]

gridscope<-train_seasons
traindata <- teamresults%>%filter(season %in% gridscope & where=="Home")
#traindata <- alldata%>%filter(!spieltag %in% 10:24)
#traindata <- alldata%>%filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")

# quotes <- 1/traindata[,quote_names]
# #quotes <- quotes / rowSums(quotes)
# quotes[,1:3] <- quotes[,1:3] / rowSums(quotes[,1:3])
# quotes[,4:6] <- quotes[,4:6] / rowSums(quotes[,4:6])

testdata <- teamresults%>%filter(season %in% test_seasons & where=="Home")
#testdata <- alldata%>%filter(spieltag %in% 10:24)
#testdata <- alldata%>%filter(HomeTeam=="Bayern Munich" | AwayTeam=="Bayern Munich")
# testquotes <- 1/testdata[,quote_names]
# #testquotes <- testquotes / rowSums(testquotes)
# testquotes[,1:3] <- testquotes[,1:3] / rowSums(testquotes[,1:3])
# testquotes[,4:6] <- testquotes[,4:6] / rowSums(testquotes[,4:6])
str(traindata)
colnames(traindata)
X_train <- traindata%>%dplyr::select(bwinWin:t_oppRed_where, -points, -oppPoints)
X_test <- testdata%>%dplyr::select(bwinWin:t_oppRed_where, -points, -oppPoints)
Y_train <- traindata%>%dplyr::select(GS, GC)%>%mutate(FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))
Y_test <- testdata%>%dplyr::select(GS, GC)%>%mutate(FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))

Y_train<-Y_train%>%mutate(FTX=FTR) %>%
  #mutate(FTX=factor(ifelse(GS==0 & GC==0, "0:0", as.character(FTX)))) %>%
  mutate(FTX=factor(ifelse(GS-GC==2, "2:0", as.character(FTX)))) %>%
  mutate(FTX=factor(ifelse(GS-GC>=3, "3:0", as.character(FTX)))) 
  #mutate(FTX=factor(ifelse(GC-GS==2, "0:2", as.character(FTX)))) 

Y_test<-Y_test%>%mutate(FTX=FTR) %>%
  #mutate(FTX=factor(ifelse(GS==0 & GC==0, "0:0", as.character(FTX)))) %>%
  mutate(FTX=factor(ifelse(GS-GC==2, "2:0", as.character(FTX)))) %>%
  mutate(FTX=factor(ifelse(GS-GC>=3, "3:0", as.character(FTX)))) 
  #mutate(FTX=factor(ifelse(GC-GS==2, "0:2", as.character(FTX)))) 
  
table(Y_train$FTX)
table(Y_train$FTR)

#feature_columns<-c(normalized_quote_names, "draw_prior")
#feature_columns<-normalized_quote_names[1:3]
#quotes<-traindata[, feature_columns]
#testquotes<-testdata[, feature_columns]
#pca(as.matrix(traindata))
#cor(as.matrix(X_train))
metric = c("orthonormalized", "plain", "weighted")
trans = preProcess(as.matrix(X_train), c("BoxCox", "center", "scale"))
X_train <- data.frame(trans = predict(trans, as.matrix(X_train)))
#ggplot(melt(quotes))+facet_wrap(variable~.)+geom_histogram(aes(x=value), bins=40)

X_test <- data.frame(trans = predict(trans, as.matrix(X_test)))
#ggplot(melt(testquotes))+facet_wrap(variable~.)+geom_histogram(aes(x=value), bins=40)
summary(X_test)

#library(mlogit)
#linmodel <- mlogit(FTR~., data=cbind(X_train, FTR=Y_train$FTR))

library(glmnet)
library(glmnetUtils)

require(doMC)
registerDoMC(cores=3)

library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)


model.cva.glmnet <- cva.glmnet(FTX~., family = "multinomial", data = cbind(X_train, FTX=Y_train$FTX), alpha = c(0.8, 1), 
                               parallel = TRUE, outerParallel=cl, type.measure="class")
                               #type.measure="auc", parallel = TRUE, outerParallel=cl)
stopCluster(cl)


model.cva.glmnet$alpha
#model.cva.glmnet$modlist
plot(model.cva.glmnet)
minlossplot(model.cva.glmnet, cv.type = "min") # "1se"
minlossplot(model.cva.glmnet, cv.type = "1se") # 
coef(model.cva.glmnet, alpha=1)
coef(model.cva.glmnet, alpha=0.8)


vnat=coef(model.cva.glmnet, alpha=0.8) 
vnat=coef(model.cva.glmnet, alpha=0) 
#vnat=as.matrix(vnat)[,1] 
df<-data.frame(name=names(vnat), size=abs(vnat), value=vnat) %>% filter(size>0) %>% arrange(-size)
print(df)

plot(model.cva.glmnet$modlist[[1]]) 
plot(model.cva.glmnet$modlist[[2]]) 
plot(model.cva.glmnet$modlist[[3]]) 
plot(model.cva.glmnet$modlist[[4]]) 
plot(model.cva.glmnet$modlist[[7]]) 
model.cva.glmnet$modlist[[7]]$lambda.min

lambda <- 0.01108088
lambda <- model.cva.glmnet$modlist[[2]]$lambda.min

# lambda <- 0.04075969
# lambda <- model.cva.glmnet$modlist[[2]]$lambda.1se

print(log(lambda))

lambda<-exp(-4)
model.glm <- glmnet(FTX~., family = "multinomial", data = cbind(X_train, FTX=Y_train$FTX), alpha = 1)

#model.glm <- glmnet(x = as.matrix(X_train), y=Y_train[,1:2], family = "mgaussian", alpha = 1)
#str(X_train)

print(model.glm)
vnat=coef(model.glm, s=lambda) 
print(Matrix(cbind(seq_along(t(vnat$A))-1, sapply(vnat, function(x) x[,1]))))
#unlist(vnat)
#vnat=vnat[,1] 
#df<-data.frame(name=names(vnat), size=abs(vnat), value=vnat) %>% filter(size>0) %>% arrange(-size)
#print(head(df, 200))
#selected_features <- cbind(GS=vnat$GS, GC=vnat$GC)
selected_features <- sapply(vnat, function(x) x[,1])
selected_features<-selected_features[rowSums(selected_features)!=0,]
print(Matrix(selected_features))
print(data.frame(sort(rowSums(abs(selected_features)), decreasing = T)))
selected_features<-setdiff(c(rownames(selected_features)), "(Intercept)")
#selected_features<-setdiff(c(rownames(selected_features), "trans.bwinDraw"), "(Intercept)")
print(selected_features)

#selected_features<-c("trans.bwinWin","trans.bwinLoss",   "trans.bwinDraw" , "trans.ema_oppCards") #, "trans.t_GS_where",  "trans.t_oppShotstarget"          
selected_features<-c("bwinWin","bwinLoss",   "bwinDraw")# , "ema_oppCards") #, "trans.t_GS_where",  "trans.t_oppShotstarget" 

par(mfrow=c(2,3))
plot(model.glm, label = TRUE)
plot(model.glm, xvar = "dev", label = TRUE)
plot(model.glm, xvar = "lambda", label = TRUE)
plot(model.glm$df, model.glm$dev.ratio, xlab="model parameters", ylab="%dev", main="%deviance explained", col="blue")
abline(v=length(selected_features), col="red")
par(mfrow=c(1,1))

colnames(X_train[,selected_features])

calc_glmnet_cv_scores<-function(model, lambda, X_train, Y_train){
  pred<-apply(predict(model, newdata=X_train, s=lambda, type="response"), 1, which.max)
  pred<-model.glm$classnames[pred]
  Y_train$pred<-pred
  Y_train<-Y_train %>% mutate(pFTHG=ifelse(pred %in% c('H','D','A'),  1+(pred=="H"), strtoi(substr(as.character(pred), 1, 1))),
                              pFTAG=ifelse(pred %in% c('H','D','A'),  1+(pred=="A"), strtoi(substr(as.character(pred), 3, 3))))
  
  return(c(lambda=lambda, pistor=mean(p_points(Y_train$pFTHG, Y_train$pFTAG, Y_train$GS, Y_train$GC)$pistor),
           sky=mean(p_points(Y_train$pFTHG, Y_train$pFTAG, Y_train$GS, Y_train$GC)$sky),
         home=mean(pred %in% c("H","2:0")),
         draw=mean(pred %in% c("D","0:0")),
         away=mean(pred %in% c("A","0:2"))))
}

calc_glmnet_cv_scores(model.glm, lambda=0.001, X_train, Y_train)
calc_glmnet_cv_scores(model.glm, lambda=0.001, X_test, Y_test)

cvtrain<-t(sapply(seq(-11, -2.5, 0.1), function(l) calc_glmnet_cv_scores(model.glm, lambda=exp(l), X_train, Y_train)))
cvtest<-t(sapply(seq(-11, -2.5, 0.1), function(l) calc_glmnet_cv_scores(model.glm, lambda=exp(l), X_test, Y_test)))

library(reshape2)
dd <- melt(data.frame(cvtrain, cvtest[,-1]), id=c("lambda"))
ggplot(dd) + geom_line(aes(x=log(lambda), y=value, colour=variable), lwd=1)

print(data.frame(train=cvtrain, test=cvtest[,-1]))

data.frame(train=cvtrain, test=cvtest[,-1])%>%mutate(loglambda=log(train.lambda))%>%top_n(n = 10, wt = test.pistor)
data.frame(train=cvtrain, test=cvtest[,-1])%>%mutate(loglambda=log(train.lambda))%>%top_n(n = 10, wt = test.sky)
print(as.data.frame(cvtrain)%>%summarise(max.pistor=max(pistor), max.sky=max(sky)))
print(as.data.frame(cvtest)%>%summarise(max.pistor=max(pistor), max.sky=max(sky)))

lambda<-exp(-6.2)
lambda<-exp(-5.4)
pred<-apply(predict(model.glm, newdata=X_train, s=lambda, type="response"), 1, which.max)
pred<-model.glm$classnames[pred]
table(pred, Y_train$FTX)
mean(pred == Y_train$FTX)

testpred<-apply(predict(model.glm, newdata=X_test, s=lambda, type="response"), 1, which.max)
testpred<-model.glm$classnames[testpred]
table(testpred, Y_test$FTX)
mean(testpred == Y_test$FTX)

Y_train$pred<-pred
Y_train<-Y_train %>% mutate(pFTHG=ifelse(pred %in% c('H','D','A'),  1+(pred=="H"), strtoi(substr(as.character(pred), 1, 1))),
                            pFTAG=ifelse(pred %in% c('H','D','A'),  1+(pred=="A"), strtoi(substr(as.character(pred), 3, 3))))

Y_test$pred<-testpred
Y_test<-Y_test %>% mutate(pFTHG=ifelse(pred %in% c('H','D','A'),  1+(pred=="H"), strtoi(substr(as.character(pred), 1, 1))),
                          pFTAG=ifelse(pred %in% c('H','D','A'),  1+(pred=="A"), strtoi(substr(as.character(pred), 3, 3))))

mean(p_points(Y_train$pFTHG, Y_train$pFTAG, Y_train$GS, Y_train$GC)$pistor)
mean(p_points(Y_train$pFTHG, Y_train$pFTAG, Y_train$GS, Y_train$GC)$sky)

mean(p_points(Y_test$pFTHG, Y_test$pFTAG, Y_test$GS, Y_test$GC)$pistor)
mean(p_points(Y_test$pFTHG, Y_test$pFTAG, Y_test$GS, Y_test$GC)$sky)





model <- lfda(as.matrix(X_train[,selected_features]), Y_train$FTR, r=1,  metric = metric, knn = 20)

#model <- lfda(as.matrix(X_train), Y_train$FTR, r=1,  metric = c(), knn = 3)

rownames(model$T)<-selected_features
print(model$T)
print(model)
summary(X_train[,selected_features])

plot(model$Z, col=as.integer(Y_train$FTR)+1)
points(model$Z, col=pred)
plot(model$Z[,c(1,3)], col=as.integer(Y_train$FTR)+1)
plot(model$Z[,c(2,3)], col=as.integer(Y_train$FTR)+1)
plot(model$Z[,c(1,4)], col=as.integer(Y_train$FTR)+1)

pred<-apply(predict(model, newdata=as.matrix(X_train[,selected_features])), 1, which.max)
pred<-(ifelse(pred==1, "A", ifelse(pred==2, "H", "D")))
table(pred, Y_train$FTR)
mean(pred == Y_train$FTR)


ldamodel <- lda(FTR ~ ., data=cbind(FTR=Y_train$FTR, X_train[,selected_features]), CV = T)
ldamodel <- lda(FTR ~ ., data=cbind(FTR=Y_train$FTR, X_train), CV = T)
ldamodel <- lda(FTX ~ ., data=cbind(FTX=Y_train$FTX, X_train[,selected_features]), CV = T)
ldamodel <- lda(FTX ~ ., data=cbind(FTX=Y_train$FTX, X_train), CV = T)

table(ldamodel$class, Y_train$FTX)/length(ldamodel$class)*100
mean(ldamodel$class == Y_train$FTX)

ldamodel <- lda(FTR ~ ., data=cbind(FTR=Y_train$FTR, X_train[,selected_features]), CV = F)
ldamodel <- lda(FTR ~ ., data=cbind(FTR=Y_train$FTR, X_train), CV = F)
ldamodel <- lda(FTX ~ ., data=cbind(FTX=Y_train$FTX, X_train[,selected_features]), CV = F)
#ldamodel <- lda(FTX ~ ., data=cbind(FTX=Y_train$FTX, X_train), CV = F)
plda <- predict(ldamodel, newdata = X_train)
prop <- ldamodel$svd^2/sum(ldamodel$svd^2)
print(prop)

dataset <- data.frame(FTR = Y_train$FTR, lda = plda$x, pred=predict(ldamodel, newdata=X_train)$class)
dataset <- data.frame(FTX = Y_train$FTX, lda = plda$x, pred=predict(ldamodel, newdata=X_train)$class)
centr <- predict(ldamodel, newdata = data.frame(ldamodel$means))

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = FTX, shape=pred), size = 1.5, alpha=0.2) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  scale_colour_brewer(palette = "Set1")+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTX), size=10, pch=4)+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTX), size=4)

ggplot(dataset) + geom_point(aes(lda.LD3, lda.LD4, colour = FTX, shape=pred), size = 1.5, alpha=0.2) + 
  labs(x = paste("LD3 (", (prop[3]), ")", sep=""),
       y = paste("LD4 (", (prop[4]), ")", sep=""))+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD3, y=LD4, colour=FTX), size=10, pch=4)+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD3, y=LD4, colour=FTX), size=4)

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = pred, shape=FTX), size = 1.5, alpha=0.2) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  scale_colour_brewer(palette = "Spectral")+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTX), size=10, pch=4)+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTX), size=4)
  
ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD3, colour = pred, shape=FTX), size = 1.5, alpha=0.2) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD3 (", (prop[3]), ")", sep=""))+
  scale_colour_brewer(palette = "Spectral")+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD1, y=LD3, colour=FTX), size=10, pch=4)+
  geom_point(data=data.frame(centr$x, FTX=rownames(centr$x)), aes(x=LD1, y=LD3, colour=FTX), size=4)

pred<-predict(ldamodel, newdata=X_train)$class
table(pred, Y_train$FTX)/length(Y_train$FTX)*100
mean(pred == Y_train$FTX)
plot(pred, Y_train$FTX)

testpred<-predict(ldamodel, newdata=X_test)$class
table(testpred, Y_test$FTX)/length(Y_test$FTX)*100
mean(testpred == Y_test$FTX)
plot(testpred, Y_test$FTX)

Y_train$pred<-pred
Y_train<-Y_train %>% mutate(pFTHG=ifelse(pred %in% c('H','D','A'),  1+(pred=="H"), strtoi(substr(as.character(pred), 1, 1))),
                            pFTAG=ifelse(pred %in% c('H','D','A'),  1+(pred=="A"), strtoi(substr(as.character(pred), 3, 3))))
    
Y_test$pred<-testpred
Y_test<-Y_test %>% mutate(pFTHG=ifelse(pred %in% c('H','D','A'),  1+(pred=="H"), strtoi(substr(as.character(pred), 1, 1))),
                            pFTAG=ifelse(pred %in% c('H','D','A'),  1+(pred=="A"), strtoi(substr(as.character(pred), 3, 3))))

mean(p_points(Y_train$pFTHG, Y_train$pFTAG, Y_train$GS, Y_train$GC)$pistor)
mean(p_points(Y_train$pFTHG, Y_train$pFTAG, Y_train$GS, Y_train$GC)$sky)

mean(p_points(Y_test$pFTHG, Y_test$pFTAG, Y_test$GS, Y_test$GC)$pistor)
mean(p_points(Y_test$pFTHG, Y_test$pFTAG, Y_test$GS, Y_test$GC)$sky)

# plot(model$Z, col=as.integer(thedata$FTR)+1)
# plot(model$Z[,2:3], col=as.integer(thedata$FTR)+1)
# plot(model$Z[,c(1,3)], col=as.integer(thedata$FTR)+1)
# ggplot(data.frame(model$Z, FTR=thedata$FTR), aes(x=X1, fill=FTR))+geom_density(alpha=0.4) #+geom_histogram()


###########################################################################################################

#selected_features<-c("trans.bwinWin","trans.bwinLoss",   "trans.bwinDraw" , "trans.ema_oppCards") #, "trans.t_GS_where",  "trans.t_oppShotstarget"          
selected_features<-c("bwinWin","bwinLoss",   "bwinDraw", "ema_oppCards") #, "trans.t_GS_where",  "trans.t_oppShotstarget" 

trans = preProcess(as.matrix(X_train), c("BoxCox", "center", "scale"))
XX_train <- data.frame(trans = predict(trans, as.matrix(X_train)))
XX_test <- data.frame(trans = predict(trans, as.matrix(X_test)))

Xselected_features<-paste0("trans.",selected_features)
model <- lfda(as.matrix(XX_train[,Xselected_features]), Y_train$FTR, r=1,  metric = metric, knn = 20)
rownames(model$T)<-Xselected_features
print(model$T)

traindata<-traindata%>%dplyr::select(-dplyr::matches("X[0-9]"))
traindata<-data.frame(traindata, X1=predict(model, XX_train[,Xselected_features]))
testdata<-testdata%>%dplyr::select(-dplyr::matches("X[0-9]"))
testdata<-data.frame(testdata, X1=predict(model, XX_test[,Xselected_features]))




model <- lfda(as.matrix(X_train[,selected_features]), Y_train$FTR, r=1,  metric = metric, knn = 20)

#model <- lfda(as.matrix(X_train), Y_train$FTR, r=1,  metric = c(), knn = 3)

rownames(model$T)<-selected_features
print(model$T)

traindata<-traindata%>%dplyr::select(-dplyr::matches("X[0-9]"))
traindata<-data.frame(traindata, X1=predict(model, X_train[,selected_features]))
testdata<-testdata%>%dplyr::select(-dplyr::matches("X[0-9]"))
testdata<-data.frame(testdata, X1=predict(model, X_test[,selected_features]))

traindata <- traindata%>%mutate(FTHG=GS, FTAG=GC, FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))
testdata <- testdata%>%mutate(FTHG=GS, FTAG=GC, FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))

# move HomeWins to high end of scale
#orientation<-traindata%>%group_by(FTR)%>%summarise(X1=median(X1), X2=median(X2), X3=median(X3))%>%filter(FTR %in% c("H", "A"))%>%mutate_at(vars(X1:X3), rank)%>%mutate_at(vars(X1:X3), function(x) 2*(x-1.5))%>%filter(FTR=="H")
orientation<-traindata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A"))%>%mutate_at(vars(X1), rank)%>%mutate_at(vars(X1), function(x) 2*(x-1.5))%>%filter(FTR=="H")
print(orientation)
traindata$X1<-traindata$X1*orientation$X1
testdata$X1<-testdata$X1*orientation$X1
# traindata$X2<-traindata$X2*orientation$X2
# traindata$X3<-traindata$X3*orientation$X3
# testdata$X2<-testdata$X2*orientation$X2
# testdata$X3<-testdata$X3*orientation$X3

q<-prepare_plot_data_lfda(traindata)
qtest<-prepare_plot_data_lfda(testdata)

print(q%>%top_n(1, pistor))
print(q%>%top_n(1, sky))
print(qtest%>%top_n(1, pistor))
print(qtest%>%top_n(1, sky))


#traindata$ema_oppCards

print(traindata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A")))
print(testdata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A")))
ggplot(traindata, aes(x=X1, fill=FTR, group=FTR))+facet_grid(FTR~.)+geom_histogram(bins=40)
plot(FTR~X1, data=traindata, main="Train Data")
plot(FTR~X1, data=testdata, main="Test Data")
#ggplot(traindata, aes(x=X1, fill=FTR, group=FTR))+geom_histogram(bins=40)

ggplot(traindata, aes(y=X1, color=FTR)) +geom_boxplot()+ coord_flip()
ggplot(traindata, aes(x = FTR, y=X1, color=FTR)) + geom_violin(draw_quantiles = c(0.1, 0.25, 0.5, 0.75, 0.9), trim=F)+ coord_flip()
# ggplot(traindata, aes(x=X3, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=40)
# ggplot(traindata, aes(x=X1, y=X2, colour=FTR))+geom_point(alpha=0.4)
# ggplot(traindata, aes(x=X1, y=X3, colour=FTR))+geom_point(alpha=0.4)

# plot(FTR~X2, data=traindata, main="Train Data")
# plot(FTR~X2, data=testdata, main="Test Data")
# plot(FTR~X3, data=traindata, main="Train Data")
# plot(FTR~X3, data=testdata, main="Test Data")


ggplot(testdata, aes(x=X1, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=20)
ggplot(testdata, aes(x = FTR, y=X1, color=FTR)) + geom_violin(draw_quantiles = c(0.1, 0.25, 0.5, 0.75, 0.9), trim=F)+ coord_flip()
# ggplot(testdata, aes(x=X2, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=20)
# ggplot(testdata, aes(x=X3, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=20)
# ggplot(testdata, aes(x=X1, y=X2, colour=FTR))+geom_point(alpha=0.4)
# ggplot(testdata, aes(x=X1, y=X3, colour=FTR))+geom_point(alpha=0.4)

#ggplot(alldata, aes(shape=FTR, y=log(BWA), x=log(BWH), col=-log(BWD)))+scale_colour_gradientn(colours = rev(rainbow(10)))+geom_point(alpha=0.2)

p_points<-function(pHG, pAG, FTHG, FTAG){
  hw<-FTHG>FTAG
  aw<-FTHG<FTAG
  draw<-FTHG==FTAG
  
  tendency<-sign(pHG-pAG)==sign(FTHG-FTAG)
  gdiff<-(pHG-pAG)==(FTHG-FTAG)
  hit<-gdiff & (pHG==FTHG)
  pistor <- tendency+gdiff+hit
  sky <- 2*tendency+3*hit
  return (list(pistor=pistor, sky=sky))
}

###############

point_system<-"pistor"
#point_system<-"sky"


newdata<-matrix(ncol=3, byrow = T, 
                c(2.2, 3.4, 3.3,
                  3.3, 3.4, 2.2,
                  2.35, 3.6, 2.85,
                  1.8, 3.7, 4.5,
                  1.9, 3.6, 4.1,
                  1.36, 5.25, 8.25,
                  1.53, 4.6, 5.5,
                  3.7, 4.25, 1.85,
                  2.1, 3.6, 3.4
                ))
colnames(newdata)<-feature_columns

newdatafile<-"D:/gitrepository/Football/football/TF/quotes_bwin.csv"
#newdatafile<-"quotes_bwin.csv"
newdata_df<-read.csv(newdatafile, sep = ",", encoding = "utf-8")

newdata<-as.matrix(newdata_df[, quote_names])
rownames(newdata)<-paste(newdata_df$HomeTeam, newdata_df$AwayTeam, sep="-")
colnames(newdata)<-feature_columns

newquotes <- 1/newdata
print(newquotes)
print(rowSums(newquotes))
#newquotes <- newquotes / rowSums(newquotes)
newquotes[,1:3] <- newquotes[,1:3] / rowSums(newquotes[,1:3])
#newquotes[,4:6] <- newquotes[,4:6] / rowSums(newquotes[,4:6])

if (F){
  newquotes<-as.matrix(tail(alldata%>%dplyr::select(feature_columns), 9))
}  


newquotes <- data.frame(trans = predict(trans, newquotes))

# newdata<-data.frame(X1=predict(model, newquotes))
# newdata$X1<-newdata$X1*orientation$X1

newdata<-data.frame(X1=predict(model, newquotes))
newdata$X1<-newdata$X1*orientation$X1
# newdata$X2<-newdata$X2*orientation$X2
# newdata$X3<-newdata$X3*orientation$X3

l <- nrow(traindata)
plot(seq_along(traindata$X1)/l, sort(traindata$X1), axes=F, col="forestgreen", pch=".", type="l", ylim=range(traindata$X1)*1.3)
axis(1,at=seq(0,1,0.05),labels=T)
axis(2,labels=T)

perc<-sapply(newdata$X1, function(x) sum(traindata$X1 < x))
abline(v=perc/l, col="black", lty=3)
print(perc/l)
print(1-perc/l)

best<-q %>% arrange_(.dots=paste0("desc(", point_system, ")")) %>% head(20)

segments(x0 = best$away, x1 = (best$draw+best$away), 
         y0=quantile(traindata$X1, best$away) , 
         y1=quantile(traindata$X1, best$draw+best$away), col="darkolivegreen1")
segments(x0 = best$away, x1 = (best$draw+best$away), 
         y0=quantile(traindata$X1, best$away) , 
         y1=quantile(traindata$X1, best$away), col="darkolivegreen1")
points(x = (best$draw+best$away), y=quantile(traindata$X1, best$away), col="darkolivegreen1")

ltest <- nrow(testdata)
points(seq_along(testdata$X1)/ltest, sort(testdata$X1), col="tan2", type="l", lwd=2)

testperc<-sapply(newdata$X1, function(x) sum(testdata$X1 < x))
abline(v=testperc/ltest, col="yellowgreen", lty=3)

print(testperc/ltest)
print(1-testperc/ltest)
testbest<-qtest %>% arrange_(.dots=paste0("desc(", point_system, ")")) %>% head(20)

segments(x0 = testbest$away, x1 = (testbest$draw+testbest$away), 
         y0=quantile(testdata$X1, testbest$away) , 
         y1=quantile(testdata$X1, testbest$draw+testbest$away), col="orange")
segments(x0 = testbest$away, x1 = (testbest$draw+testbest$away), 
         y0=quantile(testdata$X1, testbest$draw+testbest$away) , 
         y1=quantile(testdata$X1, testbest$draw+testbest$away), col="orange")
points(x = testbest$away, y=quantile(testdata$X1, testbest$draw+testbest$away), col="orange")

text(perc/l, newdata$X1+0.3, 1:nrow(newdata), col="red")
text(testperc/ltest, newdata$X1+0.3, 1:nrow(newdata), col="blue")
points(perc/l, newdata$X1, col="red", lwd=2)
points(testperc/ltest, newdata$X1, col="red", lwd=2)
text(perc/l, newdata$X1+0.5, rownames(newdata), col="red", cex=0.5)


trainseq<-traindata %>% arrange(X1)
testseq<-testdata %>% arrange(X1)

best1<-q %>% arrange_(.dots=paste0("desc(", point_system, ")")) %>% head(1)
testbest1<-qtest %>% arrange_(.dots=paste0("desc(", point_system, ")")) %>% head(1)

make_cumsum<-function(x, q, system="pistor"){
  best1<-q %>% arrange_(.dots=paste0("desc(", system, ")")) %>% head(1)
  print(best1)
  
  x <- x%>%mutate(p12=p_points(1,2,FTHG, FTAG)[[system]], 
                  p11=p_points(1,1,FTHG, FTAG)[[system]], 
                  p21=p_points(2,1,FTHG, FTAG)[[system]],
                  p02=p_points(0,2,FTHG, FTAG)[[system]], 
                  p20=p_points(2,0,FTHG, FTAG)[[system]])
  x$q <- seq_along(x$FTR)/nrow(x)
  x <- x%>%mutate(ep = ifelse(q>=1-best1$hwin, p21, ifelse(q<=best1$away, p12, p11)))
  
  cp1base <- (x%>%mutate(bp = ifelse(q>=1-best1$hwin, p21, p11))%>%summarise(bp=mean(bp)))$bp[1]
  x <- x%>%mutate(cp1 = cumsum(p12-p11)/nrow(x)+cp1base)
  cp2base <- (x%>%mutate(bp = ifelse(q<=best1$away, p12, p11))%>%summarise(bp=mean(bp)))$bp[1]
  x <- x%>%mutate(cp2 = rev(cumsum(rev(p21-p11)))/nrow(x)+cp2base)
  
  cp20base <- (x %>% summarise(ep = mean(ep)))$ep
  x <- x%>%mutate(cp20 = rev(cumsum(rev(p20-p21)))/nrow(x)+cp20base)
  x <- x%>%mutate(cp02 = cumsum(p02-p12)/nrow(x)+cp20base)
  return (x)  
}

trainseq<-make_cumsum(trainseq, q, point_system)
testseq<-make_cumsum(testseq, qtest, point_system)

par(new = T)
yrange <- rbind(trainseq, testseq) %>% summarise(min=min(cp1, cp2, cp20), max=max(cp1, cp2, cp20))
plot(cp1~q, data = testseq, type="l", axes=F, xlab=NA, ylab=NA, col="tomato", lwd=2, ylim=c((yrange$min+yrange$max)/2, yrange$max))
points(cp2~q, data = testseq, type="l", col="cornflowerblue", lwd=2)
points(cp20~q, data = testseq, type="l", col="darkgreen", lwd=2, lty=3)
points(cp02~q, data = testseq, type="l", col="darkgray", lwd=2, lty=2)
abline(h=testbest1[,point_system], col="tan3", lwd=2)

points(cp1~q, data = trainseq, type="l", xlab=NA, ylab=NA, col="red")
points(cp2~q, data = trainseq, type="l", col="blue")
points(cp20~q, data = trainseq, type="l", col="seagreen", lty=3)
points(cp02~q, data = trainseq, type="l", col="gainsboro", lty=2)
abline(h=best1[,point_system], col="forestgreen")

axis(side = 4, at=seq(0.5,2.0,0.01))
abline(v=best1$away, col="red")
abline(v=1-best1$hwin, col="blue")
abline(v=testbest1$away, col="tomato", lwd=2)
abline(v=1-testbest1$hwin, col="cornflowerblue", lwd=2)

trainseq%>%summarise(ep=mean(ep), cp1=max(cp1), cp2=max(cp2), cp20=max(cp20), cp02=max(cp02))
testseq%>%summarise(ep=mean(ep), cp1=max(cp1), cp2=max(cp2), cp20=max(cp20), cp02=max(cp02))

print(data.frame(newdata, train=perc/l, test=testperc/ltest))







ggplot(q, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  #geom_tile()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-pistor) %>% head(10)

ggplot(qtest, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  #geom_tile()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
qtest %>% arrange(-pistor) %>% head(10)

ggplot(q, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-sky) %>% head(10)

ggplot(qtest, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
#  stat_dens2d_filter(color = "red", keep.fraction = 0.05)
#+stat_peaks(col = "black", span = 5, strict = T, geom = "text")
qtest %>% arrange(-sky) %>% head(20)




#q<-prepare_plot_data2(alldata)


colist <- q %>% arrange(-pistor) %>% head(20) %>% dplyr::select(hwin, away)
q2<-apply(colist, 1, eval_partition_points, testdata%>%arrange(X1))
q2 <- data.frame(t(q2))
ggplot(q2, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor, label=round(pistor, 4)))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_text(size=3, col="black", nudge_y = 0.002, check_overlap = T)

ggplot(q, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_tile(alpha=0.2)+
  geom_point(data=qtest)+
  geom_contour(binwidth=0.01, size=1)+
  geom_text_contour(binwidth=0.01, min.size = 10, size=6, colour="blue")+
  geom_contour(data=qtest, binwidth=0.02, col="red", size=1)+
  geom_text_contour(data=qtest, binwidth=0.02, min.size = 10, colour="red", size=6)+
  geom_point(data=q2, colour="black", size=1)+
  geom_text(data=q2, size=3, colour="black", check_overlap = F, show.legend=F, aes(label=round(pistor, 3)), nudge_y = 0.004)

q %>% arrange(-pistor) %>% head(20) %>% data.frame(q2) %>% dplyr::select(-V1, -V2)


colist <- q %>% arrange(-sky) %>% head(20) %>% dplyr::select(hwin, away)
q2<-apply(colist, 1, eval_partition_points, testdata%>%arrange(X1))
q2 <- data.frame(t(q2))
ggplot(q2, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky, label=round(sky, 4)))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_text(size=3, col="black", nudge_y = 0.002, check_overlap = T)

ggplot(q, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_tile(alpha=0.2)+
  geom_point(data=qtest)+
  geom_contour(binwidth=0.01, size=1)+
  geom_text_contour(binwidth=0.01, min.size = 10, size=6, colour="blue")+
  geom_contour(data=qtest, binwidth=0.02, col="red", size=1)+
  geom_text_contour(data=qtest, binwidth=0.02, min.size = 10, colour="red", size=6)+
  geom_point(data=q2, colour="black", size=1)+
  geom_text(data=q2, size=3, colour="black", check_overlap = F, show.legend=F, aes(label=round(sky, 3)), nudge_y = 0.004)

q %>% arrange(-sky) %>% head(20) %>% data.frame(q2) %>% dplyr::select(-V1, -V2)





seasonsq <- lapply(setdiff(unique(traindata$season),"0506"), function(x) cbind(season=x, prepare_plot_data_lfda(traindata%>%filter(season==x), step=0.02)))
seasonsq <- do.call(rbind, seasonsq)
seasonsq%>%group_by(season)%>%summarise(pistor=max(pistor), sky=max(sky), tcs=max(tcs))

ggplot(seasonsq, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.03, min.size = 5, size=4)+
  geom_point(size=2, colour="black", data=seasonsq%>%group_by(season)%>%top_n(1, pistor))
print.data.frame(seasonsq%>%group_by(season)%>%top_n(1, pistor))

ggplot(seasonsq, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.03, min.size = 10, size=4)+
  geom_point(size=2, colour="black", data=seasonsq%>%group_by(season)%>%top_n(1, sky))
print.data.frame(seasonsq%>%group_by(season)%>%top_n(1, sky))



##############################################################################################################################





























