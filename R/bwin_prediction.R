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
library(lubridate)
library(tidyr)
library(TTR)

newdatafile<-"D:/gitrepository/Football/football/TF/quotes_bwin.csv"
#newdatafile<-"quotes_bwin.csv"
newdata_df<-read.csv(newdatafile, sep = ",", encoding = "utf-8")
newdata_df$Date<-dmy(newdata_df$Date)
newdata<-newdata_df[, c('HomeTeam', 'AwayTeam','Date', 'BWH','BWD','BWA')]
newdata$isnew<-T


fetch_data<-function(season){
  url <- paste0("http://www.football-data.co.uk/mmz4281/", season, "/D1.csv")
  inputFile <- paste0("BL",season,".csv")
  #download.file(url, inputFile, method = "libcurl")
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
  results$Date<-dmy(results$Date) #as.Date(results$Date, "%d/%m/%y")
  results$dayofweek<-weekdays(results$Date)
  results$gameindex<-(0:(nrow(results)-1))%%9+1
  results$isnew<-F
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

newdata$HomeTeam<-factor(levels(alldata$HomeTeam)[apply(adist(gsub('FC |FSV |Borussia ', '', newdata$HomeTeam), levels(alldata$HomeTeam)), 1, which.min)], levels = levels(alldata$HomeTeam))
newdata$AwayTeam<-factor(levels(alldata$HomeTeam)[apply(adist(gsub('FC |FSV |Borussia ', '', newdata$AwayTeam), levels(alldata$HomeTeam)), 1, which.min)], levels = levels(alldata$AwayTeam))

alldata<-bind_rows(alldata, as.data.frame(newdata))

table(alldata$season, useNA = "ifany")
alldata$season[is.na(alldata$season)] <- max(as.character(alldata$season), na.rm = T)
alldata$spieltag[is.na(alldata$spieltag)] <- max(alldata$spieltag[alldata$season==max(as.character(alldata$season))], na.rm = T)+1
alldata$round<-alldata$spieltag

alldata <- alldata %>%
  mutate(FTHG = replace_na(FTHG, 0),
         FTAG = replace_na(FTAG, 0),
         HTHG = replace_na(HTHG, 0),
         HTAG = replace_na(HTAG, 0),
         HS = replace_na(HS, 0),
         AS = replace_na(AS, 0),
         HST = replace_na(HST, 0),
         AST = replace_na(AST, 0),
         HF = replace_na(HF, 0),
         AF = replace_na(AF, 0),
         HC = replace_na(HC, 0),
         AC = replace_na(AC, 0),
         HY = replace_na(HY, 0),
         AY = replace_na(AY, 0),
         HR = replace_na(HR, 0),
         AR = replace_na(AR, 0)
  )

quote_names<-c('BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')
quote_names<-c('BWH', 'BWD', 'BWA')
normalized_quote_names<-paste0("p", quote_names)

alldata[,normalized_quote_names]<-1/alldata[,quote_names]
alldata[,normalized_quote_names[1:3]]<-alldata[,normalized_quote_names[1:3]]/rowSums(alldata[,normalized_quote_names[1:3]])


#########################################################################################################################

tail(alldata, 18)


teams <- unique(alldata$HomeTeam)
results <- alldata[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'round', 'Date', 'dayofweek', 'HS', 'AS', 'HST', 'AST', 
                      'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR', normalized_quote_names, 'isnew' )]
results$season<-as.factor(results$season)
table(results$FTR , results$season)
results$gameindex<-(0:(nrow(results)-1))%%9+1
teamresults <- data.frame(team=results$HomeTeam, oppTeam=results$AwayTeam, 
                          GS=results$FTHG, GC=results$FTAG, where="Home", round=results$round, season=results$season, 
                          GS1H=results$HTHG, GC1H=results$HTAG, GS2H=results$FTHG-results$HTHG, GC2H=results$FTAG-results$HTAG, 
                          dow=results$dayofweek, gameindex=results$gameindex,
                          Shots=results$HS, Shotstarget=results$HST, 
                          Fouls=results$HF, Corners=results$HC, Yellow=results$HY, Red=results$HR,
                          oppShots=results$AS, oppShotstarget=results$AST, 
                          oppFouls=results$AF, oppCorners=results$AC, oppYellow=results$AY, oppRed=results$AR,
                          bwinWin=results$pBWH, bwinLoss=results$pBWA, bwinDraw=results$pBWD, isnew=results$isnew)
teamresults <- rbind(data.frame(team=results$AwayTeam, oppTeam=results$HomeTeam, 
                                GS=results$FTAG, GC=results$FTHG, where="Away", round=results$round, season=results$season,
                                GS1H=results$HTAG, GC1H=results$HTHG, GS2H=results$FTAG-results$HTAG, GC2H=results$FTHG-results$HTHG, 
                                dow=results$dayofweek, gameindex=results$gameindex,
                                Shots=results$AS, Shotstarget=results$AST, 
                                Fouls=results$AF, Corners=results$AC, Yellow=results$AY, Red=results$AR,
                                oppShots=results$HS, oppShotstarget=results$HST, 
                                oppFouls=results$HF, oppCorners=results$HC, oppYellow=results$HY, oppRed=results$HR,
                                bwinWin=results$pBWA, bwinLoss=results$pBWH, bwinDraw=results$pBWD, isnew=results$isnew),
                     teamresults)

teamresults<-teamresults[order(teamresults$season, teamresults$round, teamresults$team),]
#unique(teamresults$season)
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

teamresults<-teamresults[order(teamresults$season, teamresults$round, teamresults$gameindex, teamresults$where),]

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
traindata <- teamresults%>%filter(season %in% gridscope & where=="Home" & !isnew)
#traindata <- teamresults%>%filter(round %in% 10:24)
#traindata <- alldata%>%filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")

# quotes <- 1/traindata[,quote_names]
# #quotes <- quotes / rowSums(quotes)
# quotes[,1:3] <- quotes[,1:3] / rowSums(quotes[,1:3])
# quotes[,4:6] <- quotes[,4:6] / rowSums(quotes[,4:6])

testdata <- teamresults%>%filter(season %in% test_seasons & where=="Home" & !isnew)
newdata <- teamresults%>%filter(where=="Home" & isnew)
#testdata <- teamresults%>%filter(!round %in% 10:24)
#testdata <- alldata%>%filter(HomeTeam=="Bayern Munich" | AwayTeam=="Bayern Munich")
# testquotes <- 1/testdata[,quote_names]
# #testquotes <- testquotes / rowSums(testquotes)
# testquotes[,1:3] <- testquotes[,1:3] / rowSums(testquotes[,1:3])
# testquotes[,4:6] <- testquotes[,4:6] / rowSums(testquotes[,4:6])

X_train <- traindata%>%dplyr::select(bwinWin:t_diffBothGoals2H_where, -points, -oppPoints, -isnew)
Y_train <- traindata%>%dplyr::select(GS, GC)%>%mutate(FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))
X_test <- testdata%>%dplyr::select(bwinWin:t_diffBothGoals2H_where, -points, -oppPoints, -isnew)
Y_test <- testdata%>%dplyr::select(GS, GC)%>%mutate(FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))
X_new <- newdata%>%dplyr::select(bwinWin:t_diffBothGoals2H_where, -points, -oppPoints, -isnew)

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


###########################################################################################################

# 
# bwinWin                                                     7.9081507282
# (Intercept)                                                 5.2514975068
# bwinDraw                                                    2.5592698929
# bwinLoss                                                    0.3163580460
# ema_oppCards                                                0.2934007279
# t_GS2H_where                                                0.0798846793
# t_Shots                                                     0.0237800272
# ema_Cards                                                   0.0235225730
# t_GC2H                                                      0.0186090863
# t_oppShots_where                                            0.0065582804
# t_GS_where                                                  0.0040227806
# t_oppCorners_where                                          0.0008940306



# bwinWin                                                     2.600898e+00
# bwinLoss                                                    2.394924e+00
# bwinDraw                                                    2.090361e+00
# (Intercept)                                                 4.109418e-01
# ema_oppCards                                                3.369446e-01
# ema_Cards                                                   6.952404e-02
# t_GS_where                                                  6.407113e-02
# t_GC2H                                                      5.098452e-03
# t_oppShots_where                                            4.839730e-03
# t_GS2H_where                                                4.096011e-03
# t_Rank                                                      7.381224e-04
# t_oppCorners_where                                          1.753280e-04
# t_diffBothGoals                                             3.228686e-05
# 

#selected_features<-c("trans.bwinWin","trans.bwinLoss",   "trans.bwinDraw" , "trans.ema_oppCards") #, "trans.t_GS_where",  "trans.t_oppShotstarget"          
selected_features<-c("trans.bwinWin","trans.bwinLoss",   "trans.bwinDraw", "trans.ema_oppCards", "trans.ema_Cards")#, "trans.t_GS2H_where", "trans.t_Shots") #, "trans.t_GS_where",  "trans.t_oppShotstarget" 

trans = preProcess(as.matrix(X_train), c("BoxCox", "center", "scale"))
XX_train <- data.frame(trans = predict(trans, as.matrix(X_train)))
XX_test <- data.frame(trans = predict(trans, as.matrix(X_test)))
XX_new <- data.frame(trans = predict(trans, as.matrix(X_new)))

X2_train<-cbind(X_train, XX_train)
X2_test<-cbind(X_test, XX_test)
X2_new<-cbind(X_new, XX_new)

model <- lfda(X2_train[,selected_features], Y_train$FTR, r=1,  metric = metric, knn = 20)

rownames(model$T)<-selected_features
print(model$T)

traindata<-traindata%>%dplyr::select(-dplyr::matches("X[0-9]"))
traindata<-data.frame(traindata, X1=predict(model, X2_train[,selected_features]))
testdata<-testdata%>%dplyr::select(-dplyr::matches("X[0-9]"))
testdata<-data.frame(testdata, X1=predict(model, X2_test[,selected_features]))
newdata<-newdata%>%dplyr::select(-dplyr::matches("X[0-9]"))
newdata<-data.frame(newdata, X1=predict(model, X2_new[,selected_features]))


traindata <- traindata%>%mutate(FTHG=GS, FTAG=GC, FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))
testdata <- testdata%>%mutate(FTHG=GS, FTAG=GC, FTR=factor(ifelse(GS>GC, "H", ifelse(GS<GC, "A", "D"))))

# move HomeWins to high end of scale
#orientation<-traindata%>%group_by(FTR)%>%summarise(X1=median(X1), X2=median(X2), X3=median(X3))%>%filter(FTR %in% c("H", "A"))%>%mutate_at(vars(X1:X3), rank)%>%mutate_at(vars(X1:X3), function(x) 2*(x-1.5))%>%filter(FTR=="H")
orientation<-traindata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A"))%>%mutate_at(vars(X1), rank)%>%mutate_at(vars(X1), function(x) 2*(x-1.5))%>%filter(FTR=="H")
print(orientation)
traindata$X1<-traindata$X1*orientation$X1
testdata$X1<-testdata$X1*orientation$X1
newdata$X1<-newdata$X1*orientation$X1
# traindata$X2<-traindata$X2*orientation$X2
# traindata$X3<-traindata$X3*orientation$X3
# testdata$X2<-testdata$X2*orientation$X2
# testdata$X3<-testdata$X3*orientation$X3

plot(FTR~X1, data=traindata, main="Train Data")
plot(FTR~X1, data=testdata, main="Test Data")

q<-prepare_plot_data_lfda(traindata)
qtest<-prepare_plot_data_lfda(testdata)

print(q%>%top_n(1, pistor))
print(q%>%top_n(1, sky))
print(qtest%>%top_n(1, pistor))
print(qtest%>%top_n(1, sky))

qqplot(q$pistor, qtest$pistor)
qqplot(q$sky, qtest$sky)
#traindata$ema_oppCards
plot(q$pistor, qtest$pistor)
plot(q$sky, qtest$sky)


print(traindata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A")))
print(testdata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A")))
if (F) {
  ggplot(traindata, aes(x=X1, fill=FTR, group=FTR))+facet_grid(FTR~.)+geom_histogram(bins=40)
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
}

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
text(perc/l, newdata$X1+0.5, paste(newdata$team, newdata$oppTeam, sep=" - "), col="red", cex=0.5)


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

print(data.frame(newdata%>%dplyr::select(team, oppTeam, X1), train=perc/l, test=testperc/ltest, X2_new[,selected_features]))



################################################################################################################################




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





























