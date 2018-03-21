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

teamresults <- 
  data.frame(team=results$HomeTeam, oppTeam=results$AwayTeam, 
             where="Home", round=results$round, season=results$season, 
             dow=results$dayofweek, gameindex=results$gameindex,
             cnt_own_Goals=results$FTHG, 
             cnt_opp_Goals=results$FTAG, 
             cnt_own_G1H=results$HTHG, 
             cnt_opp_G1H=results$HTAG, 
             cnt_own_G2H=results$FTHG-results$HTHG, 
             cnt_opp_G2H=results$FTAG-results$HTAG, 
             cnt_own_TargetShots=results$HST, 
             cnt_opp_TargetShots=results$AST,
             cnt_own_Shots=results$HS, 
             cnt_opp_Shots=results$AS,
             cnt_own_Corners=results$HC, 
             cnt_opp_Corners=results$AC,
             cnt_own_Fouls=results$HF, 
             cnt_opp_Fouls=results$AF,
             cnt_own_Yellow=results$HY, 
             cnt_opp_Yellow=results$AY,
             cnt_own_Red=results$HR, 
             cnt_opp_Red=results$AR)

teamresults <- rbind(
  data.frame(team=results$AwayTeam, oppTeam=results$HomeTeam, 
             where="Away", round=results$round, season=results$season,
             dow=results$dayofweek, gameindex=results$gameindex,
             cnt_own_Goals=results$FTAG, 
             cnt_opp_Goals=results$FTHG, 
             cnt_own_G1H=results$HTAG, 
             cnt_opp_G1H=results$HTHG, 
             cnt_own_G2H=results$FTAG-results$HTAG, 
             cnt_opp_G2H=results$FTHG-results$HTHG, 
             cnt_own_TargetShots=results$AST, 
             cnt_opp_TargetShots=results$HST,
             cnt_own_Shots=results$AS, 
             cnt_opp_Shots=results$HS,
             cnt_own_Corners=results$AC, 
             cnt_opp_Corners=results$HC,
             cnt_own_Fouls=results$AF, 
             cnt_opp_Fouls=results$HF,
             cnt_own_Yellow=results$AY, 
             cnt_opp_Yellow=results$HY,
             cnt_own_Red=results$AR, 
             cnt_opp_Red=results$HR),
  teamresults)

teamresults<-teamresults %>% mutate(
  team_promoted = as.integer(
    season=="2014_15" & (team=="FC Koln" | team=="Paderborn") |
      season=="2015_16" & (team=="Ingolstadt" | team=="Darmstadt") |
      season=="2016_17" & (team=="Freiburg" | team=="RB Leipzig") |
      season=="2017_18" & (team=="Stuttgart" | team=="Hannover") 
  ),
  oppTeam_promoted = as.integer(
    season=="2014_15" & (oppTeam=="FC Koln" | oppTeam=="Paderborn") |
      season=="2015_16" & (oppTeam=="Ingolstadt" | oppTeam=="Darmstadt") |
      season=="2016_17" & (oppTeam=="Freiburg" | oppTeam=="RB Leipzig") |
      season=="2017_18" & (oppTeam=="Stuttgart" | oppTeam=="Hannover") 
  )
)
##############################################################################################################
# Points

teamresults<-teamresults %>% mutate(
  cnt_own_Points=ifelse(  sign(cnt_own_Goals - cnt_opp_Goals)==1, 3, ifelse(cnt_own_Goals == cnt_opp_Goals, 1, 0)),
  cnt_opp_Points=ifelse(  sign(cnt_opp_Goals - cnt_own_Goals)==1, 3, ifelse(cnt_opp_Goals == cnt_own_Goals, 1, 0)),
  cnt_own_P1H=ifelse(  sign(cnt_own_G1H - cnt_opp_G1H)==1, 3, ifelse(cnt_own_G1H == cnt_opp_G1H, 1, 0)),
  cnt_opp_P1H=ifelse(  sign(cnt_opp_G1H - cnt_own_G1H)==1, 3, ifelse(cnt_opp_G1H == cnt_own_G1H, 1, 0)),
  cnt_own_P2H_1=as.integer(cnt_own_Points-cnt_own_P1H==1),
  cnt_opp_P2H_1=as.integer(cnt_opp_Points-cnt_opp_P1H==1),
  cnt_own_P2H_2=as.integer(cnt_own_Points-cnt_own_P1H==2),
  cnt_opp_P2H_2=as.integer(cnt_opp_Points-cnt_opp_P1H==2),
  cnt_own_P2H_3=as.integer(cnt_own_Points-cnt_own_P1H==3),
  cnt_opp_P2H_3=as.integer(cnt_opp_Points-cnt_opp_P1H==3)
)

##############################################################################################################
# Table summaries

features<-c("Goals", "G1H", "G2H", "TargetShots", "Shots", "Corners", "Fouls", "Yellow", "Red", "Points", "P1H", "P2H_1", "P2H_2", "P2H_3")

teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) %>%
  mutate(t_Matches_total = ifelse(round>1, round-1, 1))

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_own_",f)))), paste0("t_t1_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_opp_",f)))), paste0("t_t1_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("t_t1_own_",f)), opp=as.name(paste0("t_t1_opp_",f)))), paste0("t_t1_diff_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_Matches_total, total_cnt=as.name(paste0("t_t1_own_",f)))), paste0("mt_t1_own_",f))) %>% 
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_Matches_total, total_cnt=as.name(paste0("t_t1_opp_",f)))), paste0("mt_t1_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("mt_t1_own_",f)), opp=as.name(paste0("mt_t1_opp_",f)))), paste0("mt_t1_diff_",f)))
}

teamresults<-
  teamresults %>%
  group_by(season, where, team) %>%
  arrange(round) %>%
  mutate(t_MatchesT1_where = ifelse(row_number()>1, row_number()-1, 1))
  
for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_own_",f)))), paste0("t_t1_own_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_opp_",f)))), paste0("t_t1_opp_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("t_t1_own_",f,"_where")), opp=as.name(paste0("t_t1_opp_",f,"_where")))), paste0("t_t1_diff_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_MatchesT1_where, total_cnt=as.name(paste0("t_t1_own_",f,"_where")))), paste0("mt_t1_own_",f,"_where"))) %>% 
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_MatchesT1_where, total_cnt=as.name(paste0("t_t1_opp_",f,"_where")))), paste0("mt_t1_opp_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("mt_t1_own_",f,"_where")), opp=as.name(paste0("mt_t1_opp_",f,"_where")))), paste0("mt_t1_diff_",f,"_where")))
}

teamresults<-
  teamresults %>%
  group_by(season, oppTeam) %>%
  arrange(round)

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_own_",f)))), paste0("t_t2_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_opp_",f)))), paste0("t_t2_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("t_t2_own_",f)), opp=as.name(paste0("t_t2_opp_",f)))), paste0("t_t2_diff_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_Matches_total, total_cnt=as.name(paste0("t_t2_own_",f)))), paste0("mt_t2_own_",f))) %>% 
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_Matches_total, total_cnt=as.name(paste0("t_t2_opp_",f)))), paste0("mt_t2_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("mt_t2_own_",f)), opp=as.name(paste0("mt_t2_opp_",f)))), paste0("mt_t2_diff_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("t_t1_own_",f)), t2=as.name(paste0("t_t2_own_",f)))), paste0("t_t12_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("mt_t1_own_",f)), t2=as.name(paste0("mt_t2_own_",f)))), paste0("mt_t12_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("t_t1_opp_",f)), t2=as.name(paste0("t_t2_opp_",f)))), paste0("t_t12_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("mt_t1_opp_",f)), t2=as.name(paste0("mt_t2_opp_",f)))), paste0("mt_t12_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("t_t1_diff_",f)), t2=as.name(paste0("t_t2_diff_",f)))), paste0("t_t12_diff_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("mt_t1_diff_",f)), t2=as.name(paste0("mt_t2_diff_",f)))), paste0("mt_t12_diff_",f)))
}

teamresults<-
  teamresults %>%
  group_by(season, where, oppTeam) %>%
  arrange(round) %>%
  mutate(t_MatchesT2_where = ifelse(row_number()>1, row_number()-1, 1))

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_own_",f)))), paste0("t_t2_own_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ cumsum(cnt)-cnt, cnt=as.name(paste0("cnt_opp_",f)))), paste0("t_t2_opp_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("t_t2_own_",f,"_where")), opp=as.name(paste0("t_t2_opp_",f,"_where")))), paste0("t_t2_diff_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_MatchesT2_where, total_cnt=as.name(paste0("t_t2_own_",f,"_where")))), paste0("mt_t2_own_",f,"_where"))) %>% 
    mutate_(.dots= setNames(list(interp(~ total_cnt/t_MatchesT2_where, total_cnt=as.name(paste0("t_t2_opp_",f,"_where")))), paste0("mt_t2_opp_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("mt_t2_own_",f,"_where")), opp=as.name(paste0("mt_t2_opp_",f,"_where")))), paste0("mt_t2_diff_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("t_t1_own_",f,"_where")), t2=as.name(paste0("t_t2_own_",f,"_where")))), paste0("t_t12_own_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("mt_t1_own_",f,"_where")), t2=as.name(paste0("mt_t2_own_",f,"_where")))), paste0("mt_t12_own_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("t_t1_opp_",f,"_where")), t2=as.name(paste0("t_t2_opp_",f,"_where")))), paste0("t_t12_opp_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("mt_t1_opp_",f,"_where")), t2=as.name(paste0("mt_t2_opp_",f,"_where")))), paste0("mt_t12_opp_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("t_t1_diff_",f,"_where")), t2=as.name(paste0("t_t2_diff_",f,"_where")))), paste0("t_t12_diff_",f,"_where"))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("mt_t1_diff_",f,"_where")), t2=as.name(paste0("mt_t2_diff_",f,"_where")))), paste0("mt_t12_diff_",f,"_where"))) 
}

teamresults<-
  teamresults %>% ungroup() %>%
  mutate(mt_t1_Goal_efficiency = t_t1_own_Goals/t_t1_own_TargetShots) %>%
  mutate(mt_t1_Shot_efficiency = t_t1_own_TargetShots/t_t1_own_Shots) %>%
  mutate(mt_t2_Goal_efficiency = t_t2_own_Goals/t_t2_own_TargetShots) %>%
  mutate(mt_t2_Shot_efficiency = t_t2_own_TargetShots/t_t2_own_Shots) %>%
  mutate(mt_t1_Goal_efficiency_where = t_t1_own_Goals_where/t_t1_own_TargetShots_where) %>%
  mutate(mt_t1_Shot_efficiency_where = t_t1_own_TargetShots_where/t_t1_own_Shots_where) %>%
  mutate(mt_t2_Goal_efficiency_where = t_t2_own_Goals_where/t_t2_own_TargetShots_where) %>%
  mutate(mt_t2_Shot_efficiency_where = t_t2_own_TargetShots_where/t_t2_own_Shots_where)
  
#######################################################################################################
## Rank calculation

teamresults<-
  teamresults %>%
  group_by(season, round) %>%
  mutate(t_t1_Rank = rank(-t_t1_own_Points, ties.method="min"),
         t_t2_Rank = rank(-t_t2_own_Points, ties.method="min")) %>% 
  group_by(season, round, where) %>%
  mutate(t_t1_Rank_where = rank(-t_t1_own_Points, ties.method="min"),
         t_t2_Rank_where = rank(-t_t2_own_Points, ties.method="min")) %>% 
  arrange(season, round, team) %>%
  ungroup() %>%  
  mutate(t_t12_diff_Rank = t_t2_Rank-t_t1_Rank,
         t_t12_diff_Rank_where = t_t2_Rank_where-t_t1_Rank_where)

#######################################################################################################
## Last five games summary


lag5<-function(x, round) {
  return(ifelse(round>5, x-lag(x, 5), x))
}

teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) 

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ lag5(cnt, round), cnt=as.name(paste0("t_t1_own_",f)))), paste0("l5_t1_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ lag5(cnt, round), cnt=as.name(paste0("t_t1_opp_",f)))), paste0("l5_t1_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("l5_t1_own_",f)), opp=as.name(paste0("l5_t1_opp_",f)))), paste0("l5_t1_diff_",f)))
}

teamresults<-
  teamresults %>%
  group_by(season, oppTeam) %>%
  arrange(round) 

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ lag5(cnt, round), cnt=as.name(paste0("t_t2_own_",f)))), paste0("l5_t2_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ lag5(cnt, round), cnt=as.name(paste0("t_t2_opp_",f)))), paste0("l5_t2_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("l5_t2_own_",f)), opp=as.name(paste0("l5_t2_opp_",f)))), paste0("l5_t2_diff_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("l5_t1_own_",f)), t2=as.name(paste0("l5_t2_own_",f)))), paste0("l5_t12_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("l5_t1_opp_",f)), t2=as.name(paste0("l5_t2_opp_",f)))), paste0("l5_t12_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("l5_t1_diff_",f)), t2=as.name(paste0("l5_t2_diff_",f)))), paste0("l5_t12_diff_",f)))
}

teamresults<-
  teamresults %>%
  ungroup()

#######################################################################################################
## Last game summary

teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) 

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ lag(cnt), cnt=as.name(paste0("cnt_own_",f)))), paste0("l1_t1_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ lag(cnt), cnt=as.name(paste0("cnt_opp_",f)))), paste0("l1_t1_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("l1_t1_own_",f)), opp=as.name(paste0("l1_t1_opp_",f)))), paste0("l1_t1_diff_",f)))
}

teamresults<-
  teamresults %>%
  group_by(season, oppTeam) %>%
  arrange(round) 

for(f in features){
  teamresults <- teamresults %>%
    mutate_(.dots= setNames(list(interp(~ lag(cnt), cnt=as.name(paste0("cnt_opp_",f)))), paste0("l1_t2_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ lag(cnt), cnt=as.name(paste0("cnt_own_",f)))), paste0("l1_t2_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("l1_t2_own_",f)), opp=as.name(paste0("l1_t2_opp_",f)))), paste0("l1_t2_diff_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("l1_t1_own_",f)), t2=as.name(paste0("l1_t2_own_",f)))), paste0("l1_t12_own_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("l1_t1_opp_",f)), t2=as.name(paste0("l1_t2_opp_",f)))), paste0("l1_t12_opp_",f))) %>%
    mutate_(.dots= setNames(list(interp(~ t1-t2, t1=as.name(paste0("l1_t1_diff_",f)), t2=as.name(paste0("l1_t2_diff_",f)))), paste0("l1_t12_diff_",f)))
}

teamresults<-
  teamresults %>%
  ungroup()


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

for(f in features){
  print (f)
  for (s in c("t1_own_", "t1_opp_", "t2_own_", "t2_opp_")) {
    print(s)
    teamresults <- teamresults %>%
      mutate_(.dots= setNames(list(interp(~ NA)), paste0("SVD1_",s,f))) %>%
      mutate_(.dots= setNames(list(interp(~ NA)), paste0("SVD2_",s,f)))
    
    for (iseason in levels(teamresults$season)){
      print(iseason)
      for (iwhere in levels(teamresults$where))  {
        idata<-teamresults %>% filter(season==iseason & where==iwhere)
        l<-nlevels(teamresults$team)
        m<-matrix(NA, ncol=l, nrow=l)
        for (iround in 2:(idata %>% summarise(maxround=max(round)))$maxround) {
          d<-idata %>% filter(round==iround-1) %>% 
            dplyr::select_(.dots=setNames(obj=list("team", "oppTeam", interp(~ cnt, cnt=as.name(paste0("cnt_",substr(s,4,8),f)))),
                                          nm =list("team", "oppTeam", "svdx_input")))
          m[cbind(as.integer(d$team), as.integer(d$oppTeam))]<-d$svdx_input
          
          m0<-mean(m, na.rm=TRUE)
          m.svd1<-funkSVD(m-m0, k = 1)
          predMatrix1<-predict(m.svd1, newdata=m)+m0
          m.svd2<-funkSVD(m-m0, k = 2)
          predMatrix2<-predict(m.svd2, newdata=m)+m0
          
          teamresults <- teamresults %>%
            mutate_(.dots= setNames(list(
              interp(~ ifelse(round==iround, predMatrix1[cbind(as.integer(team), as.integer(oppTeam))], same), 
                     same=as.name(paste0("SVD1_",s,f)))), paste0("SVD1_",s,f))) %>%
            mutate_(.dots= setNames(list(
              interp(~ ifelse(round==iround, predMatrix2[cbind(as.integer(team), as.integer(oppTeam))], same), 
                     same=as.name(paste0("SVD2_",s,f)))), paste0("SVD2_",s,f)))
        }
      }
      data <- teamresults %>%
        dplyr::select_(.dots=setNames(obj=list(interp(~ svdx_input, svdx_input=as.name(paste0("cnt_",substr(s,4,8),f))),
                                               interp(~ pred1, pred1=as.name(paste0("SVD1_",s,f))),
                                               interp(~ pred2, pred2=as.name(paste0("SVD2_",s,f)))                         
                                               ),
                                      nm = list("svdx_input", "svdx_output1", "svdx_output2")))
                                      
      qqplot(data$svdx_output1, data$svdx_input)
      data %>% dplyr::select(svdx_output1, svdx_input) %>% smoothScatter(main=paste(s,f,(data %>% dplyr::select(svdx_output1, svdx_input) %>% cor(use = "complete.obs"))[1,2]))
      abline(lm(svdx_input ~ svdx_output1, data=data), col="red")
      print(paste0(s,f))
      print(data %>% dplyr::select(svdx_output1, svdx_input) %>% cor(use = "complete.obs"))
      print(data %>% dplyr::select(svdx_output2, svdx_input) %>% cor(use = "complete.obs"))
    }
  }
}

for(f in features){
    teamresults <- teamresults %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD1_t1_own_",f)), opp=as.name(paste0("SVD1_t1_opp_",f)))), paste0("SVD1_t1_diff_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD2_t1_own_",f)), opp=as.name(paste0("SVD2_t1_opp_",f)))), paste0("SVD2_t1_diff_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD1_t2_own_",f)), opp=as.name(paste0("SVD1_t2_opp_",f)))), paste0("SVD1_t2_diff_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD2_t2_own_",f)), opp=as.name(paste0("SVD2_t2_opp_",f)))), paste0("SVD2_t2_diff_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ s1-s2, s1=as.name(paste0("SVD1_t1_own_",f)), s2=as.name(paste0("SVD2_t1_own_",f)))), paste0("SVD12_t1_own_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ s1-s2, s1=as.name(paste0("SVD1_t2_own_",f)), s2=as.name(paste0("SVD2_t2_own_",f)))), paste0("SVD12_t2_own_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ s1-s2, s1=as.name(paste0("SVD1_t1_opp_",f)), s2=as.name(paste0("SVD2_t1_opp_",f)))), paste0("SVD12_t1_opp_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ s1-s2, s1=as.name(paste0("SVD1_t2_opp_",f)), s2=as.name(paste0("SVD2_t2_opp_",f)))), paste0("SVD12_t2_opp_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD1_t1_own_",f)), opp=as.name(paste0("SVD1_t2_opp_",f)))), paste0("SVD1_t12_own_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD1_t1_own_",f)), opp=as.name(paste0("SVD2_t2_opp_",f)))), paste0("SVD2_t12_own_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD1_t1_opp_",f)), opp=as.name(paste0("SVD1_t2_own_",f)))), paste0("SVD1_t12_opp_",f))) %>%
      mutate_(.dots= setNames(list(interp(~ own-opp, own=as.name(paste0("SVD1_t1_opp_",f)), opp=as.name(paste0("SVD2_t2_own_",f)))), paste0("SVD2_t12_opp_",f)))
}      


enrichPoisson<-function(input_data) {
  poiss_data<-input_data %>% 
    dplyr::select(poisx_input, gameindex, season, where, round, team, oppTeam, starts_with("mt_")
                  # mt_Points, mt_GS, mt_GC, mt_GS1H, mt_GC1H, mt_GS2H, mt_GC2H, mt_Shots, mt_Shotstarget, 
                  # mt_Fouls, mt_Corners, mt_Yellow, 
                  # mt_oppPoints, mt_oppGS, mt_oppGC, mt_oppGS1H, mt_oppGC1H, mt_oppGS2H, mt_oppGC2H, mt_oppShots, mt_oppShotstarget, 
                  # mt_oppFouls, mt_oppCorners, mt_oppYellow,
                  # mt_Shots_where, mt_oppShots_where
                  # mt_goal_efficiency, mt_Shot_efficiency, mt_oppgoal_efficiency, mt_oppShot_efficiency, mt_Red,  mt_oppRed
                  ) %>% dplyr::select(-contains("_diff_"), -contains("P2H"), -contains("_where"), -contains("t12"), -contains("_opp_")) %>% 
    mutate(poisx_output=NA, simple_poisx_output=NA)
  print(colnames(poiss_data))
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

#teamresults<-teamresults %>% mutate(poisx_input = cnt_own_Goals) %>% enrichPoisson() %>% mutate(pois1_own_Goals=simple_poisx_output, pois2_own_Goals=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_own_Shots) %>% enrichPoisson() %>% mutate(pois1_own_Shots=simple_poisx_output, pois2_own_Shots=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_own_TargetShots) %>% enrichPoisson() %>% mutate(pois1_own_TargetShots=simple_poisx_output, pois2_own_TargetShots=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_own_Yellow) %>% enrichPoisson() %>% mutate(pois1_own_Yellow=simple_poisx_output, pois2_own_Yellow=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_own_Fouls) %>% enrichPoisson() %>% mutate(pois1_own_Fouls=simple_poisx_output, pois2_own_Fouls=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_own_Corners) %>% enrichPoisson() %>% mutate(pois1_own_Corners=simple_poisx_output, pois2_own_Corners=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)

teamresults<-teamresults %>% mutate(poisx_input = cnt_opp_Shots) %>% enrichPoisson() %>% mutate(pois1_opp_Shots=simple_poisx_output, pois2_opp_Shots=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_opp_TargetShots) %>% enrichPoisson() %>% mutate(pois1_opp_TargetShots=simple_poisx_output, pois2_opp_TargetShots=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
#teamresults<-teamresults %>% mutate(poisx_input = cnt_opp_Yellow) %>% enrichPoisson() %>% mutate(pois1_opp_Yellow=simple_poisx_output, pois2_opp_Yellow=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_opp_Fouls) %>% enrichPoisson() %>% mutate(pois1_opp_Fouls=simple_poisx_output, pois2_opp_Fouls=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
teamresults<-teamresults %>% mutate(poisx_input = cnt_opp_Corners) %>% enrichPoisson() %>% mutate(pois1_opp_Corners=simple_poisx_output, pois2_opp_Corners=poisx_output) %>% dplyr::select(-poisx_input, -simple_poisx_output, -poisx_output)
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
    mutate(score=factor(paste(GS, GC, sep = "."), 
                        levels=c("0.4", "1.4", "0.3", "2.4", "1.3", "0.2", "3.4", "2.3", "1.2", "0.1",
                                 "0.0", "1.1", "2.2", "3.3", "4.4", "1.0", "2.1", "3.2", "4.3", 
                                 "2.0", "3.1", "4.2", "3.0", "4.1", "4.0" ), ordered=TRUE)) %>%
    dplyr::select(-GS, -GC)
}

scorecountsTeam<-sparse.model.matrix(~.-1, data = teamresults %>% dplyr::select(GS=cnt_own_Goals, GC=cnt_opp_Goals) %>% limitMaxScore(3) %>% buildFactorScore(), drop.unused.levels = TRUE)
scorecountsOppTeam<-sparse.model.matrix(~.-1, data = teamresults %>% dplyr::select(GS=cnt_opp_Goals, GC=cnt_own_Goals) %>% limitMaxScore(3) %>% buildFactorScore(), drop.unused.levels = TRUE)

scorecounttable<-
  cbind(teamresults %>% dplyr::select(team, oppTeam, season, round, gameindex, where), 
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

teamresults<-inner_join(teamresults, scorecounttable %>% dplyr::select(-starts_with("sc_")), 
                        by=c("team", "oppTeam", "season", "round", "where", "gameindex"))


######################################################################################################################

teamresults<-
  teamresults %>%
  mutate(mt_teamWin = (t_sc_team_score1.0+t_sc_team_score2.0+t_sc_team_score3.0+t_sc_team_score2.1+t_sc_team_score3.1+t_sc_team_score3.2)/t_Matches_total) %>%
  mutate(mt_teamLoss = (t_sc_team_score0.1+t_sc_team_score0.2+t_sc_team_score0.3+t_sc_team_score1.2+t_sc_team_score1.3+t_sc_team_score2.3)/t_Matches_total) %>%
  mutate(mt_teamDraw = (t_sc_team_score0.0+t_sc_team_score1.1+t_sc_team_score2.2+t_sc_team_score3.3)/t_Matches_total) %>%
  mutate(mt_oppTeamWin = (t_sc_oppTeam_score1.0+t_sc_oppTeam_score2.0+t_sc_oppTeam_score3.0+t_sc_oppTeam_score2.1+t_sc_oppTeam_score3.1+t_sc_oppTeam_score3.2)/t_Matches_total) %>%
  mutate(mt_oppTeamLoss = (t_sc_oppTeam_score0.1+t_sc_oppTeam_score0.2+t_sc_oppTeam_score0.3+t_sc_oppTeam_score1.2+t_sc_oppTeam_score1.3+t_sc_oppTeam_score2.3)/t_Matches_total) %>%
  mutate(mt_oppTeamDraw = (t_sc_oppTeam_score0.0+t_sc_oppTeam_score1.1+t_sc_oppTeam_score2.2+t_sc_oppTeam_score3.3)/t_Matches_total) %>%
  mutate(mt_teamWin_where = (t_sc_team_score1.0_where+t_sc_team_score2.0_where+t_sc_team_score3.0_where+t_sc_team_score2.1_where+t_sc_team_score3.1_where+t_sc_team_score3.2_where)/t_MatchesT1_where) %>%
  mutate(mt_teamLoss_where = (t_sc_team_score0.1_where+t_sc_team_score0.2_where+t_sc_team_score0.3_where+t_sc_team_score1.2_where+t_sc_team_score1.3_where+t_sc_team_score2.3_where)/t_MatchesT1_where) %>%
  mutate(mt_teamDraw_where = (t_sc_team_score0.0_where+t_sc_team_score1.1_where+t_sc_team_score2.2_where+t_sc_team_score3.3_where)/t_MatchesT1_where) %>%
  mutate(mt_oppTeamWin_where = (t_sc_oppTeam_score1.0_where+t_sc_oppTeam_score2.0_where+t_sc_oppTeam_score3.0_where+t_sc_oppTeam_score2.1_where+t_sc_oppTeam_score3.1_where+t_sc_oppTeam_score3.2_where)/t_MatchesT2_where) %>%
  mutate(mt_oppTeamLoss_where = (t_sc_oppTeam_score0.1_where+t_sc_oppTeam_score0.2_where+t_sc_oppTeam_score0.3_where+t_sc_oppTeam_score1.2_where+t_sc_oppTeam_score1.3_where+t_sc_oppTeam_score2.3_where)/t_MatchesT2_where) %>%
  mutate(mt_oppTeamDraw_where = (t_sc_oppTeam_score0.0_where+t_sc_oppTeam_score1.1_where+t_sc_oppTeam_score2.2_where+t_sc_oppTeam_score3.3_where)/t_MatchesT2_where) 

write.csv(x = teamresults, file="teamresults_2.csv", quote = TRUE, row.names = FALSE)

#teamresults<-read.csv(file="teamresults_2.csv")

x<-teamresults %>% dplyr::select(-starts_with("cnt_"))
y<-teamresults %>% dplyr::select(GS=cnt_own_Goals,GC=cnt_opp_Goals)
xy<-data.frame(y, x)

write.csv(x = xy, file="BLfeatures_2.csv", quote = TRUE, row.names = FALSE)


