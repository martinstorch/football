setwd("~/LearningR/Bundesliga/Analysis")

seasons<-c("0405","0506","0607", "0708","0809","0910","1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718")

library(dplyr)

fetch_data<-function(season){
  url <- paste0("http://www.football-data.co.uk/mmz4281/", season, "/D1.csv")
  inputFile <- paste0("BL",season,".csv")
  download.file(url, inputFile, method = "libcurl")
  data<-read.csv(inputFile)
  data[is.na(data<-data)]<-0
  data$season<-season
  results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'Date', 'HS', 'AS', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR' )]
  results$season<-as.factor(results$season)
  with(results[!is.na(results$FTR),],{
    print(cbind ( table(FTR , season), table(FTR , season)/length(FTR)*100))
    print(table(FTHG , FTAG))
    print(cbind(meanHG = mean(FTHG), varHG = var(FTHG), meanAG = mean(FTAG),varAG = var(FTAG)))
    print(cor(FTHG, FTAG))
  })
  results$spieltag <- floor((9:(nrow(results)+8))/9)
  results$round <- ((results$spieltag-1) %% 34) +1
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
alldata$SeasonTeam <- paste(alldata$season, floor(alldata$round/5))
FTRs <- alldata %>% group_by(SeasonTeam) %>% dplyr::select(FTR, SeasonTeam) %>% table()

FTRs <- alldata %>% group_by(season) %>% dplyr::select(FTR, season) %>% table()

relFTRs <- t(t(FTRs) / rowSums(t(FTRs)))
colSums(relFTRs)
plot(t(relFTRs))

plot((relFTRs[1,]), type="l", ylim = c(0, 0.5), col="red", xaxt = "n")
lines((relFTRs[2,]), type="l", ylim = c(0, 0.5), col="green")
lines((relFTRs[3,]), type="l", ylim = c(0, 0.5), col="blue")
axis(1, at=seq_along(seasons), labels=seasons)

df <- as.data.frame(relFTRs)
plot(df %>% dplyr::filter(FTR=="A") %>% select(Freq), type="l", ylim = c(0, 0.5), col="red")

correlations <- sapply(unique(alldata$SeasonTeam), function(s) alldata%>%filter(SeasonTeam==s)%>%dplyr::select(FTHG, FTAG)%>%cor)[2,]

correlations <- sapply(seasons, function(s) alldata%>%filter(season==s)%>%dplyr::select(FTHG, FTAG)%>%cor)[2,]
draws_vs_corr = as.data.frame(cbind(cor=correlations, draws=relFTRs[2,]))
plot(draws_vs_corr)
abline(lm(draws~cor, data = draws_vs_corr))
cor(draws_vs_corr)



alldata$round3 <- floor(alldata$round/5)
alldata <- alldata %>% mutate(is11 = ifelse(FTHG==1 & FTAG==1, 1, 0))

draws_per_round <- (alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, FTR) %>% table())[,,2] 
draws_per_round <- (alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, FTR) %>% table())[14:14,,2] 
draws_per_round <- (alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, FTR) %>% table())[8:13,,2] 
draws_per_round <- (alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, FTR) %>% table())[1:2,,2] 
plot(Freq~round3, data=as.data.frame(draws_per_round))
lines(colMeans(draws_per_round), type="l", col="red")
points((alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, FTR) %>% table())[14,,2], col="green", pch="x")

plot(t(draws_per_round))
hist(draws_per_round)


is11 <- (alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, is11) %>% table())[1:14,,2] 
draws_per_round <- (alldata %>% group_by(season, round3) %>% dplyr::select(season, round3, FTR) %>% table())[,,2] 
is11 <- 2*draws_per_round+4*is11
is11 <- is11/45

hist(is11)
plot(Freq~round3, data=as.data.frame(is11))
lines(colMeans(is11), type="l", col="red")
points(is11[14,], col="green", pch="x")


dowdraw <- alldata %>% group_by(dayofweek) %>% dplyr::select(FTR, dayofweek) %>%table()
plot(t(dowdraw))
dowdraw <- apply(dowdraw, 1, function(x) x / colSums(dowdraw))
plot(dowdraw)
boxplot(Freq~dayofweek, data=as.data.frame(dowdraw))

boxplot(as.data.frame(t(dowdraw)))

dowdraw <- alldata  %>% group_by(FTR, season, round, dayofweek) %>% dplyr::select(FTR, season, round, dayofweek) %>% count()
boxplot(n~dayofweek+FTR, data=as.data.frame(dowdraw))
print.data.frame(dowdraw %>% group_by(dayofweek, FTR, n) %>% dplyr::summarize(n()))

print.data.frame(
  alldata  %>% group_by(FTR, season, dayofweek) %>% summarize(results=n())  %>% group_by(season, dayofweek) %>% summarize(n()) 
)

