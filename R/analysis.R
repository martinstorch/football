setwd("~/LearningR/Bundesliga/Analysis")
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

fetch_data<-function(season){
  url <- paste0("http://www.football-data.co.uk/mmz4281/", season, "/D1.csv")
  inputFile <- paste0("BL",season,".csv")
  #download.file(url, inputFile, method = "libcurl")
  data<-read.csv(inputFile)
  data[is.na(data<-data)]<-0
  data$season<-season
  #print(str(data))
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
  results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'Date', 'HS', 'AS', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR' , 'BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')]
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
alldata$SeasonTeam <- paste(alldata$season, floor(alldata$round/5))

draws_vs_corr<-alldata%>%group_by(round)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG))
#draws_vs_corr<-alldata%>%group_by(season)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())
#draws_vs_corr<-alldata%>%group_by(HomeTeam)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG))
#draws_vs_corr<-alldata%>%group_by(AwayTeam)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG))
#draws_vs_corr<-alldata%>%group_by(r = sample(seq(1,nrow(alldata))%%34))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())
#draws_vs_corr<-alldata%>%group_by(r = sample(15, size = nrow(alldata), replace = T))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())
table(draws_vs_corr$n)

sample(seq(1,nrow(alldata))%%34)

drawmodel<-lm(pct_draw~poly(round, 2), data=draws_vs_corr)
summary(drawmodel)

plot(pct_draw~round, data=draws_vs_corr)
points(pdraws ~ round, data = draws_vs_corr%>%mutate(pdraws=predict(drawmodel, as.data.frame(round)))%>%arrange(round), type="l")

alldata$draw_prior <- predict(drawmodel, newdata=alldata)

quote_names<-c('BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')
normalized_quote_names<-paste0("p", quote_names)

alldata[,normalized_quote_names]<-1/alldata[,quote_names]
alldata[,normalized_quote_names[1:3]]<-alldata[,normalized_quote_names[1:3]]/rowSums(alldata[,normalized_quote_names[1:3]])
alldata[,normalized_quote_names[4:6]]<-alldata[,normalized_quote_names[4:6]]/rowSums(alldata[,normalized_quote_names[4:6]])

w<-4
offset<-0
draws_vs_corr<-alldata%>%group_by(season, floor((round+offset)/w))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())
rsq <- summary(lm(pct_draw ~ corr, data=draws_vs_corr))$r.square
print(rsq)
rsqrand<-function(x){
  draws_vs_corr<-alldata%>%group_by(r = sample(paste(season, floor((round+offset)/w))))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())
  s<-summary(lm(pct_draw ~ corr, data=draws_vs_corr))
  rsq<-s$r.squared
  return(rsq)
}
x <- sapply(1:300, rsqrand)
hist(x)
abline(v=rsq, col="red")
print(mean(x>rsq))

print.data.frame(draws_vs_corr)

w<-1
offset<-0
alldata0 <- alldata %>% filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")
ts<-alldata0%>%group_by(r=paste(season, format(floor((round+offset)/w), digits=2)))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())%>%
  arrange(r) # %>%dplyr::select(r, pct_draw, corr)
acf(ts[,2:3], lag.max = 34, plot = T)

library(astrochron)
ts$r<-1:nrow(ts)
mwCor(ts, win = 4, cols = 2:3)
mwCor(ts[sample(nrow(ts)),], win = 4, cols = 2:3)

library('forecast')
library('tseries')

draw_ts <- ts(ts$pct_draw, frequency = 34, start = 2004)
#draw_ts <- ts(ts[,c("pct_draw", "corr")])
#draw_ts <- ts(ts$corr)

plot(draw_ts)
plot(ma(draw_ts, order=34))

decomp <- stl(ts(na.omit(ma(draw_ts, order=1)), frequency=34), s.window="periodic")
decomp <- stl(draw_ts, s.window="periodic")
deseasonal_cnt <- seasadj(decomp)
#deseasonal_cnt == decomp$time.series[,2]+decomp$time.series[,3]
plot(decomp)
summary(decomp)
ggplot(decomp$time.series, aes(x=seasonal))+geom_histogram(bins=50)

adf.test(draw_ts, alternative = "stationary", k=34)
adf.test(deseasonal_cnt, alternative = "stationary", k=34)

adf.test(ts(na.omit(ts$pct_draw - lag(ts$pct_draw, 3*34))), alternative = "stationary", k=34)

acf(draw_ts)
pacf(draw_ts)

count_d1 = diff(deseasonal_cnt, lag=1, differences = 1)
plot(count_d1)
adf.test(count_d1, alternative = "stationary")

Acf(count_d1, main='ACF for Differenced Series')
Pacf(count_d1, main='PACF for Differenced Series')

auto.arima(deseasonal_cnt, seasonal=FALSE)
auto.arima(count_d1, seasonal=FALSE)
auto.arima(draw_ts, seasonal=FALSE)

fit<-auto.arima(draw_ts, seasonal=FALSE)
fit<-arima(draw_ts, order=c(1,1,1))
str(predict(fit))
tsdisplay(residuals(fit), lag.max=45, main='(1,1,1) Model Residuals')

fitted<-draw_ts-fit$residuals
plot(x=(draw_ts), y=fitted)
cor(draw_ts , fitted)
summary(lm(fitted ~ draw_ts))


fit2 = auto.arima(count_d1, seasonal=F) #, order=c(1,1,7))
library(DescTools)

alldata%>%group_by(r=format(AddMonths(Date, 6),"%m"), season)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())%>%arrange(-pct_draw)%>%
  group_by(season)%>%mutate(rk=rank(-pct_draw, ties.method = "rand"))%>%arrange(r)%>% # print.data.frame()
  ggplot(aes(x=rk, fill=r))+ scale_fill_brewer(palette = "Spectral")+geom_density(alpha=0.4)+geom_histogram(bins=10)

  print.data.frame()

alldata0 <- alldata %>% filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")
ts<-alldata%>%group_by(r=paste(season, format(round, digits=2)))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())%>%
  arrange(r) # %>%dplyr::select(r, pct_draw, corr)
draw_ts <- ts(ts$pct_draw, frequency = 34, start = 2004)
draw_ts <- ts(ts$corr, frequency = 34, start = 2004)
Acf(count_d1, main='ACF for Differenced Series')
Pacf(count_d1, main='PACF for Differenced Series')
table(ts$n)
w<-window(draw_ts, c(2004, 1), c(2012, 27))
wtest<-window(draw_ts, c(2012, 28))

decomp <- stl(w, s.window="periodic")
periodic<-decomp$time.series[1:34,1]
plot(periodic, type="l")
points(smooth(periodic), type="l", col="red")
#ggplot(data.frame(p=rep(periodic, rep=3), i=1:(3*34)), aes(x=i, y=p))+geom_line()+geom_smooth()+geom_vline(xintercept = c(35, 68), col="red")
#loess(p~i, data=data.frame(p=rep(periodic, rep=3), i=1:(3*34)))
ggplot(data.frame(p=rep(periodic, rep=1), i=1:(34)), aes(x=i, y=p))+geom_line()+geom_smooth()
periodic_smooth<-predict(loess(p~i, data=data.frame(p=rep(periodic, rep=1), i=1:(1*34))))*0

#deseasonal_cnt <- seasadj(decomp)
deseasonal_cnt <- w-periodic_smooth
deseasonal_test <- wtest-periodic_smooth

count_d1 = diff(deseasonal_cnt, lag=2, differences = 2)
count_d1_test = diff(deseasonal_test, lag=2, differences = 2)
count_d2 = diff(deseasonal_cnt, lag=1, differences = 4)
count_d2_test = diff(deseasonal_test, lag=1, differences = 4)
var(count_d1)
var(count_d2)
#wtest[1:34]
#count_d1_test[1:34]
#count_d2_test[1:34]

dm<-mean(deseasonal_cnt)
fit0 = arima(deseasonal_cnt-dm, order=c(2,0,1), xreg = deseasonal_cnt-lag(as.vector(deseasonal_cnt[]),2))
fit1 = arima(count_d1, order=c(4,1,1))
fit2 = arima(count_d2, order=c(4,2,1))
summary(fit0)
var(deseasonal_cnt)
sd(deseasonal_cnt)
# fit1
# fit2
# fit2$aic
mean(draw_ts)
fc<-forecast(fit0, h = 20, xreg = deseasonal_cnt-lag(as.vector(deseasonal_cnt[]),2))
plot(c(deseasonal_cnt, deseasonal_test), xlim=length(deseasonal_cnt)+c(-10, 20), type="o")
abline(v=length(deseasonal_cnt))
points(length(deseasonal_cnt)+(1:20), fc$mean+dm, col="red", type="o")

fc<-forecast(fit1, h = 20)
plot(c(count_d1, count_d1_test), xlim=length(count_d1)+c(-10, 20), type="o")
abline(v=length(count_d1))
points(length(count_d1)+(1:20), fc$mean, col="red", type="o")

fc<-forecast(fit2, h = 20)
plot(c(count_d2, count_d2_test), xlim=length(count_d2)+c(-10, 20), type="o")
abline(v=length(count_d2))
points(length(count_d2)+(1:20), fc$mean, col="red", type="o")

length(count_d2)
length(count_d2_test)

fitted1<-count_d1-fit1$residuals
plot(count_d1, fitted1)
abline(lm(fitted1 ~ count_d1))
cor(count_d1 , fitted1)
summary(lm(fitted1 ~ count_d1))$r.square

newfit1<-Arima(count_d1_test, model=fit1) 
newfit1$aic
points(count_d1_test, newfit1$fitted, col="red")
abline(lm(newfit1$fitted ~ count_d1_test), col="red")
cor(count_d1_test, newfit1$fitted)
summary(lm(newfit1$fitted ~ count_d1_test))$r.square

fitted2<-count_d2-fit2$residuals
plot(count_d2, fitted2)
abline(lm(fitted2 ~ count_d2))
cor(count_d2 , fitted2)
summary(lm(fitted2 ~ count_d2))$r.square

newfit2<-Arima(count_d2_test, model=fit2) 
newfit2$aic
points(count_d2_test, newfit2$fitted, col="red")
abline(lm(newfit2$fitted ~ count_d2_test), col="red")
cor(count_d2_test, newfit2$fitted)
summary(lm(newfit2$fitted ~ count_d2_test))$r.square

refitted_train1 <- deseasonal_cnt+fitted1 +periodic_smooth[c(2:34, 1)]
refitted_test1 <- deseasonal_test+newfit1$fitted +periodic_smooth[c(2:34, 1)]
refitted_train2 <- deseasonal_cnt+fitted2 +periodic_smooth[c(3:34, 1:2)]
refitted_test2 <- deseasonal_test+newfit2$fitted +periodic_smooth[c(3:34, 1:2)]

plot(w[-1:-1], refitted_train1)
abline(lm(refitted_train1 ~ w[-1:-1]))
cor(w[-1:-1] , refitted_train1)
summary(lm(refitted_train1 ~ w[-1:-1]))$r.square

points(wtest[-1:-1], refitted_test1, col="red")
abline(lm(refitted_test1 ~ wtest[-1:-1]), col="red")
cor(wtest[-1:-1] , refitted_test1)
summary(lm(refitted_test1 ~ wtest[-1:-1]))$r.square

points(w[-2:-1], refitted_train2, col="blue")
abline(lm(refitted_train2 ~ w[-2:-1]), col="blue")
cor(w[-2:-1] , refitted_train2)
summary(lm(refitted_train2 ~ w[-2:-1]))$r.square

points(wtest[-2:-1], refitted_test2, col="green")
abline(lm(refitted_test2 ~ wtest[-2:-1]), col="green")
cor(wtest[-2:-1] , refitted_test2)
summary(lm(refitted_test2 ~ wtest[-2:-1]))$r.square

summary(lm(w[-2:-1]~refitted_train1[-1:-1]+refitted_train2))$r.square
summary(lm(wtest[-2:-1]~refitted_test1[-1:-1]+refitted_test2))$r.square

boxplot(refitted_train1 ~ w[-1:-1])
boxplot(refitted_test1 ~ wtest[-1:-1])
boxplot(refitted_train2 ~ w[-2:-1])
boxplot(refitted_test2 ~ wtest[-2:-1])

#' Reverse pre-processing done by caret to a dataset
#'
#' @param preProc a preProcess object from the caret package
#' @param data a data.frame with the same elements as the preProcess object
#'
#' @return a data.frame
#' @details This only reverses scaling and centering done by preProcess currently. 
#' It cannot undo PCA transformations, it cannot undo imputation, exponential transformations, 
#' or any other method
#' @export
unPreProc <- function(preProc, data){
  stopifnot(class(preProc) == "preProcess")
  stopifnot(class(data) == "data.frame")
  for(i in names(preProc$mean)){
    tmp <- data[, i] * preProc$std[[i]] + preProc$mean[[i]]
    data[, i] <- tmp
  }
  return(data)  
}
invBoxCox <- function(x, lambda) 
  if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda) 

far2 <- function(x, h){forecast(Arima(x, order=c(5,1,5)), h=h)}
e <- tsCV(draw_ts-periodic_smooth, far2, h=1)#, window=7*34)
print(e)
hist(e)
summary(e[108:504])
sd(e[108:504], na.rm = T)
plot(draw_ts-periodic_smooth, draw_ts-periodic_smooth-e)
act<-draw_ts
pred<-draw_ts-e
act<-act[108:504]
pred<-pred[108:504]
cor(act, pred, use = "complete.obs")
summary(lm(act ~ pred, na.action = na.exclude))

#####################################################################################
alldata0 <- alldata %>% filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")
ts<-alldata%>%group_by(r=paste(season, format(round, digits=2)))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG), meangoals=mean(FTHG+FTAG), FTHG=mean(FTHG), FTAG=mean(FTAG), gdiff=mean(FTHG-FTAG), n=n())%>%
  arrange(r) # %>%dplyr::select(r, pct_draw, corr)
draw_ts <- ts(ts$pct_draw, frequency = 34, start = 2004)
draw_ts <- ts(ts$corr, frequency = 34, start = 2004)
Acf(draw_ts, main='ACF for Original Series')
Pacf(draw_ts, main='PACF for Original Series')
count_d1 = diff(draw_ts, lag=1, differences = 1)
Acf(count_d1, main='ACF for Differenced Series')
Pacf(count_d1, main='PACF for Differenced Series')
count_d2 = diff(draw_ts, lag=1, differences = 2)
Acf(count_d1, main='ACF for 2nd Differenced Series')
Pacf(count_d1, main='PACF for 2nd Differenced Series')
count_d4 = diff(draw_ts, lag=1, differences = 4)
Acf(count_d4, main='ACF for 4th Differenced Series')
Pacf(count_d4, main='PACF for 4th Differenced Series')

w<-window(draw_ts, c(2004, 1), c(2013, 34))
wtest<-window(draw_ts, 2014)
decomp <- stl(w, s.window="periodic")
periodic<-decomp$time.series[1:34,1]
ggplot(data.frame(p=rep(periodic, rep=1), i=1:(34)), aes(x=i, y=p))+geom_line()+geom_smooth()
ggplot(data.frame(p=stl(wtest, s.window="periodic")$time.series[1:34,1], i=1:(34)), aes(x=i, y=p))+geom_line()+geom_smooth()
periodic_smooth<-predict(loess(p~i, data=data.frame(p=rep(periodic, rep=1), i=1:(1*34))))
periodic_smooth<-periodic*0
mn <- mean(draw_ts)
m0<-Arima(w-periodic_smooth-mn, order=c(10,0,2))
m1<-Arima(w-periodic_smooth-mn, order=c(6,1,4))
m2<-Arima(w-periodic_smooth-mn, order=c(2,2,6))
# far0 <- function(x, h){forecast(Arima(x, order=c(10,0,0)), h=h)}
# far1 <- function(x, h){forecast(Arima(x, order=c(20,0,0)), h=h)}
# far2 <- function(x, h){forecast(Arima(x, order=c(30,0,0)), h=h)}
far0 <- function(x, h){forecast(Arima(x, model=m0), h=h)}
far1 <- function(x, h){forecast(Arima(x, model=m1), h=h)}
far2 <- function(x, h){forecast(Arima(x, model=m2), h=h)}
e0 <- tsCV(draw_ts-periodic_smooth-mn, far0, h=1)#, window=34)
e1 <- tsCV(draw_ts-periodic_smooth-mn, far1, h=1)#, window=34)
e2 <- tsCV(draw_ts-periodic_smooth-mn, far2, h=1)#, window=34)
plot((e0+e1+e2)/3)
summary((e0+e1+e2)/3)
sd((e0+e1+e2)/3, na.rm = T)
#pred0<-draw_ts*0.0+0.247+periodic_smooth
#pred1<-draw_ts*0.0+0.247+periodic
pred0<-draw_ts-lag(as.vector(e0[]))+periodic_smooth
pred1<-draw_ts-lag(as.vector(e1[]))+periodic_smooth
pred2<-draw_ts-lag(as.vector(e2[]))+periodic_smooth
act<-draw_ts
n<-length(act)
act<-act[-seq_along(w)]
pred0<-pred0[-seq_along(w)]
pred1<-pred1[-seq_along(w)]
pred2<-pred2[-seq_along(w)]
cor(act, pred0, use = "complete.obs", method = "pearson")
cor(act, pred1, use = "complete.obs")
cor(act, pred2, use = "complete.obs")
cor(act, pred0+pred1+pred2, use = "complete.obs")
summary(lm(act ~ pred0, na.action = na.exclude))
summary(lm(act ~ pred1, na.action = na.exclude))
summary(lm(act ~ pred2, na.action = na.exclude))
summary(lm(act ~ pred1+pred2, na.action = na.exclude))
summary(lm(act ~ pred0+pred1+pred2, na.action = na.exclude))
#plot(act, (pred0+pred1+pred2)/3)
plot(act+0*rnorm(n = length(act), sd=0.004), pred0, xlim=range(act)+c(-0.05, 0.05), ylim=range(act)*1.2)
points(act+0*rnorm(n = length(act), sd=0.004)+0*0.02, pred1, col="red")
points(act+0*rnorm(n = length(act), sd=0.004)+0*0.04, pred2, col="blue")
abline(lm(act ~ pred0, na.action = na.exclude))
abline(lm(act ~ pred1, na.action = na.exclude), col="red")
abline(lm(act ~ pred2, na.action = na.exclude), col="blue")

m<-lm(act ~ pred0+pred1+pred2, na.action = na.exclude)
plot(act, predict(m))
ppp<-predict(m)
plot((seq_along(ppp)-1)%%34, ppp)

plot(act[477:504], type="l", xlim=c(1, 34), lwd=2)
points(act[477:504], type="p")
points(ppp[477:504], type="l", col="red")
points(pred2[477:504], type="l", col="blue")
points(ppp[477:504], type="p", col="red")
points(pred2[477:504], type="p", col="blue")

# new predictions from current season
p0<-far0(window(draw_ts, 2018)-periodic_smooth[1:28]-0.247, h=6)
p1<-far1(window(draw_ts, 2018)-periodic_smooth[1:28], h=6)
p2<-far2(window(draw_ts, 2018)-periodic_smooth[1:28], h=6)
preddraw<-data.frame(
  pred0 = p0$mean+0.247+periodic_smooth[29:34],
  pred1 = p1$mean+periodic_smooth[29:34],
  pred2 = p2$mean+periodic_smooth[29:34]
)
preddraw$r<-predict(m, newdata=preddraw)
preddraw

# new predictions from all seasons
p0<-far0(window(draw_ts, 2004)-periodic_smooth[1:34]-0.247, h=6)
p1<-far1(window(draw_ts, 2004)-periodic_smooth[1:34], h=6)
p2<-far2(window(draw_ts, 2004)-periodic_smooth[1:34], h=6)
preddraw<-data.frame(
  pred0 = p0$mean+0.247+periodic_smooth[29:34],
  pred1 = p1$mean+periodic_smooth[29:34],
  pred2 = p2$mean+periodic_smooth[29:34]
)
preddraw$r<-predict(m, newdata=preddraw)
preddraw

points(29:34, preddraw$r, type="l", col="red")
points(29:34, preddraw$r, type="p", col="red")
points(29:34, preddraw$pred2, type="l", col="blue")
points(29:34, preddraw$pred2, type="p", col="blue")


summary(ppp)



draw_ts_trans

draw_ts_trans = preProcess(data.frame(draws=as.vector(draw_ts[])), c("center", "scale"))
predictorsTrans = data.frame(trans = predict(draw_ts_trans, data.frame(draws=as.vector(draw_ts[]))))
hist(predictorsTrans$draws, breaks = 50)
periodic_smoothTrans = data.frame(trans = predict(draw_ts_trans, data.frame(draws=periodic_smooth)))

newmodel<-Arima(predictorsTrans$draws-periodic_smooth, order=c(5,1,5))
#newmodel<-auto.arima(predictorsTrans$draws-periodic_smooth, seasonal = F)
newmodel
fc<-forecast(newmodel, h=6)
fc$mean+periodic_smooth[29:34]
unPreProc(draw_ts_trans , data.frame(draws=fc$mean+periodic_smooth[29:34]))

plot(abs(e))

tsdisplay(residuals(fit2), lag.max=45, main='Seasonal Model Residuals')


fcast <- forecast(fit2, h=30)
plot(fcast)

hold <- window(ts(deseasonal_cnt), start=380)

fit_no_holdout = arima(ts(deseasonal_cnt[1:400]), order=c(4,1,4))
fcast_no_holdout <- forecast(fit_no_holdout,h=100)
plot(fcast_no_holdout, main=" ")
lines(ts(deseasonal_cnt))

plot(deseasonal_cnt[401:425], fcast_no_holdout$mean[1:25])
cor(deseasonal_cnt[401:425], fcast_no_holdout$mean[1:25])

fit_w_seasonality = arima(deseasonal_cnt, seasonal = list(order = c(1L, 1L, 1L), period = 34), order = c(4L, 1L, 4L))
fit_w_seasonality
seas_fcast <- forecast(fit_w_seasonality, h=30)
plot(seas_fcast)




plot(pct_draw ~ corr, data=draws_vs_corr)
abline(lm(pct_draw ~ corr, data=draws_vs_corr))
summary(lm(pct_draw ~ corr, data=draws_vs_corr))$r.square

plot(pct_draw ~ meangoals, data=draws_vs_corr)
abline(lm(pct_draw ~ meangoals, data=draws_vs_corr))
summary(lm(pct_draw ~ meangoals, data=draws_vs_corr))

plot(pct_draw ~ I(corr-meangoals), data=draws_vs_corr)
abline(lm(pct_draw ~ I(corr-meangoals), data=draws_vs_corr))
summary(lm(pct_draw ~ I(corr-meangoals), data=draws_vs_corr))

plot(pct_draw ~ I(corr-gdiff), data=draws_vs_corr)
abline(lm(pct_draw ~ I(corr-gdiff), data=draws_vs_corr))
summary(lm(pct_draw ~ I(corr-gdiff), data=draws_vs_corr))

plot(pct_draw ~ FTHG, data=draws_vs_corr)
abline(lm(pct_draw ~ FTHG, data=draws_vs_corr))
summary(lm(pct_draw ~ FTHG, data=draws_vs_corr))

cor(draws_vs_corr[,-1])
#draws_vs_corr<-(draws_vs_corr%>%arrange(meangoals))[2:33,]

FTRs <- alldata %>% group_by(SeasonTeam) %>% dplyr::select(FTR, SeasonTeam) %>% table()

FTRs <- alldata %>% group_by(season) %>% dplyr::select(FTR, season) %>% table()

relFTRs <- t(t(FTRs) / rowSums(t(FTRs)))
colSums(relFTRs)
plot(t(relFTRs))

goals <- alldata %>% group_by(season) %>% dplyr::select(season, FTHG, FTAG) %>% summarise(FTHG=mean(FTHG), FTAG=mean(FTAG))
ggplot(goals, aes(x = FTHG, y=FTAG, col=season))+ scale_fill_brewer(palette = "Spectral")+geom_text(aes(label=season))

f <-alldata  %>% select(FTHG, FTAG) %>% filter(FTHG<=6, FTAG<=6) %>% table() / nrow(alldata)
f * 100
f2<-log(1/f)/100
f2[is.infinite(f2)]<-0.01
f2 <- f2 / mean(f2)
paste(as.vector(t(f2)), collapse=", ")

alldata %>% filter(season==1819) %>% select(FTR) %>% table()
alldata %>% filter(season==1819) %>% select(FTHG, FTAG) %>% table()
alldata %>% filter(season==1819) %>% group_by(FTHG) %>% count()
alldata %>% filter(season==1819) %>% group_by(FTAG) %>% count()
alldata %>% filter(season==1819) %>% select(FTHG, FTAG) %>% summary()


print(alldata %>% filter(season==1819) %>% group_by(FTR) %>% summarise(x=n()) %>% ungroup() %>% mutate(x=x/sum(x)*100))
s1<-alldata %>% filter(season==1819) %>% mutate(GD=FTHG-FTAG) %>% group_by(GD) %>% summarise(x=n()) %>% ungroup() %>% mutate(x=x/sum(x)*100) 
print(s1)
s1<-alldata %>% filter(season==1819) %>% group_by(FTHG, FTAG) %>% summarise(x=n()) %>% ungroup() %>% mutate(x=x/sum(x)*100) 
s2<-acast(s1, FTHG~FTAG, value.var="x")
s2[is.na(s2)]<-0
print(s2)


# Scatterplot
theme_set(theme_bw())  # pre-set the bw theme.
g <- ggplot(alldata %>% filter(season==1819)%>%select(FTHG,FTAG)%>%group_by(FTHG,FTAG)%>%mutate(n=n()), aes(FTHG, FTAG, size=n))
ggMarginal(g+geom_point(col="blue", show.legend=F), type = "histogram", fill="tomato3", colour = "tomato3")
#ggMarginal(g, type = "boxplot", fill="transparent")

plot((relFTRs[1,]), type="l", ylim = c(0, 0.5), col="red", xaxt = "n")
lines((relFTRs[2,]), type="l", ylim = c(0, 0.5), col="green")
lines((relFTRs[3,]), type="l", ylim = c(0, 0.5), col="blue")
axis(1, at=seq_along(seasons), labels=seasons)

draws_vs_corr<-alldata%>%group_by(season)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG))
draws_vs_corr%>%summary()
plot(pct_draw~corr, data=draws_vs_corr)
points(pct_draw~corr, data=draws_vs_corr%>%filter(season=="1819"), col="red")
abline(lm(pct_draw~corr, data = draws_vs_corr))
cor(draws_vs_corr[,2:3])

draws_vs_corr<-alldata%>%group_by(season, floor(round/6))%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG))
draws_vs_corr%>%summary()
plot(pct_draw~corr, data=draws_vs_corr)
points(pct_draw~corr, data=draws_vs_corr%>%filter(season=="1819"), col="red")
abline(lm(pct_draw~corr, data = draws_vs_corr))
cor(draws_vs_corr[,3:4])

draws_vs_corr<-alldata%>%group_by(season, round)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),                                                                                                 corr=cor(FTHG, FTAG))
draws_vs_corr%>%summary()
plot(pct_draw~corr, data=draws_vs_corr)
points(pct_draw~corr, data=draws_vs_corr%>%filter(season=="1819"), col="red")
abline(lm(pct_draw~corr, data = draws_vs_corr))
cor(draws_vs_corr[,3:4])

draws_vs_corr<-alldata%>%group_by(f=as.factor(floor(round/6)), season)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG))
draws_vs_corr<-alldata%>%group_by(f=subround, season)%>%mutate(isdraw = ifelse(FTR=="D", 1,0))%>%summarise(pct_draw=mean(isdraw),corr=cor(FTHG, FTAG))
draws_vs_corr%>%summary()
plot(pct_draw~corr, data=draws_vs_corr)
points(pct_draw~corr, data=draws_vs_corr%>%filter(season=="1819"), col="red")
abline(lm(pct_draw~corr, data = draws_vs_corr))
cor(draws_vs_corr[,3:4])


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



#########################################################################################################################

thedata <- alldata  [alldata$season=="1819",]
quotes <- thedata[,c('B365H', 'B365D', 'B365A')]
quotes <- thedata[,c('BWH', 'BWD', 'BWA')]
mixit<-function(x){
  drawpred<-(abs(quotes[,1]-quotes[,3]) < x)
  qpred <- 2-apply(quotes, 1, which.min)
  qpred[drawpred]<- 0
  gof<-cbind(qpred, sign(thedata$FTHG-thedata$FTAG))
  cm <- table(sign(thedata$FTHG-thedata$FTAG), qpred)
  hw<-thedata$FTHG>thedata$FTAG
  aw<-thedata$FTHG<thedata$FTAG
  draw<-thedata$FTHG==thedata$FTAG
  
  tendency<-qpred==sign(thedata$FTHG-thedata$FTAG)
  gdiff<-tendency & ((abs(thedata$FTHG-thedata$FTAG)==1) | draw)
  hit<-gdiff & ((thedata$FTHG+thedata$FTAG) %in% 2:3)
  table(tendency)
  table(gdiff)
  table(hit)
  pistor <- (sum(tendency)+sum(gdiff)+sum(hit))/length(qpred)
  sky <- (2*sum(tendency)+0*sum(gdiff)+3*sum(hit))/length(qpred)
  tcs <- (2*sum(tendency&hw)+1*sum(gdiff&hw)+1*sum(hit&hw)+4*sum(tendency&aw)+1*sum(gdiff&aw)+2*sum(hit&aw)+2*sum(tendency&draw)+4*sum(hit&draw))/length(qpred)
  #print(c(pistor, sky, tcs))
  tt<-table(qpred)/length(qpred)
  return (c(pistor, sky, tcs, sum(qpred==0)/length(qpred), sum(qpred==1)/length(qpred)))
}
mr <- t(sapply(seq(0,5,0.1), mixit))
plot(mr[,4], mr[,2])

mixit(0.2)

levels(alldata$season)

train_seasons<-c("0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718")
test_seasons<-c("1819")

train_seasons<-c("0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819")
test_seasons<-c("0405", "0506", "0607", "0708")

train_seasons<-c("0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314")
test_seasons<-c( "1415", "1516", "1617", "1718", "1819")

train_seasons<-sample(levels(alldata$season), 8)
test_seasons<-setdiff(levels(alldata$season), train_seasons)

qmix<-function(pwin, ploss, thedata, quotes){
  if (pwin+ploss>1.0)
    return (pwin, ploss, NA, NA, NA, NA, NA, NA, NA, NA)
  cutoffs <- quantile(quotes[,1]-quotes[,3], c(ploss, 1-pwin), names=F) # +0.1*quotes[,2]+0.01*quotes[,1]
  #print(c(pwin, ploss, cutoffs))
  qqpred<-cut(quotes[,1]-quotes[,3], c(-1e10, cutoffs+c(0.0, 0.00001), 1e10), labels=c("A", "D", "H"))
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
  return (c(pwin=pwin, ploss=ploss, q=cutoffs, pistor=pistor, sky=sky, tcs=tcs, hwin=sum(qpred==1)/length(qpred), draw=sum(qpred==0)/length(qpred), away=sum(qpred==-1)/length(qpred)))
}

prepare_plot_data<-function(thedata, step=0.01){
  #quotes <- 1/thedata[,c('B365H', 'B365D', 'B365A')]
  quotes <- 1/thedata[,c('BWH', 'BWD', 'BWA')]
  quotes <- quotes / rowSums(quotes)
  q<-c()  
  for (pwin in seq(0.4,0.9,step)) {
    for (ploss in seq(0.1-step, 1.00-pwin, step)) {
      q<-rbind(q, qmix(pwin, ploss, thedata, quotes))  
    }
  }
  q <- data.frame(q)
  q<-unique(q %>% select(-pwin, -ploss))
  return(q)
}

alldata %>% filter(season %in% train_seasons) %>% mutate(p=(1/BWH-1/BWA)/(1/BWH+1/BWA+1/BWD)) %>% ggplot()+geom_histogram(aes(p), binwidth=0.01)

gridscope<-levels(alldata$season)[-2]
seasonsq <- lapply(gridscope, function(x) cbind(season=x, prepare_plot_data(alldata%>%filter(season==x), step=0.02)))
seasonsq <- do.call(rbind, seasonsq)

ggplot(seasonsq, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.03)+
  geom_text_contour(binwidth=0.03, min.size = 10)

ggplot(seasonsq, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.03)+
  geom_text_contour(binwidth=0.03, min.size = 10)


qmix2<-function(pwin, ploss, thedata, quotes){
  if (pwin+ploss>1.0)
    return (pwin, ploss, NA, NA, NA, NA, NA, NA, NA, NA)
  ord <- order(quotes[,1]-quotes[,3], quotes[,2], quotes[,3])
  thedata<-thedata[ord,]
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
  return (c(pwin=pwin, ploss=ploss, q=cutoffs, pistor=pistor, sky=sky, tcs=tcs, hwin=sum(qpred==1)/length(qpred), draw=sum(qpred==0)/length(qpred), away=sum(qpred==-1)/length(qpred)))
}
prepare_plot_data2<-function(thedata, step=0.01){
  #quotes <- 1/thedata[,c('B365H', 'B365D', 'B365A')]
  quotes <- 1/thedata[,c('BWH', 'BWD', 'BWA')]
  quotes <- quotes / rowSums(quotes)
  q<-c()  
  for (pwin in seq(0.4,0.9,step)) {
    for (ploss in seq(0.1-step, 1.00-pwin, step)) {
      q<-rbind(q, qmix2(pwin, ploss, thedata, quotes))  
    }
  }
  q <- data.frame(q)
  q<-unique(q %>% select(-pwin, -ploss))
  return(q)
}

gridscope<-levels(alldata$season)#[-2]
seasonsq <- lapply(gridscope, function(x) cbind(season=x, prepare_plot_data2(alldata%>%filter(season==x), step=0.02)))
seasonsq <- do.call(rbind, seasonsq)

ggplot(seasonsq, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.03)+
  geom_text_contour(binwidth=0.03, min.size = 10)

ggplot(seasonsq, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.03)+
  geom_text_contour(binwidth=0.03, min.size = 10)


q<-prepare_plot_data2(alldata %>% filter(season %in% train_seasons))
q<-prepare_plot_data2(alldata)

ggplot(q, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  #geom_tile()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-pistor) %>% head(10)

ggplot(q, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-sky) %>% head(10)

ggplot(q, aes(x=hwin, y=away, z=tcs, colour=tcs, fill=tcs, size=tcs))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_label_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-tcs) %>% head(10)




score_from_qpred<-function(qpred, thedata){
  hw<-thedata$FTHG>thedata$FTAG
  aw<-thedata$FTHG<thedata$FTAG
  draw<-thedata$FTHG==thedata$FTAG
  
  tendency<-qpred==sign(thedata$FTHG-thedata$FTAG)
  gdiff<-tendency & ((abs(thedata$FTHG-thedata$FTAG)==1) | draw)
  hit<-gdiff & ((thedata$FTHG+thedata$FTAG) %in% 2:3)
  pistor <- (sum(tendency)+sum(gdiff)+sum(hit))/length(qpred)
  sky <- (2*sum(tendency)+0*sum(gdiff)+3*sum(hit))/length(qpred)
  tcs <- (2*sum(tendency&hw)+1*sum(gdiff&hw)+1*sum(hit&hw)+4*sum(tendency&aw)+1*sum(gdiff&aw)+2*sum(hit&aw)+2*sum(tendency&draw)+4*sum(hit&draw))/length(qpred)
  return (data.frame(as.integer(tendency), as.integer(gdiff), as.integer(hit), pistor, sky, tcs, as.factor(qpred)))
}



qmix_draw<-function(pwin, pdraw, thedata, quotes){
  ord <- order(quotes[,2], quotes[,3], quotes[,1])
  thedata<-thedata[ord,]
  thedata$i <- 1:nrow(thedata)
  quotes<-quotes[ord,]
  
  cutoff_draw <- quantile(thedata$i, 1-pdraw, names=F) # c(ploss, 1-pwin)

  qqpred<-cut(thedata$i, c(-1e10, cutoff_draw, 1e10), labels=c("A0", "D")) # s+c(0.0, 0.00001)
  ord <- order(as.integer(qqpred), quotes[,1]-quotes[,3], quotes[,3], quotes[,2])
  thedata<-thedata[ord,]
  qqpred<-qqpred[ord]
  thedata$i <- 1:nrow(thedata)
  
  cutoff_win <- quantile(thedata$i, 1-pwin-pdraw, names=F)
  qqpred2<-cut(thedata$i, c(-1e10, cutoff_win, 1e10), labels=c("A", "H")) # s+c(0.0, 0.00001)
  
  qpred<-ifelse(qqpred=="D", 0, ifelse(qqpred2=="A", -1, 1))
  # print(c(pwin=pwin, pdraw=pdraw))
  # print(table(qpred)/length(qpred))
  
  hw<-thedata$FTHG>thedata$FTAG
  aw<-thedata$FTHG<thedata$FTAG
  draw<-thedata$FTHG==thedata$FTAG
  
  tendency<-qpred==sign(thedata$FTHG-thedata$FTAG)
  gdiff<-tendency & ((abs(thedata$FTHG-thedata$FTAG)==1) | draw)
  hit<-gdiff & ((thedata$FTHG+thedata$FTAG) %in% 2:3)
  pistor <- (sum(tendency)+sum(gdiff)+sum(hit))/length(qpred)
  sky <- (2*sum(tendency)+0*sum(gdiff)+3*sum(hit))/length(qpred)
  tcs <- (2*sum(tendency&hw)+1*sum(gdiff&hw)+1*sum(hit&hw)+4*sum(tendency&aw)+1*sum(gdiff&aw)+2*sum(hit&aw)+2*sum(tendency&draw)+4*sum(hit&draw))/length(qpred)
  #print(c(pistor, sky, tcs))
  #tt<-table(qpred)/length(qpred)
  return (c(pwin=pwin, pdraw=pdraw, q1=cutoff_draw, q2=cutoff_win, pistor=pistor, sky=sky, tcs=tcs, hwin=sum(qpred==1)/length(qpred), draw=sum(qpred==0)/length(qpred), away=sum(qpred==-1)/length(qpred)))
}

prepare_plot_data_draw<-function(thedata, step=0.01){
  #quotes <- 1/thedata[,c('B365H', 'B365D', 'B365A')]
  quotes <- 1/thedata[,c('BWH', 'BWD', 'BWA')]
  quotes <- quotes / rowSums(quotes)
  q<-c()  
  for (pwin in seq(0.4,0.99,step)) {
    for (pdraw in seq(0.0, 1.00-pwin, step)) {
      q<-rbind(q, qmix_draw(pwin, pdraw, thedata, quotes))  
    }
  }
  q <- data.frame(q)
  q<-unique(q %>% select(-pwin, -pdraw))
  return(q)
}

gridscope<-levels(alldata$season)#[-2]
seasonsq <- lapply(gridscope, function(x) cbind(season=x, prepare_plot_data_draw(alldata%>%filter(season==x), step=0.02)))
seasonsq <- do.call(rbind, seasonsq)

ggplot(seasonsq, aes(x=hwin, y=draw, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.1, min.size = 10)

ggplot(seasonsq, aes(x=hwin, y=draw, z=sky, colour=sky, fill=sky, size=sky))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.03)+
  geom_text_contour(binwidth=0.03, min.size = 10)


q<-prepare_plot_data_draw(alldata)
q<-prepare_plot_data2(alldata)

ggplot(q, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  #geom_tile()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-pistor) %>% head(10)

ggplot(q, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-sky) %>% head(10)



library(MASS)
library(caret)

thedata <- alldata
quotes <- 1/thedata[,c('BWH', 'BWD', 'BWA')]
quotes <- quotes / rowSums(quotes)

set.seed(1)

trans = preProcess(quotes, 
                   c("BoxCox", "center", "scale"))
predictorsTrans = data.frame(trans = predict(trans, quotes))
summary(predictorsTrans)
var(predictorsTrans) 
cor(predictorsTrans)
plot(predictorsTrans, col=thedata$FTR)
hist(predictorsTrans$trans.BWD)


ldamodel <- lda(FTR ~ ., data=cbind(FTR=thedata$FTR, predictorsTrans), CV = T)

table(ldamodel$class, alldata$FTR)/length(ldamodel$class)
sum(diag(table(ldamodel$class, alldata$FTR)))/length(ldamodel$class)

ldamodel <- lda(FTR ~ ., data=cbind(FTR=thedata$FTR, predictorsTrans), CV = F, prior=c(0.2746764, 0.2896134, 0.4357102))
plda <- predict(ldamodel, newdata = predictorsTrans)
prop <- ldamodel$svd^2/sum(ldamodel$svd^2)
print(prop)

table(plda$class)
table(plda$class, alldata$FTR)/length(plda$class)
sum(diag(table(plda$class, alldata$FTR)))/length(plda$class)

dataset <- data.frame(FTR = alldata$FTR, lda = plda$x)
centr <- predict(ldamodel, newdata = data.frame(ldamodel$means))

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = FTR, shape = FTR), size = 2.5, alpha=0.4) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  geom_point(data=data.frame(centr$x, FTR=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTR), size=10, pch=4)

# print decision boundary

score_from_qpred(as.integer(plda$class)-2, thedata)

qdamodel <- qda(FTR ~ ., data=cbind(FTR=thedata$FTR, predictorsTrans), CV = F)
qlda <- predict(qdamodel, newdata = predictorsTrans)
str(qlda)
qdamodel$ldet
qdamodel$scaling

print(prop)


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

x <- data.frame(predictorsTrans[,c(2,3)], FTR=thedata$FTR)
x <- data.frame(plda$x, FTR=thedata$FTR)

model <- lda(FTR ~ ., data=x, prior=c(0.2746764, 0.2896134, 0.4357102))
decisionplot(model, x, class = "FTR", main = "LDA")
summary(score_from_qpred(as.integer(predict(model, newdata=x)$class)-2, thedata))

model <- qda(FTR ~ ., data=x, prior=c(0.2546764, 0.3296134, 0.4157102))
decisionplot(model, x, class = "FTR", main = "QDA")
summary(score_from_qpred(as.integer(predict(model, newdata=x)$class)-2, thedata))

model$scaling

x <- data.frame(predict(lfdamodel, newdata=quotes), FTR=a0$FTR)
x<-x %>% dplyr::select(-X1)
thedata<-a0
library("rpart")
model <- rpart(FTR ~ ., data=x)
decisionplot(model, x, class = "FTR", main = "CART")
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

library(C50)
model <- C5.0(FTR ~ ., data=x)
decisionplot(model, x, class = "FTR", main = "C50")
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

library(randomForest)
model <- randomForest(FTR ~ ., data=x, ntree=50, maxnodes=10)
decisionplot(model, x, class = "FTR", main = "RF")
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

library(e1071)
model <- svm(FTR ~ ., data=x, kernel="linear")
decisionplot(model, x, class = "FTR", main = "SVD (linear)")  
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

model <- svm(FTR ~ ., data=x, kernel="radial")
decisionplot(model, x, class = "FTR", main = "SVD (radial)")  
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

model <- svm(FTR ~ ., data=x, kernel="polynomial")
decisionplot(model, x, class = "FTR", main = "SVD (polynomial)")  
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

model <- svm(FTR ~ ., data=x, kernel="sigmoid")
decisionplot(model, x, class = "FTR", main = "SVD (sigmoid)")  
summary(score_from_qpred(as.integer(predict(model, newdata=x, type="class"))-2, thedata))

library(nnet)
model <- nnet(FTR ~ ., data=x, size = 20, maxit = 1000, trace = FALSE, censored=F, linout=F, entropy=T, decay=0.01)
decisionplot(model, x, class = "FTR", main = "NN (1)")
summary(score_from_qpred(as.integer(as.factor(predict(model, newdata=x, type="class")))-2, thedata))

library(lfda)
#install.packages("lfda")
metric = c("orthonormalized", "plain", "weighted")
model <- lfda(predictorsTrans, thedata$FTR, r=3,  metric = metric, knn = 20)

model <- lfda(quotes, thedata$FTR, r=3,  metric = metric, knn = 20)
model <- lfda(quotes[thedata$season=="1819",], thedata$FTR[thedata$season=="1819"], r=1,  metric = metric, knn = 20)
summary(model$Z)
hist(model$Z[,1])
hist(model$Z[,2])
plot(model$Z, col=as.integer(thedata$FTR)+1)
plot(model$Z[,2:3], col=as.integer(thedata$FTR)+1)
plot(model$Z[,c(1,3)], col=as.integer(thedata$FTR)+1)

ggplot(data.frame(X=model$Z, FTR=thedata$FTR[thedata$season=="1819"]), aes(x=X, fill=FTR))+geom_density(alpha=0.4)
ggplot(data.frame(model$Z, FTR=thedata$FTR[thedata$season=="1819"]), aes(x=X2, fill=FTR))+geom_density(alpha=0.4)
ggplot(data.frame(model$Z, FTR=thedata$FTR[thedata$season=="1819"]), aes(x=X3, fill=FTR))+geom_density(alpha=0.4)
ggplot(data.frame(model$Z, FTR=thedata$FTR[thedata$season=="1819"]), aes(x=X1, fill=FTR))+geom_histogram()
ggplot(data.frame(model$Z, FTR=thedata$FTR[thedata$season=="1819"]), aes(x=X2, fill=FTR))+geom_histogram()
ggplot(data.frame(model$Z, FTR=thedata$FTR[thedata$season=="1819"]), aes(x=X3, fill=FTR))+geom_histogram()

qmix_lfda<-function(pwin, ploss, thedata, quotes){
  #ord <- order(quotes[,1], quotes[,2], quotes[,3])
  ord <- do.call(order, quotes) # varargs ... uses lexicagraphic sorting
  thedata<-thedata[ord,]
  quotes<-quotes[ord,]
  n<-nrow(thedata)
  #print(n)
  #print(thedata$FTR[1:floor(n/2)])
  # print(c(sum(as.integer(thedata$FTR[1:floor(n/2)]=="H")), sum(as.integer(thedata$FTR[floor(n/2):n]=="H"))))
  # if (sum(as.integer(thedata$FTR[1:floor(n/2)]=="H"))>sum(as.integer(thedata$FTR[floor(n/2):n]=="H"))){
  #   thedata<-thedata[n:1,]
  #   quotes<-quotes[n:1,]    
  # }
  # print(c(sum(as.integer(thedata$FTR[1:floor(n/2)]=="H")), sum(as.integer(thedata$FTR[floor(n/2):n]=="H"))))
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
  #print(c(pistor, sky, tcs))
  #tt<-table(qpred)/length(qpred)
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


train_seasons<-c("0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718")
test_seasons<-c("1819")

train_seasons<-c("0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819")
test_seasons<-c("0405", "0506", "0607", "0708")

train_seasons<-c("0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314")
test_seasons<-c( "1415", "1516", "1617", "1718", "1819")

train_seasons<-sample(levels(alldata$season), 8)
test_seasons<-setdiff(levels(alldata$season), train_seasons)

quote_names<-c('BWH', 'BWD', 'BWA')
quote_names<-c('B365H', 'B365D', 'B365A')
quote_names<-c('BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')

gridscope<-levels(alldata$season)#[-2]

gridscope<-train_seasons
traindata <- alldata%>%filter(season %in% gridscope)
#traindata <- alldata%>%filter(!spieltag %in% 10:24)
#traindata <- alldata%>%filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")

# quotes <- 1/traindata[,quote_names]
# #quotes <- quotes / rowSums(quotes)
# quotes[,1:3] <- quotes[,1:3] / rowSums(quotes[,1:3])
# quotes[,4:6] <- quotes[,4:6] / rowSums(quotes[,4:6])

testdata <- alldata%>%filter(season %in% test_seasons)
#testdata <- alldata%>%filter(spieltag %in% 10:24)
#testdata <- alldata%>%filter(HomeTeam=="Bayern Munich" | AwayTeam=="Bayern Munich")
# testquotes <- 1/testdata[,quote_names]
# #testquotes <- testquotes / rowSums(testquotes)
# testquotes[,1:3] <- testquotes[,1:3] / rowSums(testquotes[,1:3])
# testquotes[,4:6] <- testquotes[,4:6] / rowSums(testquotes[,4:6])

feature_columns<-c(normalized_quote_names, "draw_prior")
quotes<-traindata[, feature_columns]
testquotes<-testdata[, feature_columns]

metric = c("orthonormalized", "plain", "weighted")
trans = preProcess(quotes, c("BoxCox", "center", "scale"))
quotes <- data.frame(trans = predict(trans, quotes))
ggplot(melt(quotes))+facet_wrap(variable~.)+geom_histogram(aes(x=value), bins=40)

testquotes <- data.frame(trans = predict(trans, testquotes))
ggplot(melt(testquotes))+facet_wrap(variable~.)+geom_histogram(aes(x=value), bins=40)

model <- lfda(quotes, traindata$FTR, r=3,  metric = metric, knn = 20)

rownames(model$T)<-feature_columns
print(model$T)
print(model)

# plot(model$Z, col=as.integer(thedata$FTR)+1)
# plot(model$Z[,2:3], col=as.integer(thedata$FTR)+1)
# plot(model$Z[,c(1,3)], col=as.integer(thedata$FTR)+1)
# ggplot(data.frame(model$Z, FTR=thedata$FTR), aes(x=X1, fill=FTR))+geom_density(alpha=0.4) #+geom_histogram()
traindata<-data.frame(traindata, model$Z)
testdata<-data.frame(testdata, predict(model, testquotes))
# move HomeWins to high end of scale
orientation<-traindata%>%group_by(FTR)%>%summarise(X1=median(X1), X2=median(X2), X3=median(X3))%>%filter(FTR %in% c("H", "A"))%>%mutate_at(vars(X1:X3), rank)%>%mutate_at(vars(X1:X3), function(x) 2*(x-1.5))%>%filter(FTR=="H")
#orientation<-traindata%>%group_by(FTR)%>%summarise(X1=median(X1))%>%filter(FTR %in% c("H", "A"))%>%mutate_at(vars(X1), rank)%>%mutate_at(vars(X1), function(x) 2*(x-1.5))%>%filter(FTR=="H")
print(orientation)
traindata$X1<-traindata$X1*orientation$X1
testdata$X1<-testdata$X1*orientation$X1
traindata$X2<-traindata$X2*orientation$X2
traindata$X3<-traindata$X3*orientation$X3
testdata$X2<-testdata$X2*orientation$X2
testdata$X3<-testdata$X3*orientation$X3

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

plot(FTR~X2, data=traindata, main="Train Data")
plot(FTR~X2, data=testdata, main="Test Data")
plot(FTR~X3, data=traindata, main="Train Data")
plot(FTR~X3, data=testdata, main="Test Data")


ggplot(testdata, aes(x=X1, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=20)
ggplot(testdata, aes(x = FTR, y=X1, color=FTR)) + geom_violin(draw_quantiles = c(0.1, 0.25, 0.5, 0.75, 0.9), trim=F)+ coord_flip()
# ggplot(testdata, aes(x=X2, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=20)
# ggplot(testdata, aes(x=X3, fill=FTR))+geom_density(alpha=0.4) +geom_histogram(bins=20)
# ggplot(testdata, aes(x=X1, y=X2, colour=FTR))+geom_point(alpha=0.4)
# ggplot(testdata, aes(x=X1, y=X3, colour=FTR))+geom_point(alpha=0.4)

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


q<-prepare_plot_data_lfda(traindata)
qtest<-prepare_plot_data_lfda(testdata)
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

a0<-alldata
a0$draw_prior<-rep(ppp, each=9)

#install.packages("tidyimpute")
library(tidyimpute)
library(na.tools)

a0<-a0%>%impute(draw_prior=na.mean)

summary(a0$draw_prior)
quotes<-a0%>%dplyr::select(pBWH:draw_prior)

library(lfda)
#install.packages("lfda")
metric = c("orthonormalized", "plain", "weighted")
lfdamodel <- lfda(quotes, a0$FTR, r=3,  metric = metric, knn = 20)

summary(lfdamodel$Z)
hist(lfdamodel$Z[,1])
hist(lfdamodel$Z[,2])
plot(lfdamodel$Z, col=as.integer(a0$FTR)+1)
plot(lfdamodel$Z[,2:3], col=as.integer(a0$FTR)+1)
plot(lfdamodel$Z[,c(1,3)], col=as.integer(a0$FTR)+1)

predict(lfdamodel, newdata=quotes)


###############


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

point_system<-"pistor"
#point_system<-"sky"
newquotes <- 1/newdata
print(newquotes)
print(rowSums(newquotes))
#newquotes <- newquotes / rowSums(newquotes)
newquotes[,1:3] <- newquotes[,1:3] / rowSums(newquotes[,1:3])
newquotes[,4:6] <- newquotes[,4:6] / rowSums(newquotes[,4:6])

if (T){
  newquotes<-as.matrix(tail(alldata%>%dplyr::select(feature_columns), 9))
}  

newquotes <- data.frame(trans = predict(trans, newquotes))

# newdata<-data.frame(X1=predict(model, newquotes))
# newdata$X1<-newdata$X1*orientation$X1

newdata<-data.frame(predict(model, newquotes))
newdata$X1<-newdata$X1*orientation$X1
newdata$X2<-newdata$X2*orientation$X2
newdata$X3<-newdata$X3*orientation$X3

l <- nrow(traindata)
plot(seq_along(traindata$X1)/l, sort(traindata$X1), axes=F, col="forestgreen", pch=".", type="l")
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


#library(ggpmisc)
#install.packages("ggpmisc")
#ggpmisc:::find_peaks(y)

#cbind(traindata$FTHG, traindata$FTAG, p_points(0,0,traindata$FTHG, traindata$FTAG)$sky)

################################################################################################################

feature_columns<-c(normalized_quote_names, "draw_prior")


qmix_lda<-function(model, thedata, predictorsTrans){
  qqpred<-predict(model, newdata=predictorsTrans)
  qpred<-as.integer(qqpred$class)-2
  # print(c(pwin=pwin, pdraw=pdraw))
  print(table(qpred)/length(qpred))
  
  hw<-thedata$FTHG>thedata$FTAG
  aw<-thedata$FTHG<thedata$FTAG
  draw<-thedata$FTHG==thedata$FTAG
  
  tendency<-qpred==sign(thedata$FTHG-thedata$FTAG)
  gdiff<-tendency & ((abs(thedata$FTHG-thedata$FTAG)==1) | draw)
  hit<-gdiff & ((thedata$FTHG+thedata$FTAG) %in% 2:3)
  pistor <- (sum(tendency)+sum(gdiff)+sum(hit))/length(qpred)
  sky <- (2*sum(tendency)+0*sum(gdiff)+3*sum(hit))/length(qpred)
  tcs <- (2*sum(tendency&hw)+1*sum(gdiff&hw)+1*sum(hit&hw)+4*sum(tendency&aw)+1*sum(gdiff&aw)+2*sum(hit&aw)+2*sum(tendency&draw)+4*sum(hit&draw))/length(qpred)
  #print(c(pistor, sky, tcs))
  #tt<-table(qpred)/length(qpred)
  return (data.frame(pistor=pistor, sky=sky, tcs=tcs, hwin=sum(qpred==1)/length(qpred), draw=sum(qpred==0)/length(qpred), away=sum(qpred==-1)/length(qpred)))
}

prepare_plot_data_lda<-function(thedata, step=0.01){
  #quotes <- 1/thedata[,c('B365H', 'B365D', 'B365A')]
  #quotes <- 1/thedata[,c('BWH', 'BWD', 'BWA')]
  #quotes <- quotes / rowSums(quotes)
  quotes <- thedata[,feature_columns]

  trans = preProcess(quotes, c("BoxCox", "center", "scale"))
  predictorsTrans <- data.frame(trans = predict(trans, quotes))
  #ldatransmodel <- lda(FTR ~ ., data=cbind(FTR=thedata$FTR, predictorsTrans), CV = F)
  metric = c("orthonormalized", "plain", "weighted")
  ldatransmodel <- lfda(predictorsTrans, thedata$FTR, r=3,  metric = metric, knn = 20)
  
  predictorsTrans <- as.data.frame(predict(ldatransmodel, newdata = predictorsTrans))
  dataX <- data.frame(predictorsTrans, FTR=thedata$FTR)
  q<-c()  
  for (pwin in seq(0.1,0.6,step)) {
    for (pdraw in seq(0.25, min(1.00-pwin, 0.4), step)) {
      ploss<-1.0-pwin-pdraw
      prior<-c(ploss, pdraw, pwin)
      print(prior)
      model<-lda(FTR~., data=dataX, prior=prior)
      result <- qmix_lda(model, thedata, predictorsTrans)
      q<-rbind(q, result)  
    }
  }
  q <- data.frame(q)
  q<-unique(q)
  return(q)
}
q<-prepare_plot_data_lda(alldata, step=0.05)

gridscope<-levels(alldata$season)#[-2]
seasonsq <- lapply(gridscope, function(x) cbind(season=x, prepare_plot_data_lda(alldata%>%filter(season==x), step=0.02)))
seasonsq <- do.call(rbind, seasonsq)

ggplot(seasonsq, aes(x=hwin, y=draw, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.1, min.size = 10)

ggplot(seasonsq, aes(x=hwin, y=draw, z=sky, colour=sky, fill=sky, size=sky))+
  facet_wrap(aes(season))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.03)+
  geom_text_contour(binwidth=0.03, min.size = 10)


q<-prepare_plot_data_lda(alldata, step=0.005)
#q<-prepare_plot_data2(alldata)

ggplot(q, aes(x=hwin, y=away, z=pistor, colour=pistor, fill=pistor, size=pistor))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  #geom_tile()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-pistor) %>% head(10)

ggplot(q, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_contour(binwidth=0.01)+
  geom_text_contour(binwidth=0.01, min.size = 5)
q %>% arrange(-sky) %>% head(10)





################################################################################################################


  

evalpoints<-function(cutoffs, thedata, quotes){
  qqpred<-cut(quotes[,1]-quotes[,3]+0.1*quotes[,2], c(-1e10, cutoffs+c(0.0, 0.00001), 1e10), labels=c("A", "D", "H"))
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


# testdata <- alldata [alldata$season %in% test_seasons,]
# testquotes <- 1/testdata[,c('B365H', 'B365D', 'B365A')]
# testquotes <- 1/testdata[,c('BWH', 'BWD', 'BWA')]
# testquotes <-testquotes / rowSums(testquotes)
# 
# qtest<-c()  
# for (pwin in seq(0.3,0.9,0.01)) {
#   for (ploss in seq(0.09,1.00-pwin,0.01)) {
#     qtest<-rbind(qtest, qmix(pwin, ploss, testdata, testquotes))  
#   }
# }
# qtest <- data.frame(qtest)
# qtest<-unique(qtest %>% select(-pwin, -ploss))

qtest<-prepare_plot_data(alldata %>% filter(season %in% test_seasons))

colist <- q %>% arrange(-pistor) %>% head(20) %>% select(q1, q2)
q2<-apply(colist, 1, evalpoints, testdata, testquotes)
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
  geom_text(data=q2, size=3, colour="red", check_overlap = T, show.legend=F, aes(label=round(pistor, 3)))
  
q %>% arrange(-pistor) %>% head(20) %>% merge(q2, by=c("q1","q2")) %>% mutate(r=rank(pistor.y)) %>% arrange(-pistor.x) %>% select(-sky.x, -sky.y, -tcs.x, -tcs.y)

colist <- q %>% arrange(-sky) %>% head(20) %>% select(q1, q2)
q2<-apply(colist, 1, evalpoints, testdata, testquotes)
q2 <- data.frame(t(q2))
ggplot(q2, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky, label=round(sky, 4)))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_point()+
  geom_text(size=3, col="black", nudge_y = 0.002, check_overlap = T)

ggplot(q, aes(x=hwin, y=away, z=sky, colour=sky, fill=sky, size=sky))+
  scale_fill_gradientn(colours = rev(rainbow(10)))+scale_colour_gradientn(colours = rev(rainbow(10)))+
  geom_tile(alpha=0.2)+
  geom_point(data=qtest)+
  geom_text(data=q2, size=3, colour="maroon", check_overlap = T, show.legend=F, aes(label=round(sky, 3)))+
  geom_contour(binwidth=0.01, size=1)+
  geom_text_contour(binwidth=0.01, min.size = 10, size=6, colour="blue")+
  geom_contour(data=qtest, binwidth=0.02, col="red", size=1)+
  geom_text_contour(data=qtest, binwidth=0.02, min.size = 10, col="red", size=6)

q %>% arrange(-sky) %>% head(20) %>% merge(q2, by=c("q1","q2")) %>% mutate(r=rank(sky.y)) %>% arrange(-sky.x) %>% select(-pistor.x, -pistor.y, -tcs.x, -tcs.y)

qqplot(quotes[,1]-quotes[,3], testquotes[,1]-testquotes[,3])
abline(a=0, b=1, col="red")

plot(sort(quotes[,1]-quotes[,3]), new=T, ylim=c(-1,1))
par(new=TRUE)
plot(sort(testquotes[,1]-testquotes[,3]), col="red", ylim=c(-1,1))

boxplot(1/BWH-1/BWA ~ season, data=alldata)
boxplot(1/BWH+1/BWA+1/BWD ~ season, data=alldata)
boxplot((1/BWH-1/BWA)/(1/BWH+1/BWA+1/BWD) ~ season, data=alldata)

boxplot((1/B365H-1/B365A) ~ season, data=alldata)
boxplot((1/B365H-1/B365A)/(1/B365H+1/B365A+1/B365D) ~ season, data=alldata)
boxplot(1/B365H+1/B365A+1/B365D ~ season, data=alldata)

plot(alldata %>% group_by(season) %>% summarise(meanp = mean((1/BWH-1/BWA)/(1/BWH+1/BWA+1/BWD))))

boxplot((1/B365H-1/B365A)/(1/B365H+1/B365A+1/B365D) ~ FTR+season, data=alldata, col=2:4)

ggplot(alldata, aes(x=season, y=(1/B365H-1/B365A)/(1/B365H+1/B365A+1/B365D), fill=FTR)) + 
  geom_boxplot(varwidth = T)

ggplot(alldata, aes(x=(1/B365H-1/B365A)/(1/B365H+1/B365A+1/B365D))) + 
  geom_density(aes(fill=FTR), alpha=0.3)

ggplot(alldata, aes(x=(1/BWH-1/BWA)/(1/BWH+1/BWA+1/BWD))) + facet_wrap(aes(season))+
  geom_density(aes(fill=FTR), alpha=0.3)

ggplot(alldata, aes(x=(1/BWH-1/BWA)/(1/BWH+1/BWA+1/BWD))) + facet_wrap(aes(season))+
  geom_histogram(aes(fill=FTR), bins = 30)


#gof[hit,]
table(thedata[hit,c("FTHG", "FTAG")])
table(thedata[gdiff,c("FTHG", "FTAG")])
table(thedata[tendency,c("FTHG", "FTAG")])
#quotes[drawpred,]

alldata %>% filter(season=="0506")%>%group_by(FTHG, FTAG)%>%count()


####################################################
# analyze expected points for correct bet

n<-with(alldata, table(FTR))
s<-with(alldata, table(FTAG, FTHG, FTR))
2+3*s[,,1]/n[1]
2+3*s[,,2]/n[2]
2+3*s[,,3]/n[3]

d<-with(alldata, table(FTHG-FTAG, FTR))
alldata%>%mutate(diff=FTHG-FTAG)%>%group_by(FTHG, FTAG, FTR)%>%count()
1+s[,,1]/n[1]+d[,1]/n[1]
d[,2]/n[2]
d[,3]/n[3]

alldata%>%select(FTHG, FTAG, FTR)%>%mutate(diff=FTHG-FTAG)%>%
  group_by(FTHG, FTAG, FTR)%>%mutate(n3=n())%>%ungroup()%>%
  group_by(diff, FTR)%>%mutate(n2=n())%>%ungroup()%>%
  group_by(FTR)%>%mutate(n1=n())%>%ungroup()%>%
  unique() %>%
  mutate(sky=2+3*n3/n1, pistor=1+n2/n1+n3/n1) %>%
  arrange(FTR, pistor) %>%
  group_by(FTR) %>% summarise(sky=max(sky), pistor=max(pistor)) %>%
  print.data.frame()

alldata%>%filter(HomeTeam!="Bayern Munich" & AwayTeam!="Bayern Munich")%>%
  select(FTHG, FTAG, FTR)%>%mutate(diff=FTHG-FTAG)%>%
  group_by(FTHG, FTAG, FTR)%>%mutate(n3=n())%>%ungroup()%>%
  group_by(diff, FTR)%>%mutate(n2=n())%>%ungroup()%>%
  group_by(FTR)%>%mutate(n1=n())%>%ungroup()%>%
  unique() %>%
  mutate(sky=2+3*n3/n1, pistor=1+n2/n1+n3/n1) %>%
  arrange(FTR, pistor) %>%
  group_by(FTR) %>% summarise(sky=max(sky), pistor=max(pistor)) %>%
  print.data.frame()

# even matches 
alldata%>%filter(BWD<=3.5)%>%
  select(FTHG, FTAG, FTR)%>%mutate(diff=FTHG-FTAG)%>%
  group_by(FTHG, FTAG, FTR)%>%mutate(n3=n())%>%ungroup()%>%
  group_by(diff, FTR)%>%mutate(n2=n())%>%ungroup()%>%
  group_by(FTR)%>%mutate(n1=n())%>%ungroup()%>%
  unique() %>%
  mutate(sky=2+3*n3/n1, pistor=1+n2/n1+n3/n1) %>%
  arrange(FTR, pistor) %>%
  group_by(FTR) %>% summarise(sky=max(sky), pistor=max(pistor)) %>%
  print.data.frame()

# home-biased matches 
alldata%>%filter(BWH<=2)%>%
  select(FTHG, FTAG, FTR)%>%mutate(diff=FTHG-FTAG)%>%
  group_by(FTHG, FTAG, FTR)%>%mutate(n3=n())%>%ungroup()%>%
  group_by(diff, FTR)%>%mutate(n2=n())%>%ungroup()%>%
  group_by(FTR)%>%mutate(n1=n())%>%ungroup()%>%
  unique() %>%
  mutate(sky=2+3*n3/n1, pistor=1+n2/n1+n3/n1) %>%
  arrange(FTR, pistor) %>%
  group_by(FTR) %>% summarise(sky=max(sky), pistor=max(pistor)) %>%
  print.data.frame()

# away-biased matches 
alldata%>%filter(BWA<=2)%>%
  select(FTHG, FTAG, FTR)%>%mutate(diff=FTHG-FTAG)%>%
  group_by(FTHG, FTAG, FTR)%>%mutate(n3=n())%>%ungroup()%>%
  group_by(diff, FTR)%>%mutate(n2=n())%>%ungroup()%>%
  group_by(FTR)%>%mutate(n1=n())%>%ungroup()%>%
  unique() %>%
  mutate(sky=2+3*n3/n1, pistor=1+n2/n1+n3/n1) %>%
  arrange(FTR, pistor) %>%
  group_by(FTR) %>% summarise(sky=max(sky), pistor=max(pistor)) %>%
  print.data.frame()


summary(alldata$BWH)

with(alldata, plot(log(BWH), log(BWA)), col=as.integer(FTR))
with(alldata, boxplot(log(BWH)~FTR))
with(alldata, boxplot(log(BWA)~FTR))
with(alldata, boxplot(log(BWD)~FTR))

with(alldata, plot((BWA)~(BWH), col=as.integer(FTR), pch=as.integer(FTR)))
with(alldata, plot(log(BWA)~log(BWH), col=as.integer(FTR), pch=as.integer(FTR)))
with(alldata, plot(log(BWD)~log(BWH), col=as.integer(FTR), pch=as.integer(FTR)))
with(alldata, plot(log(BWD)~log(BWA), col=as.integer(FTR), pch=as.integer(FTR)))
with(alldata, plot(BWD~(BWA), col=as.integer(FTR), pch=as.integer(FTR)))

with(alldata, plot(log(BWA)+as.integer(FTR)-1~log(BWH), col=as.integer(FTR), pch=as.integer(FTR)))

with(alldata, plot((BWA)~(BWH), col=as.integer(FTR), pch=as.integer(FTR)))

with(alldata, smoothScatter(log(BWA), log(BWH), col=as.integer(FTR), ylim=c(0,5)))
with(alldata, points(as.integer(FTR)-1+log(BWA)+rnorm(nrow(alldata))*0.02~log(BWH), col=as.integer(FTR), pch=as.integer(FTR)))

library(Hmisc)
with(alldata, table(cut2(BWA, m = length(BWA)/10)))


library(sm)
with(alldata,   sm.density.compare(log(BWA)-log(BWH), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWA)+log(BWH), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWA), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWH), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWD)-1.3, FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWD), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWH)^log(BWA), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWA)*(log(BWD)-1.3), FTR, model = "equal"))
with(alldata,   sm.density.compare(log(BWH)^(+0.2+log(BWA)), FTR, model = "equal"))
with(alldata,   sm.density.compare((log(BWH)^(+0.2+log(BWA)))-0.1*log(BWA)^(0.7+0.3*log(BWH)), FTR, model = "none"))
with(alldata,   sm.density.compare(log(BWH)*log(BWA)-log(BWH)-log(BWA), FTR, model = "equal"))

with(alldata,   sm.density.compare((1/log(BWH)-1/log(BWA)), FTR, model = "none"))

with(alldata, smoothScatter(log(BWH)^(+0.2+log(BWA)), log(BWA)-log(BWH), col=as.integer(FTR)))
with(alldata, points(as.integer(FTR)-2+log(BWA)-log(BWH)+rnorm(nrow(alldata))*0.02~I(log(BWH)^(+0.2+log(BWA))), col=as.integer(FTR), pch=as.integer(FTR)))

library(deldir)
with(alldata, deldir(log(BWA), log(BWH), sort=TRUE, plotit=T))

#dxy<-with(alldata, deldir(log(BWA), log(BWH), dpl=list(ndx=3, ndy=3, nrad=1, nper=1), z=z, sort=TRUE, plotit=T))
#plot(dxy, wlines=c('tess'), wpoints=c('both'), col=1:5)

dt<-alldata
dt$x0<-(log(dt$BWH)^(+0.2+log(dt$BWA)))
with(dt, plot(FTR~x0))
dt$x0<-with(dt, (log(BWH)^(+0.2+log(BWA)))+0.6*log(BWA)^(0.7+0.3*log(BWH)))
dt$x0<-log(dt$BWH)-log(dt$BWA)
dt$x0<-atan((log(dt$BWA)-3.9)/(log(dt$BWH)-3.4))
dt$r0<-sqrt((log(dt$BWA)-3.0)^2 + (log(dt$BWH)-3)^2)

x<-seq(0.1,3,0.01)
y<-3.12-sqrt(3^2-(x*1.05-3.1)^2)
points(x,y,type="l")

dt$x0<-atan((log(dt$BWA)-3.12)/(1.05*log(dt$BWH)-3.1))
dt$r0<-sqrt((log(dt$BWA)-3.12)^2 + (1.05*log(dt$BWH)-3.1)^2)
summary(dt$x0)

dt$x0<--log(dt$BWA)
hist(dt$x0)

#############################################################################
dt<-alldata
dt$x<-log(dt$BWH)
dt$y<-log(dt$BWA)
dt$x<-(1/dt$BWA)/(1/dt$BWH+1/dt$BWD+1/dt$BWA)
dt$y<-1/dt$BWH/(1/dt$BWH+1/dt$BWD+1/dt$BWA)
#dt$x<-log(dt$B365H)
#dt$y<-log(dt$B365A)
dt$x0<-dt$x-dt$y
with(dt, plot(FTR~x0))
with(dt, plot(y~x))
pistor<-function(hg1, ag1, hg2, ag2){
  (sign(hg1-ag1)==sign(hg2-ag2))+((hg1-ag1)==(hg2-ag2))+((hg1==hg2)&(ag1==ag2))
}
sky<-function(hg1, ag1, hg2, ag2){
  2*(sign(hg1-ag1)==sign(hg2-ag2))+0*((hg1-ag1)==(hg2-ag2))+3*((hg1==hg2)&(ag1==ag2))
}


point.system<-sky
point.system<-pistor
steps<-data.frame(hg=c(2,1), ag=c(1,2))
steps<-data.frame(hg=c(3,2,2,1,1,0,0), ag=c(0,0,1,1,2,2,3))
steps<-data.frame(hg=c(4,2,2,1,1,0,1), ag=c(1,0,1,1,2,2,4))
steps<-data.frame(hg=c(2,2,1,1,0), ag=c(0,1,1,2,2))
#steps<-data.frame(hg=c(2,2,1,1,1,0), ag=c(0,1,0,1,2,2))

steps<-data.frame(hg=c(3,2,2,1,0,0), ag=c(0,0,1,2,2,3))
steps<-data.frame(hg=c(2,2,1,0), ag=c(0,1,2,2))
steps<-data.frame(hg=c(4,3,2,2,1,1,0,0), ag=c(0,0,0,1,1,2,2,3))
steps<-data.frame(hg=c(4,3,2,2,1,0,0), ag=c(0,0,0,1,2,2,3))
#steps<-data.frame(hg=c(4,3,2,2,1,1,0,0, 0), ag=c(0,0,0,1,1,2,2,3,4))

cuts<-data.frame(steps, hg0=lag(steps$hg), ag0=lag(steps$ag))[-1,]
cutpoints<-apply(cuts, 1, function(y) optimize(
  function(x) mean(ifelse(dt$x0<x, 
                          point.system(y[["hg0"]], y[["ag0"]], dt$FTHG, dt$FTAG),
                          point.system(y[["hg"]], y[["ag"]], dt$FTHG, dt$FTAG)))
  , range(dt$x0), maximum = T)$maximum)
cuts<-cbind(cuts, cutpoints)
st<-data.frame(steps, x0min=c(min(dt$x0), cutpoints), x0max=c(cutpoints, max(dt$x0)), hg_prev=lag(steps$hg), ag_prev=lag(steps$ag), hg_lead=lead(steps$hg), ag_next=lead(steps$ag))
st<-st%>%arrange(x0min)
print(st)

# dt<-alldata%>%filter(season=="0506")
# dt$x<-(1/dt$BWA)/(1/dt$BWH+1/dt$BWD+1/dt$BWA)
# dt$y<-1/dt$BWH/(1/dt$BWH+1/dt$BWD+1/dt$BWA)
# dt$x0<-dt$x-dt$y

dt$c0<-cut(dt$x0, breaks = c(st$x0min, max(dt$x0)), labels = paste(st$hg, st$ag, sep=":"), include.lowest = T) # , 
dt$hg0<-st$hg[as.integer(dt$c0)]
dt$ag0<-st$ag[as.integer(dt$c0)]
#dt%>%select(hg0, ag0, c0, x0, FTHG, FTAG)%>%print.data.frame()
dt$points<-point.system(dt$FTHG, dt$FTAG, dt$hg0, dt$ag0)

print(mean(dt$points))
#dt%>%mutate(pred=paste(hg0, ag0, sep=":"), label=paste(FTHG, FTAG, sep=":"))%>%select(label, pred, points)%>%table()
print(st)
#cbind(steps, levels(dt$c0))
table(dt$c0)
table(dt$c0, dt$points)
plot(table(dt$c0, dt$points))
dt%>%group_by(c0)%>%summarise(n=n(), points=mean(points))
print(mean(dt$points))
with(dt, plot(x,y,col=c0, pch=ifelse((hg0==FTHG)&(ag0==FTAG), 1, 4), 
              main=mean(dt$points)))
legend("topright", legend = levels(dt$c0), text.col = 1:nlevels(dt$c0))
l<-dt%>%group_by(c0)%>%summarize(x=max(x), y=min(y), x0min=min(x0), x0=max(x0))
text(x=l$x+0.1+0.1/(1.5+l$x),y=l$y+0.05,labels=paste(as.character(l$c0), l$x0), pos=3)

py<-predict(lm(y~poly(x,5), data=dt))
points(dt$x, py, col="red")

with(dt%>%filter(season=="1718", abs(FTHG-FTAG)<=3), boxplot((1/BWD+1/BWH-1/BWA)/(1/BWH+1/BWD+1/BWA)~FTHG+I(FTHG-FTAG), varwidth = T))
with(dt%>%filter(abs(FTHG-FTAG)<=3), boxplot((1/BWH+1/BWD+1/BWA)~FTHG+I(FTHG-FTAG), varwidth = T))

with(dt, plot(x,y,col=ifelse((0==FTHG)&(0==FTAG), "red", "black"), pch=ifelse((hg0==0)&(ag0==0), 0, 1)))
with(dt, points(x,y,col=ifelse((1==FTHG)&(1==FTAG), "green", "transparent"), pch=ifelse((hg0==1)&(ag0==1), 0, 1)))
with(dt, points(x,y,col=ifelse((0==FTHG)&(0==FTAG), "red", "transparent"), pch=ifelse((hg0==0)&(ag0==0), 0, 1)))


with(dt, plot(FTR~x0))
with(dt, plot(FTR~c0))
with(dt, plot(factor(FTHG)~I(sqrt((1-x)^2+(1-y)^2))))

######################################################################################
f<-function(s)
#for (s in levels(alldata$season)) 
{
  dt<-alldata%>%filter(season==s)
  dt$x<-(1/dt$BWA)/(1/dt$BWH+1/dt$BWD+1/dt$BWA)
  dt$y<-1/dt$BWH/(1/dt$BWH+1/dt$BWD+1/dt$BWA)
  dt$x0<-dt$x-dt$y
  cutpoints<-apply(cuts, 1, function(y) optimize(function(x) mean(ifelse(dt$x0<x, 
                                                                         point.system(y[["hg0"]], y[["ag0"]], dt$FTHG, dt$FTAG),
                                                                         point.system(y[["hg"]], y[["ag"]], dt$FTHG, dt$FTAG))), range(dt$x0), maximum = T)$maximum)
  cbind(as.integer(s), cutpoints, 
    apply(cbind(cuts, cutpoints), 1, function(y) mean(ifelse(dt$x0<y[["cutpoints"]], 
                                                  point.system(y[["hg0"]], y[["ag0"]], dt$FTHG, dt$FTAG),
                                                  point.system(y[["hg"]], y[["ag"]], dt$FTHG, dt$FTAG)))))
}
boxplot(t(sapply(levels(alldata$season), f))[,5:12])


hist(1/dt$BWH)
hist(1/dt$BWD)
hist(1/dt$BWA)
hist(1/dt$BWH+1/dt$BWD+1/dt$BWA)
summary(1/dt$BWH+1/dt$BWD+1/dt$BWA)

f1<-function(x) mean(ifelse(dt$x0>x, 
                    point.system(c$hg0, c$ag0, dt$FTHG, dt$FTAG),
                     point.system(c$hg, c$ag, dt$FTHG, dt$FTAG)))
d1<-optimize(f1, range(dt$x0), maximum = T)$maximum
print(d1)
print(f1(d1))


f1(0)
f1(1)
f1(-1)
f1(3)

f1<-function(x) mean(ifelse(dt$x0>x, dt$sA, dt$sD))
f2<-function(x) mean(ifelse(dt$x0<x, dt$sH, dt$sD))
f5<-function(x) mean(ifelse(dt$x0<x, dt$sH, dt$sA))
d2<-optimize(f1, range(dt$x0), maximum = T)$maximum
d1<-optimize(f2, range(dt$x0), maximum = T)$maximum
d3<-optimize(f5, range(dt$x0), maximum = T)$maximum
fA1<-function(x) mean(ifelse(dt$x0>x, dt$sA2, dt$sA))
fH1<-function(x) mean(ifelse(dt$x0<x, dt$sH2, dt$sH))



#dt<-dt%>%group_by(z)%>%mutate(x=mean(log(BWH)), y=mean(log(BWA)))%>%ungroup()
dt<-dt%>%mutate(sA=ifelse(FTR=="A", ifelse(FTHG==1&FTAG==2, 5, 2), 0),
                sH=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==1, 5, 2), 0),
                sA2=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==2, 5, 2), 0),
                sH2=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==0, 5, 2), 0),
                sD=ifelse(FTR=="D", ifelse(FTHG==1&FTAG==1, 5, 2), 0),
                pA=ifelse(FTR=="A", ifelse(FTHG==1&FTAG==2, 3, ifelse(FTHG-FTAG==-1, 2, 1)), 0),
                pH=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==1, 3, ifelse(FTHG-FTAG==1, 2, 1)), 0),
                pA2=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==2, 3, ifelse(FTHG-FTAG==-2, 2, 1)), 0),
                pH2=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==0, 3, ifelse(FTHG-FTAG==2, 2, 1)), 0),
                pD=ifelse(FTR=="D", ifelse(FTHG==1&FTAG==1, 3, 2), 0)
) 
dt %>% select(FTR, FTHG, FTAG, sH, sD, sA, pH, pD, pA, sH2, sA2, pH2, pA2)




with(dt, plot(FTR~r0))
with(dt, plot(as.factor(FTHG==0&FTAG==0)~r0))
with(dt, plot(as.factor(FTHG==1&FTAG==0)~r0))
with(dt, plot(as.factor(FTHG==0&FTAG==1)~r0))
with(dt, plot(as.factor(FTHG==2&FTAG==1)~r0))
z <- with(alldata, factor(kmeans(cbind(log(BWA), log(BWH)),centers=40)$cluster))
dt$z<-z
pred<-with(alldata, knn(cbind(log(BWA), log(BWH)), cbind(log(BWA), log(BWH)), FTR, k = 10, l = 0, prob = FALSE, use.all = TRUE))
pred<-with(dt, knn(data.frame(x0), data.frame(x0), FTR, k = 10, l = 0, prob = FALSE, use.all = TRUE))
dt$cl<-pred
dt<-dt%>%group_by(z)%>%mutate(x=mean(log(BWH)), y=mean(log(BWA)))%>%ungroup()
dt<-dt%>%mutate(sA=ifelse(FTR=="A", ifelse(FTHG==1&FTAG==2, 5, 2), 0),
                sH=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==1, 5, 2), 0),
                sA2=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==2, 5, 2), 0),
                sH2=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==0, 5, 2), 0),
                sA3=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==3, 5, 2), 0),
                sH3=ifelse(FTR=="H", ifelse(FTHG==3&FTAG==0, 5, 2), 0),
                sA4=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==4, 5, 2), 0),
                sH4=ifelse(FTR=="H", ifelse(FTHG==4&FTAG==0, 5, 2), 0),
                sD=ifelse(FTR=="D", ifelse(FTHG==1&FTAG==1, 5, 2), 0),
                pA=ifelse(FTR=="A", ifelse(FTHG==1&FTAG==2, 3, ifelse(FTHG-FTAG==-1, 2, 1)), 0),
                pH=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==1, 3, ifelse(FTHG-FTAG==1, 2, 1)), 0),
                pA2=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==2, 3, ifelse(FTHG-FTAG==-2, 2, 1)), 0),
                pH2=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==0, 3, ifelse(FTHG-FTAG==2, 2, 1)), 0),
                pA3=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==3, 3, ifelse(FTHG-FTAG==-3, 2, 1)), 0),
                pH3=ifelse(FTR=="H", ifelse(FTHG==3&FTAG==0, 3, ifelse(FTHG-FTAG==3, 2, 1)), 0),
                pA4=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==4, 3, ifelse(FTHG-FTAG==-4, 2, 1)), 0),
                pH4=ifelse(FTR=="H", ifelse(FTHG==4&FTAG==0, 3, ifelse(FTHG-FTAG==4, 2, 1)), 0),
                pD=ifelse(FTR=="D", ifelse(FTHG==1&FTAG==1, 3, 2), 0)
) 
dt %>% select(FTR, FTHG, FTAG, sH, sD, sA, pH, pD, pA, sH2, sA2, pH2, pA2)

g<-dt%>%group_by(z)%>%summarise(x=mean(x), y=mean(y), 
                             msH=mean(sH), msD=mean(sD), msA=mean(sA), 
                             mpH=mean(pH), mpD=mean(pD), mpA=mean(pA), 
                             n=n(), 
                             sMax=which.max(c(msH, msD, msA)), 
                             pMax=which.max(c(mpH, mpD, mpA))) 

g%>%arrange(x, y)%>% print.data.frame()
with(dt, plot(log(BWH),log(BWA),col=pred))
with(dt%>%arrange(BWH, BWA), plot(log(BWH)-log(BWA),col=pred))


#with(alldata, kmeans(cbind(log(BWA), log(BWH)), FTR, centers = 40))
library(class)


pred<-with(alldata, knn(cbind(log(BWA), log(BWH)), cbind(log(BWA), log(BWH)), FTR, k = 1, l = 0, prob = FALSE, use.all = TRUE))
table(pred, alldata$FTR)


with(g, plot(x,y,col=sMax, cex=sqrt(n)/5))
with(g, plot(x,y,col=pMax, cex=sqrt(n)/5))


with(dt,   sm.density.compare(log(BWH)^(+0.2+log(BWA)), FTR, model = "equal"))
dt$x0
range(dt$x0)
f1(0.3)%>% select(FTR, FTHG, FTAG, sH, sD, sA, pH, pD, pA, x0, r)
f2(0.3)%>% select(FTR, FTHG, FTAG, sH, sD, sA, pH, pD, pA, x0, r)

with(dt, plot(FTR~x0))

f1<-function(x) mean(ifelse(dt$x0>x, dt$sA, dt$sD))
f2<-function(x) mean(ifelse(dt$x0<x, dt$sH, dt$sD))
f5<-function(x) mean(ifelse(dt$x0<x, dt$sH, dt$sA))
d2<-optimize(f1, range(dt$x0), maximum = T)$maximum
d1<-optimize(f2, range(dt$x0), maximum = T)$maximum
d3<-optimize(f5, range(dt$x0), maximum = T)$maximum
fA1<-function(x) mean(ifelse(dt$x0>x, dt$sA2, dt$sA))
fH1<-function(x) mean(ifelse(dt$x0<x, dt$sH2, dt$sH))

f1(optimize(f1, range(dt$x0), maximum = T)$maximum)
f2(optimize(f2, range(dt$x0), maximum = T)$maximum)
f5(optimize(f5, range(dt$x0), maximum = T)$maximum)
dA1<-optimize(fA1, range(dt$x0), maximum = T)$maximum
dH1<-optimize(fH1, range(dt$x0), maximum = T)$maximum
fA1(optimize(fA1, range(dt$x0), maximum = T)$maximum)
fH1(optimize(fH1, range(dt$x0), maximum = T)$maximum)

print(paste("home2", dH1, "home", d1, "draw", d2, "away", dA1, "away2"))
print(paste("home", d3, "away"))
dt$pred0<-factor(ifelse(dt$x0<d1, "H", ifelse(dt$x0>d2, "A", "D")))
dt$pred2<-factor(ifelse(dt$x0<d1, ifelse(dt$x0<dH1, "H2", "H"), ifelse(dt$x0>d2, ifelse(dt$x0>dA1, "A2", "A"), "D")))
dt%>%select(FTR, pred0)%>%table()
with(dt, plot(log(BWH),log(BWA),col=pred0, pch=ifelse(as.character(pred0)==as.character(FTR), 1, 4), main=mean(ifelse(dt$x0<d1, dt$sH, ifelse(dt$x0>d2, dt$sA, dt$sD)))))
with(dt, plot(log(BWH),log(BWA),col=pred2, pch=ifelse(as.character(pred0)==as.character(FTR), 1, 4), 
              main=mean(ifelse(dt$x0<d1, 
                               ifelse(dt$x0<dH1, dt$sH2, dt$sH), 
                               ifelse(dt$x0>d2, 
                                      ifelse(dt$x0>dA1, dt$sA2, dt$sA), dt$sD)))))


f3<-function(x) mean(ifelse(dt$x0>x, dt$pA, dt$pD))
f4<-function(x) mean(ifelse(dt$x0<x, dt$pH, dt$pD))
d2<-optimize(f3, range(dt$x0), maximum = T)$maximum
d1<-optimize(f4, range(dt$x0), maximum = T)$maximum

f6<-function(x) mean(ifelse(dt$x0<x, dt$pH, dt$pA))
d3<-optimize(f6, range(dt$x0), maximum = T)$maximum

f3(optimize(f3, range(dt$x0), maximum = T)$maximum)
f4(optimize(f4, range(dt$x0), maximum = T)$maximum)
f6(optimize(f6, range(dt$x0), maximum = T)$maximum)

fA1<-function(x) mean(ifelse(dt$x0>x, dt$pA2, dt$pA))
fH1<-function(x) mean(ifelse(dt$x0<x, dt$pH2, dt$pH))
dA1<-optimize(fA1, range(dt$x0), maximum = T)$maximum
dH1<-optimize(fH1, range(dt$x0), maximum = T)$maximum
fA1(optimize(fA1, range(dt$x0), maximum = T)$maximum)
fH1(optimize(fH1, range(dt$x0), maximum = T)$maximum)

print(paste("home2", dH1, "home", d1, "draw", d2, "away", dA1, "away2"))
print(paste("home", d3, "away"))

dt$pred2<-factor(ifelse(dt$x0<d1, ifelse(dt$x0<dH1, "H2", "H"), ifelse(dt$x0>d2, ifelse(dt$x0>dA1, "A2", "A"), "D")))
dt%>%select(FTR, pred0)%>%table()
with(dt, plot(log(BWH),log(BWA),col=pred2, pch=ifelse(as.character(pred0)==as.character(FTR), 1, 4), 
              main=mean(ifelse(dt$x0<d1, 
                               ifelse(dt$x0<dH1, dt$pH2, dt$pH), 
                               ifelse(dt$x0>d2, 
                                      ifelse(dt$x0>dA1, dt$pA2, dt$pA), dt$pD)))))


with(dt, plot(log(BWH),log(BWA), col=ifelse(FTHG==1&FTAG==0, "red", "black"), cex=ifelse(FTHG==1&FTAG==0, 5, 1)))
with(dt, plot(log(BWH),log(BWA), col=ifelse(FTHG==0&FTAG==1, "red", "black"), cex=ifelse(FTHG==0&FTAG==1, 5, 1)))
with(dt, plot(log(BWH),log(BWA), col=ifelse(FTHG==0&FTAG==0, "red", "blue"), cex=ifelse(FTHG==0&FTAG==0, 5, 1)))


plot((dt %>% arrange(BWA) %>% mutate(diffp=pH-pD, c=cumsum(diffp) ))$c )
plot((dt %>% arrange(BWA) %>% mutate(diffp=pH2-pH, c=cumsum(diffp) ))$c )
plot((dt %>% arrange(BWA) %>% mutate(diffp=pA-pA2, c=cumsum(diffp) ))$c )
plot((dt %>% arrange(BWA) %>% mutate(diffp=pD-pA, c=cumsum(diffp) ))$c )

plot((dt %>% arrange(BWA) %>% mutate(diffp=pH-pD, c=cumsum(diffp) ))$c )
s<-sample(nrow(dt), 200); plot((dt[s,] %>% arrange(BWA) %>% mutate(diffp=pH-pD, c=cumsum(diffp) ))$c)

plot((dt %>% arrange(BWA) %>% mutate(diffp=pH2-pH, c=cumsum(diffp) ))$c )
n<-256
s<-sample(nrow(dt), n); d <- (dt[s,] %>% arrange(BWA) %>% mutate(diffp=pD-pA, c=cumsum(diffp), x=1/BWA-1/BWH ))[,c("x", "c")]; plot(d)
p<-150; abline(v=p, col="red"); abline(v=which.min(d), col="blue", lty=2)
ws<-50;low<-max(1,p-ws); high<-min(n,p+ws-1); segments(low,d[low],high,d[high])
ws<-5;low<-max(1,p-ws); high<-min(n,p+ws-1); segments(low,d[low],high,d[high])
ws<-10;low<-max(1,p-ws); high<-min(n,p+ws-1); segments(low,d[low],high,d[high])
ws<-25;low<-max(1,p-ws); high<-min(n,p+ws-1); segments(low,d[low],high,d[high])
ws<-15;low<-max(1,p-ws); high<-min(n,p+ws-1); segments(low,d[low],high,d[high])

s<-sample(nrow(dt), n); d <- (dt[s,] %>% mutate(x=1/BWH-1/BWA) %>% mutate(x=(x)) %>% arrange(x) %>% mutate(diffp=pH-pA) %>% mutate( c=cumsum(diffp) ))[,c("x", "c")]; plot(d)




dt<-dt%>%mutate(sA=ifelse(FTR=="A", ifelse(FTHG==1&FTAG==2, 5, 2), 0),
                sH=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==1, 5, 2), 0),
                sA2=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==2, 5, 2), 0),
                sH2=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==0, 5, 2), 0),
                sA3=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==3, 5, 2), 0),
                sH3=ifelse(FTR=="H", ifelse(FTHG==3&FTAG==0, 5, 2), 0),
                sA4=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==4, 5, 2), 0),
                sH4=ifelse(FTR=="H", ifelse(FTHG==4&FTAG==0, 5, 2), 0),
                sD=ifelse(FTR=="D", ifelse(FTHG==1&FTAG==1, 5, 2), 0),
                pA=ifelse(FTR=="A", ifelse(FTHG==1&FTAG==2, 3, ifelse(FTHG-FTAG==-1, 2, 1)), 0),
                pH=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==1, 3, ifelse(FTHG-FTAG==1, 2, 1)), 0),
                pA2=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==2, 3, ifelse(FTHG-FTAG==-2, 2, 1)), 0),
                pH2=ifelse(FTR=="H", ifelse(FTHG==2&FTAG==0, 3, ifelse(FTHG-FTAG==2, 2, 1)), 0),
                pA3=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==3, 3, ifelse(FTHG-FTAG==-3, 2, 1)), 0),
                pH3=ifelse(FTR=="H", ifelse(FTHG==3&FTAG==0, 3, ifelse(FTHG-FTAG==3, 2, 1)), 0),
                pA4=ifelse(FTR=="A", ifelse(FTHG==0&FTAG==4, 3, ifelse(FTHG-FTAG==-4, 2, 1)), 0),
                pH4=ifelse(FTR=="H", ifelse(FTHG==4&FTAG==0, 3, ifelse(FTHG-FTAG==4, 2, 1)), 0),
                pD=ifelse(FTR=="D", ifelse(FTHG==1&FTAG==1, 3, 2), 0)
) 

mydata<-dt
mydata <- mydata %>% mutate(x=1/BWH-1/BWA) %>% arrange(x) 
mydata <- mydata %>% mutate(x=1/B365H-1/B365A) %>% arrange(x) 
#mydata <- mydata %>% mutate(x=sample(x, replace = F))

#plot(factor(mydata$pH2) ~ mydata$x )
f<-function(n=n){
  (mydata[sample(nrow(mydata), n, replace = FALSE),]%>% arrange(x) %>% mutate(diffp=sD-sA) %>% mutate( c=cumsum(diffp) ))[,c("x", "c")]
  #(mydata[sample(nrow(mydata), n),]%>% mutate(diffp=pA-pA2) %>% mutate( c=cumsum(diffp) ))[,c("x", "c")]
}
p<-data.frame()
pmin<-c()
s<-f(nrow(mydata))
pmin<-c(pmin, s$x[which.min(s$c)])
p<-rbind(p,s)

p<-data.frame()
pmin<-c()
for (i in 1:100) {
  s<-f(256)
  pmin<-c(pmin, s$x[which.min(s$c)])
  p<-rbind(p,s)
  
}
qplot(x, c, data=p, geom=c("point", "smooth", "density2d" ))

model.gam<-mgcv::gam(c~s(x, bs = "cs"), data=p, bs="cs")
#predict.gam(model.gam, newdata=p$x)
min.point <- p$x[which.min(fitted(model.gam))]
min.point.emp <- mean(pmin)
print(c(min.point, min.point.emp))
ggplot(p, aes(x,c))+geom_density2d()+geom_smooth(level=0.95, method="gam", formula = y~s(x, bs = "cs"))+geom_vline(xintercept = min.point, color="red")+geom_vline(xintercept = min.point.emp, color="blue")

pj<-data.frame()
for (j in 1:10) {
    
  p<-data.frame()
  pmin<-c()
  for (i in 1:100) {
    s<-f(256)
    #s<-f(nrow(mydata))
    pmin<-c(pmin, s$x[which.min(s$c)])
    p<-rbind(p,s)
    
  }
  model.gam<-mgcv::gam(c~s(x, bs = "cs"), data=p, bs="cs")
  min.point <- p$x[which.min(fitted(model.gam))]
  min.point.emp <- mean(pmin)
  pj <- rbind(pj, data.frame(cs=min.point, emp=min.point.emp))
}
summary(pj)


