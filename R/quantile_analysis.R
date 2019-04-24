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

quote_names<-c('BWH', 'BWD', 'BWA', 'B365H', 'B365D', 'B365A')
quote_names<-c('BWH', 'BWD', 'BWA')
normalized_quote_names<-paste0("p", quote_names)

alldata[,normalized_quote_names]<-1/alldata[,quote_names]
alldata[,normalized_quote_names[1:3]]<-alldata[,normalized_quote_names[1:3]]/rowSums(alldata[,normalized_quote_names[1:3]])
#alldata[,normalized_quote_names[4:6]]<-alldata[,normalized_quote_names[4:6]]/rowSums(alldata[,normalized_quote_names[4:6]])


#########################################################################################################################


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
feature_columns<-normalized_quote_names[1:3]
quotes<-traindata[, feature_columns]
testquotes<-testdata[, feature_columns]

metric = c("orthonormalized", "plain", "weighted")
trans = preProcess(quotes, c("BoxCox", "center", "scale"))
quotes <- data.frame(trans = predict(trans, quotes))
ggplot(melt(quotes))+facet_wrap(variable~.)+geom_histogram(aes(x=value), bins=40)

testquotes <- data.frame(trans = predict(trans, testquotes))
ggplot(melt(testquotes))+facet_wrap(variable~.)+geom_histogram(aes(x=value), bins=40)

model <- lfda(quotes, traindata$FTR, r=1,  metric = metric, knn = 20)

rownames(model$T)<-feature_columns
print(model$T)
print(model)

# plot(model$Z, col=as.integer(thedata$FTR)+1)
# plot(model$Z[,2:3], col=as.integer(thedata$FTR)+1)
# plot(model$Z[,c(1,3)], col=as.integer(thedata$FTR)+1)
# ggplot(data.frame(model$Z, FTR=thedata$FTR), aes(x=X1, fill=FTR))+geom_density(alpha=0.4) #+geom_histogram()
traindata<-data.frame(traindata, X1=model$Z)
testdata<-data.frame(testdata, X1=predict(model, testquotes))
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
newdatafile<-"quotes_bwin.csv"
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





























