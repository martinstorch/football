setwd("D:/gitrepository/Football/football/R")

library(lubridate)
library(glmnet)
library(glmnetUtils)
library(dplyr)
library(ggplot2)

data <- read.csv("../TF/data/full_data.csv")

data$Date<-dmy(data$Date)

data$Time<-as.character(data$Time)
data$Time[data$Time==""]<-NA
data$Time<-as.numeric(substr(data$Time, 1,2))+as.numeric(substr(data$Time, 4,5))/60

data$BW1<-1/data$BWH
data$BW0<-1/data$BWD
data$BW2<-1/data$BWA
data$BW102<-data$BW1+data$BW0+data$BW2
data$BW1<-data$BW1/data$BW102
data$BW0<-data$BW0/data$BW102
data$BW2<-data$BW2/data$BW102

data <- data %>% filter(Predict=="False")
str(data)

formula = FTAG~.-FTR-HTHG-HTR-HTAG-FTAG-FTHG-BWH-BWD-BWA-BW102

formula = FTHG~.-FTR-HTHG-HTR-HTAG-FTAG-FTHG-BWH-BWD-BWA-BW102

formula = as.integer(FTR)~.-HTHG-HTR-HTAG-FTAG-FTHG-BWH-BWD-BWA-BW102

model<-glmnet(formula, data=data, family = "poisson")

model<-glmnet(formula, data=data, family = "multinomial", type.multinomial = "grouped")
cvfit = cv.glmnet(formula, data=data, family = "multinomial", type.multinomial = "grouped")

plot(model)
coef(model, s=exp(-8))
cvfit = cv.glmnet(formula, data=data, family = "poisson")

plot(cvfit)
print(log(cvfit$lambda.min))
print(log(cvfit$lambda.1se))

c<-as.data.frame(as.matrix(coef(cvfit, s="lambda.min")))
c2<-as.data.frame(as.matrix(coef(cvfit, s="lambda.1se")))
colnames(c)<-"coef"
c$name <- rownames(c)
colnames(c2)<-"coef"
c2$name <- rownames(c2)

c%>%filter(coef!=0)%>%arrange(coef)%>%left_join(c2%>%filter(coef!=0)%>%arrange(coef), by="name")


c<-as.data.frame(as.matrix(coef(cvfit, s="lambda.min")$`3`))
c2<-as.data.frame(as.matrix(coef(cvfit, s="lambda.1se")$`3`))
colnames(c)<-"coef"
c$name <- rownames(c)
colnames(c2)<-"coef"
c2$name <- rownames(c2)

c%>%filter(coef!=0)%>%arrange(coef)%>%left_join(c2%>%filter(coef!=0)%>%arrange(coef), by="name")

plot(FTHG~xHG, data=data)
abline(lm(FTHG~xHG, data=data), col="red")
print(lm(FTHG~xHG, data=data))

plot(FTAG~xAG, data=data)
abline(lm(FTAG~xAG, data=data), col="red")
print(lm(FTAG~xAG, data=data))


data <- read.csv("../TF/data/all_features.csv")
labels <- read.csv("../TF/data/all_labels.csv")

labels<-labels[data$Predict=="False",]
data<-data[data$Predict=="False",]

#labels<-labels[data$Where==1,]
#data<-data[data$Where==1,]

data$t1dayssince <-data$t1dayssince*1000
data$t2dayssince <-data$t2dayssince*1000

label <- labels$T1_GFT; family="poisson"

label <- labels$T1_GFT==labels$T2_GFT; family="binomial"
model<-glmnet(label~., data=cbind(data[,-c(2:6)], label=label), family = family, alpha=1)

plot(model)
#coef(model, s=exp(-1))
#coef(model, s=exp(-8))

seasons<-c(rep(1:10, each=17*18*2), rep(11, 4*9*2))
seasons<-c(rep(1:10, each=17*18), rep(11, 4*9))
plot(data$Date, col=seasons)

cvfit = cv.glmnet(label~., data=cbind(scale(data[,-c(2:6)]), label=label, foldid=seasons), family = family, alpha=0.9)

cvfit = cv.glmnet(label~., data=cbind(data, label=label, foldid=seasons), family = family, alpha=0.8)

plot(cvfit)
print(log(cvfit$lambda.min))
print(log(cvfit$lambda.1se))

c<-as.data.frame(as.matrix(coef(cvfit, s="lambda.min")))
c2<-as.data.frame(as.matrix(coef(cvfit, s="lambda.1se")))
colnames(c)<-"coef"
c$name <- rownames(c)
colnames(c2)<-"coef"
c2$name <- rownames(c2)

c%>%filter(coef!=0)%>%arrange(coef)%>%full_join(c2%>%filter(coef!=0)%>%arrange(coef), by="name")
min(cvfit$cvm)



var(labels$T1_GFT[data$Where==1])
mean(labels$T1_GFT[data$Where==1])
var(labels$T2_GFT[data$Where==1])
mean(labels$T2_GFT[data$Where==1])








ggplot(cbind(data, label=label), aes(x=t1dayssince, fill=label))+geom_histogram(binwidth=1)
ggplot(cbind(data, label=label), aes(x=t2dayssince, fill=label))+geom_histogram(binwidth=1)

plot(factor(label)~factor(round(data$t1dayssince)) )
plot(factor(label)~factor(round(data$t2dayssince)) )
plot(data$roundsleft)
plot(factor(label)~factor(data$t1dayssince<=15))
plot(factor(label)~factor(data$t2dayssince<=15))

table(label[data$t1dayssince==data$t2dayssince])

plot(factor(label)~cut(data$t1games, 10)) 
ggplot(cbind(data, label=label), aes(x=t1games, fill=factor(-label)))+geom_histogram(bins=30)
ggplot(cbind(data, label=label), aes(x=roundsleft, fill=factor(-label)))+geom_histogram(bins=34)

ggplot(cbind(data, labels), aes(x=roundsleft, fill=factor(T2_GFT==T1_GFT)))+geom_histogram(bins=34)


table(data$roundsleft)

plot(labels$T1_GFT~roundsleft, data=data) 

label
