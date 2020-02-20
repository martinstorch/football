setwd("D:/gitrepository/Football/football/R")

library(lubridate)
library(glmnet)
library(glmnetUtils)
library(dplyr)
library(ggplot2)
library(mpath)

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

table(data$Season)
data%>%ggplot(aes(x=xHG, y=Hxsg))+geom_point()+geom_abline(aes(slope=1,intercept=0), color="red")
data%>%ggplot(aes(x=xHG, y=(Hxsg-xHG)))+geom_point()+geom_abline(aes(slope=0, intercept=0), color="red")
data%>%ggplot(aes(x=xAG, y=Axsg))+geom_point()+geom_abline(aes(slope=1,intercept=0), color="red")
data%>%ggplot(aes(x=xAG, y=(Axsg-xAG)))+geom_point()+geom_abline(aes(slope=0, intercept=0), color="red")
summary(data$xHG-data$Hxsg)
summary(data$xAG-data$Axsg)

data<-data%>%mutate(BWpred=case_when(((BW1>BW2) & (BW1>BW0)) ~ 1, ((BW1<BW2) & (BW2>BW0)) ~ -1, is.na(BW1)~as.numeric(NA), TRUE~0))
data<-data%>%mutate(SPpred=case_when(((ppH>ppA) & (ppH>ppD)) ~ 1, ((ppH<ppA) & (ppA>ppD)) ~ -1, is.na(ppH)~as.numeric(NA), TRUE~0))
data<-data%>%mutate(SPIpred=sign(HGFTe-AGFTe))
data<-data%>%mutate(label=sign(FTHG-FTAG))
data<-data%>%mutate(Imp=sign(Himp-Aimp))
table(data$BWpred)
table(data$SPpred)
table(data$label)

table(data$label, data$BWpred)
table(data$label, data$SPpred)
table(data$label, data$Imp)

data%>%summarise(BWpred = mean(BWpred==label), SPpred = mean(SPpred==label, na.rm=T), SPIpred = mean(SPIpred==label, na.rm=T))
data%>%group_by(Season)%>%summarise(BWpred = mean(BWpred==label), SPpred = mean(SPpred==label, na.rm=T), SPIpred = mean(SPIpred==label, na.rm=T))

data%>%filter(BWpred!=SPpred)%>%summarise(BWpred = mean(BWpred==label), SPpred = mean(SPpred==label, na.rm=T), SPIpred = mean(SPIpred==label, na.rm=T))
data%>%group_by(Season)%>%filter(BWpred!=SPIpred)%>%summarise(BWpred = mean(BWpred==label), SPpred = mean(SPpred==label, na.rm=T), SPIpred = mean(SPIpred==label, na.rm=T))
data%>%group_by(Season)%>%filter(BWpred==SPpred)%>%summarise(BWpred = mean(BWpred==label), SPpred = mean(SPpred==label, na.rm=T), SPIpred = mean(SPIpred==label, na.rm=T))




data$ppA

formula = FTAG~.-FTR-HTHG-HTR-HTAG-FTAG-FTHG-BWH-BWD-BWA-BW102

formula = FTHG~.-FTR-HTHG-HTR-HTAG-FTAG-FTHG-BWH-BWD-BWA-BW102

formula = as.integer(FTR)~.-HTHG-HTR-HTAG-FTAG-FTHG-BWH-BWD-BWA-BW102

model<-glmnet(formula, data=data, family = "poisson")

model<-glmnet(formula, data=data, family = "multinomial", type.multinomial = "grouped")
cvfit = cv.glmnet(formula, data=data, family = "multinomial", type.multinomial = "grouped")

plot(model, label=T)
coef(model, s=exp(-8))
cvfit = cv.glmnet(formula, data=data, family = "poisson")

plot(cvfit)
print(log(cvfit$lambda.min))
print(log(cvfit$lambda.1se))


label <- labels$T1_GFT; family="poisson"
label <- labels$T1_GFT+labels$T2_GFT; family="poisson"

label <- labels$T1_GFT==labels$T2_GFT; family="binomial"
label <- factor(sign(labels$T1_GFT-labels$T2_GFT)); family="multinomial"
label <- labels$T1_GFT-labels$T2_GFT; family="gaussian"

cvfit <- cv.glmnet(label~., data=cbind(lfda.result, label=label), foldid=foldid, family = family, alpha=1.0, parallel=TRUE, type.measure = "class", keep=T)
plot(cvfit)

coef(cvfit, c(cvfit$lambda.1se, cvfit$lambda.min))

print(c(log(cvfit$lambda.1se), cvfit$cvm[sum(cvfit$lambda>=cvfit$lambda.1se)]))
print(c(log(cvfit$lambda.min), min(cvfit$cvm)))

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
teamdata <- read.csv("../TF/data/teams_onehot.csv")

labels<-labels[data$Predict=="False",]
teamdata<-teamdata[data$Predict=="False",]
data<-data[data$Predict=="False",]

#labels<-labels[data$Where==1,]
#data<-data[data$Where==1,]

data$t1dayssince <-data$t1dayssince*1000
data$t2dayssince <-data$t2dayssince*1000

label <- labels$T2_GFT; family="poisson"
label <- labels$T1_GFT+labels$T2_GFT; family="poisson"

label <- labels$T1_GFT==labels$T2_GFT; family="binomial"
label <- factor(sign(labels$T1_GFT-labels$T2_GFT)); family="multinomial"
label <- labels$T1_GFT-labels$T2_GFT; family="gaussian"

model<-glmnet(label~., data=cbind(data[,-c(2:6)], label=label), family = family, alpha=1)

#model<-glmregNB(label~., data=cbind(data[,-c(2:6)], label=label), trace=T, nlambda=5)

plot(model, label=T)
#coef(model)
#coef(model, s=exp(-1))
#coef(model, s=exp(-8))

seasons<-c(rep(1:10, each=17*18*2), rep(11, 4*9*2))

#seasons<-c(rep(1:10, each=17*18), rep(11, 4*9))
plot(data$Date, col=seasons)

cvfit = cv.glmnet(label~., data=cbind(scale(data[,-c(2:6)]), label=label, foldid=seasons), family = family, alpha=0.9) # , type.measure = "class"

coef(cvfit, c(cvfit$lambda.1se, cvfit$lambda.min))

print(c(log(cvfit$lambda.1se), cvfit$cvm[sum(cvfit$lambda>=cvfit$lambda.1se)]))
print(c(log(cvfit$lambda.min), min(cvfit$cvm)))

#cvfit<-cv.glmregNB(label~., data=cbind(data[,-c(2:6)], label=label, foldid=seasons), parallel=T, n.cores=3)

#cvfit = cv.glmnet(label~., data=cbind(data, label=label, foldid=seasons), family = family, alpha=0.8)

plot(cvfit)

r<-as.matrix(coef(cvfit, c(cvfit$lambda.1se, cvfit$lambda.min)))
c1_Gdiff<-rownames(r)[r[,1]!=0]
c2_Gdiff<-rownames(r)[r[,2]!=0]

c1cols<-unique(c(c1T1_GFT, c1T2_GFT, c1T12_GFT, c1_Draw, c1_WDLW, c1_WDLD, c1_WDLL, c1_Gdiff))
c2cols<-unique(c(c2T1_GFT, c2T2_GFT, c1T12_GFT, c2_Draw, c2_WDLW, c2_WDLD, c2_WDLL, c2_Gdiff))

write.csv(c1cols[-1], "feature_candidates_short.csv", row.names = F)
write.csv(c2cols[-1], "feature_candidates_long.csv", row.names = F)



var(labels$T1_GFT[data$Where==1])
mean(labels$T1_GFT[data$Where==1])
var(labels$T2_GFT[data$Where==1])
mean(labels$T2_GFT[data$Where==1])



library("party")
model.cf <- cforest(label ~ ., data = cbind(data[,-c(2:6)], label=label), control = cforest_unbiased(mtry = 4, ntree = 1000))
#varimp(model.cf, conditional = TRUE, threshold = 1.0 - 1e-25)
vi <- varimp(model.cf, conditional = FALSE, mincriterion = 0.05)
print(vi[order(vi)])
dotchart(tail(vi[order(vi)], 50))

vidf<-data.frame(name=names(vi), vi)%>%arrange(vi)
write.csv(vidf, "../TF/data/Draw_vi_1000_4.csv", row.names = F)



cforestImpPlot <- function(x) {
  cforest_importance <<- v <- varimp(x)
  dotchart(v[order(v)])
}
cforestImpPlot(model.cf)


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


data.pca <- prcomp(cbind(data[,-c(2:6)], teamdata), center = TRUE,scale. = TRUE)
summary(data.pca)
plot(data.pca$sdev)
plot(summary(data.pca)$importance[2,])
plot(summary(data.pca)$importance[3,])

data.sc <- scale((cbind(data[,-c(2:6)], teamdata)), center= data.pca$center)
data.pred <- data.sc %*% data.pca$rotation
data.pred <- data.pred[,1:170]



library(lfda)
library(caret)
library(RColorBrewer)
library("colorspace")

metric = c("orthonormalized", "plain", "weighted")
# trans = preProcess(data, c("BoxCox", "center", "scale"))
# transdata <- data.frame(trans = predict(trans, data))
# 
# 
# #testquotes <- data.frame(trans = predict(trans, testquotes))

label <- labels$T2_GFT#+labels$T2_GFT;

mylabel<-ifelse(label<(-3), -3, ifelse(label>5, 5, label))

model <- lfda(data.pred, mylabel, r=10,  metric = metric, knn = 5)

reduced_data<-predict(model, data.pred)
colors <- rainbow_hcl(length(unique(mylabel)))
colors <- diverge_hsv(length(unique(mylabel)))
colors <- heat_hcl(length(unique(mylabel)))
cl <- colors[mylabel+1]
plot(reduced_data[,c(2,1)], col=cl)
plot(reduced_data[,c(2,3)], col=cl)
plot(reduced_data[,c(4,3)], col=cl)
plot(reduced_data[,c(4,5)], col=cl)
plot(reduced_data[,c(6,5)], col=cl)
plot(reduced_data[,c(6,3)], col=cl)
plot(reduced_data[,c(5,3)], col=cl)

plot(reduced_data[,c(9,10)], col=cl)
plot(reduced_data[,c(9,8)], col=cl)
plot(reduced_data[,c(7,8)], col=cl)
plot(reduced_data[,c(7,6)], col=cl)
plot(reduced_data[,c(3,10)], col=cl)
plot(reduced_data[,c(4,9)], col=cl)






length(label)
dim(reduced_data)


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




