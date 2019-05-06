setwd("D:/gitrepository/Football/football/TF")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)
library(RColorBrewer)
library(ggplot2)
library(MASS)
library(caret)


xgoals<-read.csv("xgoals.csv")
str(xgoals)
xgoals <- xgoals%>%mutate(FTR = sign(FTHG-FTAG)+3)

plot(xAG ~ xHG, data=xgoals, col=FTR)
cor(xgoals$xAG, xgoals$xHG)

plot(pA ~ pH, data=xgoals, col=FTR)
plot(pD ~ pA, data=xgoals, col=sign(FTHG-FTAG)+3)
plot(pH ~ pA, data=xgoals, col=sign(FTHG-FTAG)+3)

trans = preProcess(xgoals%>%dplyr::select(xHG:pA), c("BoxCox", "center", "scale"))
xgoalsTrans = data.frame(trans = predict(trans, xgoals))

print(trans$mean[1])
print(trans$std[1])

xgoalsTrans$trans2.xAG <- ((xgoals$xAG^0.4-1)/0.4-trans$mean[1])/trans$std[1]+0.4551
hist(xgoalsTrans$trans.xHG, breaks=30)
shapiro.test(xgoalsTrans$trans.xHG)
hist(xgoalsTrans$trans2.xAG, breaks=30)
shapiro.test(xgoalsTrans$trans2.xAG)

plot(trans2.xAG ~ trans.xHG, data=xgoalsTrans, col=trans.FTR)


ldamodel <- lda(trans.FTR ~ trans.xHG+trans2.xAG, data=xgoalsTrans, CV = T)
str(ldamodel)

table(ldamodel$class, xgoalsTrans$trans.FTR)/length(ldamodel$class)
mean(ldamodel$class== xgoalsTrans$trans.FTR)

ldamodel <- lda(trans.FTR ~ trans.xHG+trans2.xAG, data=xgoalsTrans, CV = F)
print(ldamodel)
plda <- predict(ldamodel, newdata = xgoalsTrans)
prop <- ldamodel$svd^2/sum(ldamodel$svd^2)
print(prop)

table(plda$class, xgoalsTrans$trans.FTR)/length(plda$class)
mean(plda$class== xgoalsTrans$trans.FTR)

dataset <- data.frame(FTR = factor(xgoalsTrans$trans.FTR), lda = plda$x, class=plda$class)
centr <- predict(ldamodel, newdata = data.frame(ldamodel$means))

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = FTR, shape = class), size = 2.5, alpha=0.4) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  geom_point(data=data.frame(centr$x, FTR=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTR), size=10, pch=4)

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = class, shape = FTR), size = 2.5, alpha=0.4) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  geom_point(data=data.frame(centr$x, FTR=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTR), size=10, pch=4)


ldamodel <- lda(FTR ~ xHG+xAG, data=xgoals, CV = T)

table(ldamodel$class, xgoalsTrans$trans.FTR)/length(ldamodel$class)
mean(ldamodel$class== xgoalsTrans$trans.FTR)

ldamodel <- lda(FTR ~ xHG+xAG, data=xgoals, CV = F)
print(ldamodel)
plda <- predict(ldamodel, newdata = xgoals)

dataset <- data.frame(FTR = factor(xgoals$FTR), lda = plda$x, class=plda$class)
centr <- predict(ldamodel, newdata = data.frame(ldamodel$means))

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = FTR, shape = class), size = 2.5, alpha=0.4) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  geom_point(data=data.frame(centr$x, FTR=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTR), size=10, pch=4)#+
  #geom_contour(aes(x=lda.LD1, y=lda.LD2, z=as.integer(class)))

ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = class, shape = FTR), size = 2.5, alpha=0.4) + 
  labs(x = paste("LD1 (", (prop[1]), ")", sep=""),
       y = paste("LD2 (", (prop[2]), ")", sep=""))+
  geom_point(data=data.frame(centr$x, FTR=rownames(centr$x)), aes(x=LD1, y=LD2, colour=FTR), size=10, pch=4)



plot(xHG ~ factor(FTHG), data=xgoals)
plot(xAG ~ factor(FTAG), data=xgoals)
plot(xHG-xAG ~ factor(FTHG-FTAG), data=xgoals)
abline(lm(I(xHG-xAG) ~ I(FTHG-FTAG), data=xgoals))

summary(lm(I(xHG-xAG) ~ I(FTHG-FTAG), data=xgoals))
summary(lm(I(FTHG-FTAG) ~ I(xHG-xAG), data=xgoals))
summary(lm(FTHG ~ xHG, data=xgoals))
summary(lm(FTAG ~ xAG, data=xgoals))

qqnorm(sqrt(xgoals$xHG))
qqline(sqrt(xgoals$xHG))

qqnorm(sqrt(xgoals$xAG))
qqline(sqrt(xgoals$xAG))

qqplot(xgoals$xHG, xgoals$xAG)

hist(xgoals$xHG, breaks=30)
hist(log(xgoals$xHG), breaks=30)
hist(sqrt(xgoals$xHG), breaks=30)

shapiro.test(xgoals$xHG)
shapiro.test(log(xgoals$xHG))
shapiro.test(sqrt(xgoals$xHG))


summary(xgoalsTrans$trans.xHG - ((((xgoals$xHG)^0.4-1)/0.4-trans$mean[1])/trans$std[1]))

# boxcox, center, scale 
((xgoals$xHG^0.4-1)/0.4-trans$mean[1])/trans$std[1]

hist(xgoalsTrans$trans.xHG, breaks=30)

hist(((xgoals$xAG^0.4-1)/0.4-trans$mean[1])/trans$std[1]+0.4551, breaks=30)
shapiro.test(((xgoals$xAG^0.4-1)/0.4-trans$mean[1])/trans$std[1]+0.4551)
summary(((xgoals$xAG^0.4-1)/0.4-trans$mean[1])/trans$std[1]+0.4551)
sd(((xgoals$xAG^0.4-1)/0.4-trans$mean[1])/trans$std[1])

hist(rnorm(1000), breaks=30)

hist(((xgoals$xHG)^0.4-1)/0.4-trans$mean[1], breaks=30)
hist(xgoalsTrans$trans.xHG, breaks=30)

sd(((xgoals$xHG)^0.4-1)/0.4)

str(trans)

plot(trans2.xAG ~ trans.xHG, data=xgoalsTrans, col=sign(trans.FTHG-trans.FTAG)+3)


with(xgoals, cor(xHG-xAG, FTHG-FTAG))

xgoals$HomeTeamStd<-factor(levels(alldata$HomeTeam)[apply(adist(gsub('FC |FSV |Borussia ', '', xgoals$HomeTeam), levels(alldata$HomeTeam)), 1, which.min)], levels = levels(alldata$HomeTeam))
newdata$AwayTeam<-factor(levels(alldata$HomeTeam)[apply(adist(gsub('FC |FSV |Borussia ', '', newdata$AwayTeam), levels(alldata$HomeTeam)), 1, which.min)], levels = levels(alldata$AwayTeam))

write.csv(xgoals%>%dplyr::select(HomeTeam, HomeTeamStd)%>%unique(), "xg_team_mapping.csv")




table(xgoals$HomeTeam2, xgoals$HomeTeam)
