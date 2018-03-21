setwd("~/LearningR/Bundesliga/")
data<-read.csv("BL2015.csv")
data2<-read.csv("BL2016.csv")
data<-rbind(data, data2)
teams <- unique(data$HomeTeam)
str(teams)
results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR')]
table(results$FTR)
scores<-table(results$FTHG, results$FTAG)
scoreprob<-table(results$FTHG, results$FTAG) / nrow(results)
cov(results$FTHG, results$FTAG )
cor(results$FTHG, results$FTAG )
cor(scores)
hgprob<-table(results$FTHG)/nrow(results)
agprob<-table(results$FTAG)/nrow(results)
hgprob%o%agprob
scoreprob / hgprob%o%agprob
ave(results$FTHG)
ave(results$FTAG)
summary(results)
library(reshape)
results$spieltag <- floor((9:(nrow(results)+8))/9)

teamresults <- data.frame(team=results$HomeTeam, otherTeam=results$AwayTeam, 
                     goals=results$FTHG, otherGoals=results$FTAG, where="Home", spieltag=results$spieltag)
teamresults <- rbind(data.frame(team=results$AwayTeam, otherTeam=results$HomeTeam, 
                          goals=results$FTAG, otherGoals=results$FTHG, where="Away", spieltag=results$spieltag),
                     teamresults)


staticlambda<-glm(formula = goals ~ 1, data=teamresults, family = poisson)
predict(staticlambda, type="response")
exp(staticlambda$coefficients)

fitted(staticlambda)[1]

fittedstatic<-round(dpois(0:6, (fitted(staticlambda)[1]))*nrow(teamresults))
names(fittedstatic)<-0:6

rbind(fittedstatic, actual=table(teamresults$goals))


glm(formula = goals ~ team, data=teamresults, family = poisson)
glm(formula = goals ~ ., data=teamresults, family = poisson)
m.team<-glm(formula = goals ~ team+otherTeam+where, data=teamresults, family = poisson)
m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = poisson)
summary(m.team)
m.otherteam<-glm(formula = otherGoals ~ (team+otherTeam)*where, data=teamresults, family = poisson)
summary(m.otherteam)

cbind(teamresults, fitted(m.team), fitted(m.otherteam))

predict(m.team, type = "response",  newdata=data.frame(team="Dortmund", otherTeam="Ingolstadt", where="Home"))
predict(m.otherteam, type = "response",  newdata=data.frame(team="Dortmund", otherTeam="Ingolstadt", where="Home"))

predict(m.team, type = "response",  newdata=data.frame(team="Ingolstadt", otherTeam="Dortmund", where="Home"))
predict(m.otherteam, type = "response",  newdata=data.frame(team="Ingolstadt", otherTeam="Dortmund", where="Home"))

m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = poisson)
plot(teamresults$goals, fitted(m.team))
summary(m.team)
m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = quasipoisson)
plot(teamresults$goals, fitted(m.team))
summary(m.team)
m.team<-glm(formula = goals ~ team*where+otherTeam, data=teamresults, family = negative.binomial(theta = 1))
plot(teamresults$goals, fitted(m.team))
summary(m.team)
library(gnm)
m.team<-glm(formula = goals ~ (team:otherTeam)*where, data=teamresults, family = poisson)
plot(teamresults$goals, fitted(m.team))

library(pscl)
m.team<-hurdle(formula = goals ~ team*where+otherTeam, data=teamresults, dist = "poisson")
m.team<-hurdle(formula = goals ~ (team+otherTeam)*where, data=teamresults, dist = "geometric")
m.team<-hurdle(formula = goals ~ (team+otherTeam)*where, data=teamresults, dist = "negbin")
summary(m.team)
library(sandwich)
coeftest(m.team, vcov = sandwich)


fittedgoals<-round(dpois(0:6, (fitted(m.team)[0]))*nrow(teamresults))
names(fittedgoals)<-0:6
rbind(fittedgoals, actual=table(teamresults$goals))
rbind(fittedstatic, actual=table(teamresults$goals))

plot(teamresults$goals, fitted(m.team))
boxplot(fitted(m.team) ~ teamresults$goals)
summary(dpois(teamresults$goals, fitted(m.team)))
summary(dpois(teamresults$goals, fitted(m.team)-0.14))
plot(dpois(teamresults$goals, fitted(m.team)))
summary(dpois(teamresults$goals+1, fitted(m.team)))
summary(dpois(teamresults$goals+2, fitted(m.team)))
summary(dpois(teamresults$goals+3, fitted(m.team)))
summary(dpois(teamresults$goals-1, fitted(m.team)))
summary(dpois(0, fitted(m.team)))
summary(dpois(1, fitted(m.team)))
summary(dpois(2, fitted(m.team)))
summary(dpois(teamresults$goals, fitted(staticlambda)))
summary(m.team)
summary(fitted(m.team))
allpred<-sapply(0:6, function(x) dpois(x, fitted(m.team)))
bestpred<-apply(allpred, 1, which.max)-1
table(data.frame(pred=bestpred, act=teamresults$goals)) #, diff=bestpred - teamresults$goals)   )
summary(data.frame(pred=bestpred, act=teamresults$goals))

predict(m.team, type = "response",  newdata=data.frame(team="Augsburg", otherTeam="Leverkusen", where="Home"))

hg<-predict(m.team, type = "response",  newdata=data.frame(team="Augsburg", otherTeam="Leverkusen", where="Home"))
ag<-predict(m.team, type = "response",  newdata=data.frame(team="Leverkusen", otherTeam="Augsburg", where="Away"))
predoutcomes<-round(sapply(0:6, function(x) dpois(x, hg))%o%sapply(0:6, function(x) dpois(x, ag)), 4)*100
colnames(predoutcomes)<-0:6
rownames(predoutcomes)<-0:6
drawprob<-sum(diag(predoutcomes))
homewinprob<-sum(lower.tri(predoutcomes)*predoutcomes)
awaywinprob<-sum(upper.tri(predoutcomes)*predoutcomes)
data.frame(homewinprob, drawprob, awaywinprob)



# , teamresults$team, teamresults$otherTeam, teamresults$otherGoals, teamresults$where)

which.max(allpred)


names(fittedstatic)<-0:6


summary(fitted(m.team))

m.diff<-glm(formula = goals-otherGoals ~ (team+otherTeam)*where, data=teamresults, family = poisson)


homedefense<-glm(formula = FTHG ~ AwayTeam, data=results, family = poisson)


poisson.test(x=teamresults$goals, r = 0.3472)
poisson.test(137, 24.19893)

0.3472

reshape(results, timevar = "HomeTeam", direction = "wide", idvar = "spieltag")
recast(results, spieltag~HomeTeam~FTHG, id.var=c("HomeTeam", "spieltag", "FTHG"))

library(dplyr)
results %>% summarize(results)

aggregate(FTHG ~ HomeTeam, results, mean)
aggregate(FTAG ~ AwayTeam, results, mean)

homeattack<-glm(formula = FTHG ~ HomeTeam, data=results, family = poisson)
homedefense<-glm(formula = FTHG ~ AwayTeam, data=results, family = poisson)
awayattack<-glm(formula = FTAG ~ AwayTeam, data=results, family = poisson)
awaydefense<-glm(formula = FTAG ~ HomeTeam, data=results, family = poisson)

homegoals_x<-glm(formula = FTHG ~ HomeTeam*AwayTeam, data=results, family = poisson)
homegoals<-glm(formula = FTHG ~ HomeTeam+AwayTeam, data=results, family = poisson)
awaygoals_x<-glm(formula = FTAG ~ HomeTeam*AwayTeam, data=results, family = poisson)
awaygoals<-glm(formula = FTAG ~ HomeTeam+AwayTeam, data=results, family = poisson)
summary(homegoals)

predict(homegoals, newdata = data)
predict(homegoals)

summary(residuals(homegoals))
summary(residuals(awaygoals))
summary(residuals(homegoals_x))

summary(predict(homegoals, type = "response"))
summary(predict(awaygoals, type = "response"))
summary(predict(homegoals_x, type = "response"))
summary(predict(awaygoals_x, type = "response"))

cbind(results, H=predict(homegoals, type = "response"), A=predict(awaygoals, type = "response"))
cbind(results, 
      H=round(predict(homegoals_x, type = "response"), 2), 
      A=round(predict(awaygoals_x, type = "response"), 2))

plot(residuals(homegoals, type = "response")  ~ FTHG, data=results)
plot(results$FTAG, residuals(awaygoals, type = "response"))

plot(predict(homegoals, type = "response")  ~ FTHG, data=results)
plot(predict(awaygoals, type = "response")  ~ FTAG, data=results)

predict(homegoals, type = "response",  newdata=data.frame(HomeTeam="Dortmund", AwayTeam="Ingolstadt"))
predict(awaygoals, type = "response",  newdata=data.frame(HomeTeam="Dortmund", AwayTeam="Ingolstadt"))

predict(homegoals, newdata=data.frame(HomeTeam="Dortmund", AwayTeam="Bayern Munich"))

lambda<-predict(homegoals, type = "response", newdata=data.frame(HomeTeam="Dortmund", AwayTeam="Bayern Munich"))
lambda2<-predict(awaygoals, type = "response", newdata=data.frame(HomeTeam="Dortmund", AwayTeam="Bayern Munich"))

plot(dpois(0:10, lambda))
plot(dpois(0:10, lambda2))

dpois(0:5, lambda) %o% dpois(0:5, lambda2)

exp(-lambda)*lambda^4/factorial(4)
  

exp(0.99373-0.02707-0.95141)

0.99373-0.02707-0.03221

dpois(0, fitted(homegoals))
dpois(1, fitted(homegoals))
dpois(2, fitted(homegoals))
dpois(3, fitted(homegoals))
dpois(0:10, fitted(homegoals))



allmodel

table(results, HomeTeam~FTHG)

results


results2 <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR')]
summary(results2)
table(results2$HTR, results2$FTR) / nrow(results2) * 100
table(results2$HTR, results2$HTR)

table(results2$HTHG)
table(results2$FTHG-results2$HTHG)
table(results2$HTAG)
table(results2$FTAG-results2$HTAG)
