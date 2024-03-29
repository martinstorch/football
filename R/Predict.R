setwd("~/LearningR/Bundesliga/")
library(pscl)
download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
data2<-read.csv("BL2016.csv")
data2$season<-2016
data<-read.csv("BL2015.csv")
data$season<-2015
data<-rbind(data, data2)
teams <- unique(data$HomeTeam)
results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season')]
table(results$FTR , results$season)
results$spieltag <- floor((9:(nrow(results)+8))/9)

teamresults <- data.frame(team=results$HomeTeam, otherTeam=results$AwayTeam, 
                          goals=results$HTHG, otherGoals=results$HTAG, where="Home", spieltag=results$spieltag, season=results$season)
teamresults <- rbind(data.frame(team=results$AwayTeam, otherTeam=results$HomeTeam, 
                                goals=results$HTAG, otherGoals=results$HTHG, where="Away", spieltag=results$spieltag, season=results$season),
                     teamresults)

teamresults <- data.frame(team=results$HomeTeam, otherTeam=results$AwayTeam, 
                     goals=results$FTHG, otherGoals=results$FTAG, where="Home", spieltag=results$spieltag, season=results$season)
teamresults <- rbind(data.frame(team=results$AwayTeam, otherTeam=results$HomeTeam, 
                          goals=results$FTAG, otherGoals=results$FTHG, where="Away", spieltag=results$spieltag, season=results$season),
                     teamresults)
table(teamresults[teamresults$where=='Home', 'goals'], teamresults[teamresults$where=='Home', 'otherGoals'])
teamresults$goals <- sapply(teamresults$goals, min, 4)
teamresults$otherGoals <- sapply(teamresults$otherGoals, min, 4)

weights<-(teamresults$season-2014)*1.02^teamresults$spieltag

m.diff<-glm(formula = goals-otherGoals ~ (team+otherTeam)*where, data=teamresults, weights = weights, family = gaussian())
teamresults$diffpred <-fitted(m.diff)
summary(m.diff)  
plot(m.diff)
plot(diffpred ~ I(goals-otherGoals), data=teamresults )
abline(lm(diffpred ~ I(goals-otherGoals), data=teamresults ))
cor(fitted(m.diff), teamresults$goals-teamresults$otherGoals)
table(teamresults$goals-teamresults$otherGoals, round(fitted(m.diff)))
table((teamresults$goals-teamresults$otherGoals)==round(fitted(m.diff)))
table(sign(teamresults$goals-teamresults$otherGoals)==sign(round(fitted(m.diff))))
table(sign(teamresults$goals-teamresults$otherGoals),sign(round(fitted(m.diff))))

qqplot(y=teamresults$goals-teamresults$otherGoals, x = rnorm(nrow(teamresults)))
hist(teamresults$goals-teamresults$otherGoals)
table(teamresults$goals-teamresults$otherGoals)

# teamresults[teamresults$otherTeam=='RB Leipzig',]
m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = poisson, weights = weights)

# m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = quasipoisson, weights = weights)
# m.team<-hurdle(formula = goals ~ (team+otherTeam)*where+diffpred, data=teamresults, dist = "negbin", weights = weights)
plot(teamresults$goals, fitted(m.team))
summary(m.team)
summary(dpois(teamresults$goals, fitted(m.team)))
summary(dpois(teamresults$goals, 0))
summary(dpois(teamresults$goals, 1))
summary(dpois(teamresults$goals, 2))



fr <- teamresults[teamresults$where=='Home',]
homediff <- predict(m.diff, type = "response",  newdata=data.frame(team=fr$team, otherTeam=fr$otherTeam, where="Home"))
awaydiff <- predict(m.diff, type = "response",  newdata=data.frame(team=fr$otherTeam, otherTeam=fr$team, where="Away"))
fr$lh <- predict(m.team, type = "response",  newdata=data.frame(team=fr$team, otherTeam=fr$otherTeam, where="Home", diffpred=homediff))
fr$la <- predict(m.team, type = "response",  newdata=data.frame(team=fr$otherTeam, otherTeam=fr$team, where="Away", diffpred=awaydiff))
plot(lh-la ~ I(goals-otherGoals), data=fr )
abline(lm(lh-la ~ I(goals-otherGoals), data=fr ))
summary(lm(lh-la ~ I(goals-otherGoals), data=fr ))
cor(fr$lh-fr$la, fr$goals-fr$otherGoals)

cor(fr$lh-fr$la, homediff)
table(round(homediff))
table(fr$goals-fr$otherGoals)
table(round(homediff), fr$goals-fr$otherGoals)

table(sign(round(homediff)))
table(sign(fr$goals-fr$otherGoals))
table(sign(round(homediff)), sign(fr$goals-fr$otherGoals))

table(round(fr$lh-fr$la))
table(fr$goals-fr$otherGoals)
table(round(fr$lh-fr$la), fr$goals-fr$otherGoals)

table(sign(round(fr$lh-fr$la)))
table(sign(fr$goals-fr$otherGoals))
table(sign(round(fr$lh-fr$la)), sign(fr$goals-fr$otherGoals))

table(round(fr$lh-fr$la))
table(round(homediff))
table(round(fr$lh-fr$la), round(homediff))

table(sign(round(fr$lh-fr$la)))
table(sign(round(homediff)))
table(sign(round(fr$lh-fr$la)), sign(round(homediff)))




points(homediff ~ I(goals-otherGoals), data=fr, col=2 )
abline(lm(homediff ~ I(goals-otherGoals), data=fr ))
summary(lm(homediff ~ I(goals-otherGoals), data=fr , col=2))
cor(homediff, fr$goals-fr$otherGoals)


lambdas<-cbind(sapply(0:6, function(x) dpois(x, fr$lh)), sapply(0:6, function(x) dpois(x, fr$la)))
str(lambdas)
colnames(lambdas)<-c(paste0('LH', 0:6), paste0('LA', 0:6))
predoutcomes<-apply(lambdas, 1, function(x) {x[1:7]%o%x[8:14]})
predoutcomes<-t(predoutcomes)
tend<-apply(predoutcomes, 1, function(x) {
  rm<-matrix(7,7,data=x);
  c(
  homewinprob = sum(lower.tri(rm)*rm),
  drawprob=sum(diag(rm)),
  awaywinprob = sum(upper.tri(rm)*rm))
})
tend<-t(tend)
summary(tend)
table(apply(tend, 1, which.max))
table(sign(fr$goals-fr$otherGoals))
table(apply(tend, 1, which.max), sign(fr$goals-fr$otherGoals))

m.diff<-lm(formula = goals-otherGoals ~ (team+otherTeam)*where, data=teamresults, weights = weights)
teamresults$diffpred <-fitted(m.diff)
summary(m.diff)  
plot(m.diff)

plot(diffpred ~ I(goals-otherGoals), data=teamresults )
abline(lm(diffpred ~ I(goals-otherGoals), data=teamresults ))




allpred<-sapply(0:6, function(x) dpois(x, fitted(m.team)))
bestpred<-apply(allpred, 1, which.max)-1
table(data.frame(pred=bestpred, act=teamresults$goals)) #, diff=bestpred - teamresults$goals)   )
summary(data.frame(pred=bestpred, act=teamresults$goals))

predictMatch <- function(t1, t2) {
  team <- t1
  otherTeam <- t2
  
  hg<-predict(m.team, type = "response",  newdata=data.frame(team=team, otherTeam=otherTeam, where="Home"))
  ag<-predict(m.team, type = "response",  newdata=data.frame(team=otherTeam, otherTeam=team, where="Away"))
  hgdist<-sapply(0:6, function(x) dpois(x, hg))
  agdist<-sapply(0:6, function(x) dpois(x, ag))
  
  predoutcomes<-round(sapply(0:6, function(x) dpois(x, hg))%o%sapply(0:6, function(x) dpois(x, ag)), 4)*100
  colnames(predoutcomes)<-0:6
  rownames(predoutcomes)<-0:6
  drawprob<-sum(diag(predoutcomes))
  homewinprob<-sum(lower.tri(predoutcomes)*predoutcomes)
  awaywinprob<-sum(upper.tri(predoutcomes)*predoutcomes)
  return (list(tendency = data.frame(team=t1, otherTeam=t2, homewinprob, drawprob, awaywinprob, 
                                     hg=which.max(hgdist)-1, ag=which.max(agdist)-1), pred=predoutcomes)
  )
}


str(tend)
matrix(7,7,data = predoutcomes[1,])
lambdas[1,]

str((predoutcomes))

table(sign(fr$lh-fr$la), sign(fr$goals-fr$otherGoals))

ppois(0, 1)+dpois(1,1)
dpois(0,1)
ppois(0,1)
ppois(2, 1, lower.tail = F)
ppois(0, 1, lower.tail = T)
ppois(0, 1, lower.tail = F)

densityplot(lh-la ~ I(goals-otherGoals), data=fr)
  
  fittedresults$goals - fittedresults$otherGoals, )

hg<-predict(m.team, type = "response",  newdata=data.frame(team=team, otherTeam=otherTeam, where="Home"))
ag<-predict(m.team, type = "response",  newdata=data.frame(team=otherTeam, otherTeam=team, where="Away"))


allgamespred<-apply(results, 1, function(x) {predictMatch(x[['HomeTeam']], x[['AwayTeam']])})
allgames_tenpred<-(sapply(allgamespred, function(x) x$tendency[, c('homewinprob', 'drawprob', 'awaywinprob')]))
allgames_tenpred<-t(allgames_tenpred)
allgames_tenpred[,c('homewinprob', 'drawprob', 'awaywinprob')]
str(as.matrix(allgames_tenpred))
actualtend<-cbind(ifelse(results$FTR=='H', 1, 0), ifelse(results$FTR=='D', 1, 0), ifelse(results$FTR=='A', 1, 0))
str(actualtend)
as.matrix(allgames_tenpred)*cbind(ifelse(results$FTR=='H', 1, 0), ifelse(results$FTR=='D', 1, 0), ifelse(results$FTR=='A', 1, 0))

summary(unlist(ifelse(results$FTR=='H', allgames_tenpred[,1], ifelse(results$FTR=='D', allgames_tenpred[,2], allgames_tenpred[,3]))))

table(apply(allgames_tenpred[,c('homewinprob', 'drawprob', 'awaywinprob')], 1, function(x) which.max(x)))


allgames_tenpred[1:2,]

str(results)

results$HomeTeam
results$AwayTeam

       teams
predictMatch(teams[11],teams[17])
predictMatch(teams[15],teams[3])
predictMatch(teams[4],teams[9])
predictMatch(teams[6],teams[7])
predictMatch(teams[10],teams[1])
predictMatch(teams[13],teams[19])
predictMatch(teams[18],teams[20])
predictMatch(teams[12],teams[16])



t1<-teams[18]
t2<-teams[20]
table(results$FTHG, results$FTAG)
var(results$FTHG)
mean(results$FTHG)
41.89+29.6+28.47
var(results$FTAG)
mean(results$FTAG)


var(teamresults$goals)
mean(teamresults$goals)
var(teamresults$otherGoals)
mean(teamresults$otherGoals)


predictMatch(teams[11],teams[17])
predictMatch(teams[11],teams[17])



teams[3]
t1<-1
t2<-12


colnames(teams[5])
str(teams)

predict(m.team, type = "response",  newdata=data.frame(team="Augsburg", otherTeam="Leverkusen", where="Home"))



library(pscl)
m.team<-hurdle(formula = goals ~ team*where+otherTeam, data=teamresults, dist = "poisson")
m.team<-hurdle(formula = goals ~ (team+otherTeam)*where, data=teamresults, dist = "geometric")
m.team<-hurdle(formula = goals ~ (team+otherTeam)*where, data=teamresults, dist = "negbin")
summary(m.team)

fittedgoals<-round(dpois(0:6, (fitted(m.team)[0]))*nrow(teamresults))
names(fittedgoals)<-0:6
rbind(fittedgoals, actual=table(teamresults$goals))
rbind(fittedstatic, actual=table(teamresults$goals))

plot(teamresults$goals, fitted(m.team))
boxplot(fitted(m.team) ~ teamresults$goals)
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

library(MNP) # loads the MNP package
example(mnp) # runs the example script
detergent
m.probit<-mnp(formula = sign(goals-otherGoals)~I(as.integer(team)%%10), data=teamresults, verbose=T)
summary(m.probit)
m.probitdiff<-mnp(formula = (goals-otherGoals)~(team+otherTeam)*where, data=teamresults, verbose=T)
summary(m.probitdiff)

predict(m.probit, newdata = teamresults[1:10,])

residuals(m.probit)

as.integer(teamresults$team)
