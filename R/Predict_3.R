setwd("~/LearningR/Bundesliga/")
library(pscl)
library(amen)
library(Matrix)

loadData <- function() {
  download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
  download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
  data<-read.csv("BL2016.csv")
  data$season<-2016
  #data2<-read.csv("BL2015.csv")
  #data2$season<-2015
  #data<-read.csv("BL2015.csv")
  #data$season<-2015
  #data<-rbind(data, data2)
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
  
  teamresults$goals <- sapply(teamresults$goals, min, 4)
  teamresults$otherGoals <- sapply(teamresults$otherGoals, min, 4)
  
  teamresults$weights<-(teamresults$season-2014)*1.02^teamresults$spieltag
  return (list(games = results, teamresults = teamresults))
}

buildMask <- function() {
  cn<-expand.grid(0:4, 0:4)
  mask<-list()
  for (key in paste(cn$Var1, cn$Var2))
    mask[[key]]<-matrix(5,5,data=c(0), dimnames = list(0:4, 0:4))
  for (i in 0:4)
    for (j in 0:4) {  # draw
      mask[[paste(i,i)]][j+1,j+1]<-2
      mask[[paste(i,i)]][i+1,i+1]<-6
    }
  for (i in 1:4)
    for (j in 0:(i-1))  { 
      for (k in 1:4)
        for (l in 0:(k-1)) { # home 
          mask[[paste(i,j)]][k+1,l+1]<-2
          if (i-j==k-l) mask[[paste(i,j)]][k+1,l+1]<-3
        }
      mask[[paste(i,j)]][i+1,j+1]<-4
    }
  for (i in 0:3)
    for (j in (i+1):4)  { 
      for (k in 0:3)
        for (l in (k+1):4) { # home 
          mask[[paste(i,j)]][k+1,l+1]<-4
          if (i-j==k-l) mask[[paste(i,j)]][k+1,l+1]<-5
        }
      mask[[paste(i,j)]][i+1,j+1]<-7
    }
  return(mask)
}
ld<-loadData()
mask<-buildMask()
games<-ld$games[ld$games$season==2016,]
games$FTHG<-sapply(games$FTHG, min, 4)
games$FTAG<-sapply(games$FTAG, min, 4)
games$HTHG<-sapply(games$HTHG, min, 4)
games$HTAG<-sapply(games$HTAG, min, 4)
games$FTDIFF<-(games$FTHG-games$FTAG)
games$HTDIFF<-(games$HTHG-games$HTAG)

teams<-unique(games$AwayTeam)

Y1<-spMatrix(nrow=length(teams), 
         ncol=length(teams), 
         i = as.integer(games$HomeTeam), 
         j=as.integer(games$AwayTeam), 
         x = games$FTHG) # - games$FTAG)

Y<-as.matrix(Y1)

Y<-matrix(nrow=length(teams), 
          ncol=length(teams), 
          data=NA)
rownames(Y)<-levels(teams)
colnames(Y)<-levels(teams)
Xd<-array(dim=c(length(teams),length(teams),3),  
          data=NA, dimnames = list(home=levels(teams), away=levels(teams), 
                                   info=c('HTHG', 'HTAG', 'HTDIFF')))
rownames(Y)<-levels(teams)
colnames(Y)<-levels(teams)

for (i in 1:nrow(games))
{
  g <- games[i,]
  Y[as.integer(g$HomeTeam), as.integer(g$AwayTeam)]<-g$FTHG # -g$FTAG
  Xd[as.integer(g$HomeTeam), as.integer(g$AwayTeam),1]<-g$HTHG
  Xd[as.integer(g$HomeTeam), as.integer(g$AwayTeam),2]<-g$HTAG
  Xd[as.integer(g$HomeTeam), as.integer(g$AwayTeam),3]<-g$HTHG-g$HTAG
}

fit_SRM<-ame(Y, Xd=Xd[,,c(1,3)], nscan=5000, plot=TRUE, print=TRUE)
summary(fit_SRM)
fit_SRM$YPM

fit_SRM_bas<-ame(Y, nscan=5000, odens=10, plot=TRUE, print=TRUE)
str(fit_SRM_bas)
summary(fit_SRM_bas)
plot(fit_SRM_bas)
plot(fit_SRM)

mean(fit_SRM_bas$BETA)

fit_SRM_bas$YPM

Y[2,1]
fit_SRM_bas$YPM[2,1]
Y[1,2]
fit_SRM_bas$YPM[1,2]

fit_SRM_bas$APM[2]+fit_SRM_bas$BPM[1]+mean(fit_SRM_bas$BETA)
fit_SRM_bas$EZ[2,1]

fit_SRM_bas$APM[1]+fit_SRM_bas$BPM[2]+mean(fit_SRM_bas$BETA)
fit_SRM_bas$EZ[1,2]
fit_SRM_bas$YPM[1,2]

plot(sort(fit_SRM_bas$BETA))


gofstats(Y)
gofstats(fit_SRM$YPM)

c(Y)

str(fit_SRM)



Rowcountry<-matrix(rownames(Y),nrow(Y),ncol(Y))
Colcountry<-t(Rowcountry)
anova(lm( c(Y) ~ c(Rowcountry) + c(Colcountry) ) )
rmean<-rowMeans(Y,na.rm=TRUE) ; cmean<-colMeans(Y,na.rm=TRUE)
muhat<-mean(Y,na.rm=TRUE)
ahat<-rmean-muhat
bhat<-cmean-muhat
sd(ahat)
# additive "exporter" effects
head( sort(ahat,decreasing=TRUE) )
cov( cbind(ahat,bhat) )
cor( ahat, bhat)
R <- Y - ( muhat + outer(ahat,bhat,"+") )
cov( cbind( c(R),c(t(R)) ), use="complete")
cor( c(R),c(t(R)), use="complete")

plot( c(fit_SRM$YPM), c(Y))
plot( c(Y), c(fit_SRM$YPM)-c(Y))
plot(c(fit_SRM$YPM), c(fit_SRM$YPM-Y))
cor(c(fit_SRM$YPM-Y), c(fit_SRM$YPM), use = "na.or.complete")
cor(c(Y), c(fit_SRM$YPM), use = "na.or.complete")

summary(c(fit_SRM$YPM))

str(Y)
summary(fit_SRM$BETA)
fit_SRM$U
fit_SRM$V
summary(fit_SRM$GOF)

library(hclust)
cl<-hclust(d = dist(fit_SRM$U[,1]))
plot(cl)


apply(fit_SRM$GOF,2,mean)
gofstats(Y)


fit_SRM<-ame(Y, Xd=Xd[,,c(1,3)], nscan=5000, R=4, model="nrm", plot=TRUE, print=TRUE)
summary(fit_SRM)
muhat
cov( cbind(ahat,bhat))
apply(fit_SRM$BETA,2,mean)

fit_SRM$BETA
str(fit_SRM)

gofstats(fit_SRM)
fit_SRM$U[,1] %*% t(fit_SRM$V[,1]) == fit_SRM$UVPM
((fit_SRM$U[,1] %*% t(fit_SRM$V[,1]))
+(fit_SRM$U[,2] %*% t(fit_SRM$V[,2]))
+(fit_SRM$U[,3] %*% t(fit_SRM$V[,3]))
+(fit_SRM$U[,4] %*% t(fit_SRM$V[,4]))
)[1:6,1:6]
Y[1:6,1:6]
fit_SRM$UVPM[1:6,1:6]
fit_SRM$YPM

nextmatches<-c(
  "Augsburg", "RB Leipzig",
  "Werder Bremen", "Darmstadt",
  "Dortmund", "Leverkusen",	
  "Mainz", "Wolfsburg",
  "FC Koln", "Bayern Munich", 
  "Hoffenheim", "Ingolstadt",
  "M'gladbach", "Schalke 04",
  "Ein Frankfurt", "Freiburg",
  "Hamburg", "Hertha"
)

nextmatches<-c(
  "Leverkusen", "Werder Bremen",
  "Darmstadt", "Mainz",
  "RB Leipzig", "Wolfsburg",	
  "Hertha", "Dortmund",
  "Freiburg", "Hoffenheim", 
  "Bayern Munich", "Ein Frankfurt",
  "Ingolstadt", "FC Koln",
  "Schalke 04", "Augsburg",
  "Hamburg", "M'gladbach"
)
nm<-matrix(data = nextmatches, ncol=2, byrow = T)

sapply(1:9, function(i) paste(nm[i,1], "-", nm[i,2], ": ", fit_SRM$YPM[nm[i,1], nm[i,2]]))




fit_rm<-ame(Y,Xd=Xd[,,3],rvar=FALSE,cvar=FALSE,dcor=FALSE, nscan=5000, plot=TRUE, print=TRUE)
summary(fit_rm)
buildModel <- function(teamresults) {
  
  # m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = poisson, weights = teamresults$weights)
  
  # m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = quasipoisson, weights = weights)
  m.team<-hurdle(formula = goals ~ (team+otherTeam)*where, data=teamresults, dist = "negbin", weights = weights)
  plot(teamresults$goals, fitted(m.team))
  print(summary(m.team))
  print(summary(dpois(teamresults$goals, fitted(m.team))))
  # summary(dpois(teamresults$goals, 0))
  # summary(dpois(teamresults$goals, 1))
  # summary(dpois(teamresults$goals, 2))
  return(m.team)
}

predictMatches<-function(model, newmatches) {
  newmatches$lh <- predict(object=model, type = "response",  
                           newdata=data.frame(team=newmatches$team, otherTeam=newmatches$otherTeam, where="Home"))
  newmatches$la <- predict(object=model, type = "response",  
                           newdata=data.frame(team=newmatches$otherTeam, otherTeam=newmatches$team, where="Away"))

  lambdas<-cbind(sapply(0:4, function(x) dpois(x, newmatches$lh)), sapply(0:4, function(x) dpois(x, newmatches$la)))
  colnames(lambdas)<-c(paste0('LH', 0:4), paste0('LA', 0:4))
  predoutcomes<-apply(lambdas, 1, function(x) {x[1:5]%o%x[6:10]})
  predoutcomes<-t(predoutcomes)
  
  cn<-expand.grid(0:4, 0:4)
  colnames(predoutcomes)<-paste(cn$Var1, cn$Var2)
  predhg<-apply(lambdas[,1:5], 1, which.max)-1
  predag<-apply(lambdas[,6:10], 1, which.max)-1
  return (list(newmatches=newmatches, predoutcomes=predoutcomes, predgoals=data.frame(hg=predhg, ag=predag)))
}

recommend <- function(prediction) {
  tend<-apply(prediction$predoutcomes, 1, function(x) {
    rm<-matrix(5,5,data=x);
    c(
      homewinprob = sum(lower.tri(rm)*rm),
      drawprob=sum(diag(rm)),
      awaywinprob = sum(upper.tri(rm)*rm),
      prediction = which.max(x)
      )
  })
  tend<-t(tend)
  return(cbind(prediction$newmatches, tend[,1:3], pred=colnames(prediction$predoutcomes)[tend[,4]]))
}

maxExpectation <- function(predoutcomes) {
  expectedValues<-sapply(1:25, function(i) predoutcomes %*% unlist(mask[i]), simplify = "array")
  colnames(expectedValues)<-names(mask)
  ordering<-t(apply(-expectedValues, 1, order)[1:3,])
  data.frame(
    best=colnames(expectedValues)[ordering[,1]],
    exp=apply(expectedValues, 1, max),
    best2=colnames(expectedValues)[ordering[,2]],
    exp2=apply(expectedValues, 1, function(x) {x[order(-x)[2]]}),
    best3=colnames(expectedValues)[ordering[,3]],
    exp3=apply(expectedValues, 1, function(x) {x[order(-x)[3]]})
  )
}

ld<-loadData()
mask<-buildMask()
model<-buildModel(ld$teamresults)

newmatches<-ld$teamresults[teamresults$where=='Home',c('team', 'otherTeam')]
prediction <- predictMatches(model, newmatches)

table(prediction$predgoals$hg, prediction$predgoals$ag)
table(ld$games$HTHG, ld$games$HTAG)
table(prediction$predgoals$hg, ld$games$HTHG)
table(prediction$predgoals$ag, ld$games$HTAG)
qqplot(prediction$predgoals$hg, ld$games$HTHG)
qqplot(prediction$predgoals$ag, ld$games$HTAG)
plot(ld$games$HTHG, prediction$newmatches$lh)
plot(ld$games$HTHG - ld$games$HTAG, prediction$newmatches$lh - prediction$newmatches$la)
cor(ld$games$HTHG - ld$games$HTAG, prediction$newmatches$lh - prediction$newmatches$la)

cor(prediction$newmatches$lh, ld$games$HTHG)
plot(ld$games$HTHG, x=prediction$newmatches$lh)
plot(ld$games$HTAG, prediction$newmatches$la)


recommend(prediction)

nextmatches<-c(
"Wolfsburg", "Werder Bremen",
"Bayern Munich", "Hamburg",
"Leverkusen", "Mainz",	
"Darmstadt", "Augsburg",
"Freiburg", "Dortmund", 
"RB Leipzig", "FC Koln",
"Hertha", "Ein Frankfurt",
"Ingolstadt", "M'gladbach",
"Schalke 04", "Hoffenheim"
)
nextmatches<-as.data.frame(matrix(nextmatches,ncol=2,byrow=TRUE))
colnames(nextmatches)<-c('team', 'otherTeam')

prediction <- predictMatches(model, nextmatches)
recommend(prediction)
cbind(recommend(prediction), maxExpectation(prediction$predoutcomes))
cbind(prediction$newmatches, ld$games)
sum(maxExpectation(prediction$predoutcomes)$exp)
sum(maxExpectation(prediction$predoutcomes)$exp2)
sum(maxExpectation(prediction$predoutcomes)$exp3)

plotGamePred<-function(pred) {
  ord<-order(pred, decreasing = T)
  plot(pred[ord])
  text(pred[ord], names(pred[ord]))
  maxExpectation(pred)
}
sort(prediction$predoutcomes[1,], decreasing = T)
plot(sort(prediction$predoutcomes[1,], decreasing = T))

text(sort(prediction$predoutcomes[1,], decreasing = T), names(sort(prediction$predoutcomes[1,], decreasing = T)))

plotGamePred(prediction$predoutcomes[1,])
pred<-prediction$predoutcomes[1,]

labels

apply(expectedValues, 1, max)

expectedValues[9,order(-expectedValues[9,])]

matrix(expectedValues[8,], nrow=5, ncol=5, dimnames = list(0:4, 0:4))
matrix(prediction$predoutcomes[8,], nrow=5, ncol=5, dimnames = list(0:4, 0:4))
prediction$predoutcomes[1,]

sum(prediction$predoutcomes %*% unlist(mask[1]))
sum(prediction$predoutcomes[1,] * unlist(mask[1]))
sum(prediction$predoutcomes[1,] * unlist(mask[20]))

cbind(unlist(mask[2]), names(mask), prediction$predoutcomes[1,], names(prediction$predoutcomes[1,]))
rowSums(prediction$predoutcomes * unlist(mask[2]))

prediction$predoutcomes[1,]

ld$teamresults[teamresults$where=='Home',c('team', 'otherTeam')]

teams










fr <- teamresults[teamresults$where=='Home',]
fr$lh <- predict(m.team, type = "response",  newdata=data.frame(team=fr$team, otherTeam=fr$otherTeam, where="Home"))
fr$la <- predict(m.team, type = "response",  newdata=data.frame(team=fr$otherTeam, otherTeam=fr$team, where="Away"))
plot(lh-la ~ I(goals-otherGoals), data=fr )
abline(lm(lh-la ~ I(goals-otherGoals), data=fr ))
summary(lm(lh-la ~ I(goals-otherGoals), data=fr ))
cor(fr$lh-fr$la, fr$goals-fr$otherGoals)

lambdas<-cbind(sapply(0:4, function(x) dpois(x, fr$lh)), sapply(0:4, function(x) dpois(x, fr$la)))
str(lambdas)
colnames(lambdas)<-c(paste0('LH', 0:4), paste0('LA', 0:4))
predoutcomes<-apply(lambdas, 1, function(x) {x[1:5]%o%x[6:10]})
predoutcomes<-t(predoutcomes)

cn<-expand.grid(0:4, 0:4)
colnames(predoutcomes)<-paste(cn$Var1, cn$Var2)
tend<-apply(predoutcomes, 1, function(x) {
  rm<-matrix(5,5,data=x);
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
