setwd("~/LearningR/Bundesliga/")
library(pscl)
library(Matrix)

loadData <- function() {
  download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
  download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
  download.file("http://www.football-data.co.uk/mmz4281/1415/D1.csv", "BL2014.csv")
  data2<-read.csv("BL2016.csv")
  data2$season<-2016
  data<-read.csv("BL2015.csv")
  data$season<-2015
  data3<-read.csv("BL2014.csv")
  data3$season<-2014
  data<-rbind(data, data2)
  data<-rbind(data, data3[,colnames(data2)])
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
  
  teamresults$weights<-(teamresults$season-2013)*1.02^teamresults$spieltag
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

buildModel <- function(teamresults) {
  
  m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = poisson, weights = teamresults$weights)
  
  # m.team<-glm(formula = goals ~ (team+otherTeam)*where, data=teamresults, family = quasipoisson, weights = weights)
  #m.team<-hurdle(formula = goals ~ (team+otherTeam)*where, data=teamresults, dist = "negbin", weights = weights)
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
  
  return (list(newmatches=newmatches, predoutcomes=predoutcomes))
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
  expectedValues<-sapply(1:25, function(i) predoutcomes %*% unlist(mask[i]))
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

show_single_prediction <- function(probs, mask, teams=data.frame(team="home", otherTeam="away"), j=1) {
  ev<-sapply(1:25, function(i) probs %*% unlist(mask[i]))
  if (!is.array(ev)) ev<-array(data=ev, dim=c(1, 25))
  colnames(ev)<-names(mask)
  if (!is.array(probs)) probs<-array(data=probs, dim=c(1, 25))
  print(matrix(ev[j,], ncol=5, nrow=5, dimnames = list(paste(teams$team[j], 0:4),paste(teams$otherTeam[j], 0:4))))
  print(matrix(probs[j,]*100, ncol=5, nrow=5, dimnames = list(paste(teams$team[j], 0:4),paste(teams$otherTeam[j], 0:4))))
}

avg_match_results<-function(hometeam, awayteam) {
  weights<-(ld$games$season-min(ld$games$season)+1)*1.02^ld$games$spieltag 
  hometeam<-as.character(hometeam)
  awayteam<-as.character(awayteam)
  matches <- ld$games$HomeTeam==hometeam | ld$games$AwayTeam==awayteam
  results<- ld$games[matches,c('FTHG', 'FTAG')] 
  result_weights<-weights[matches]
  result_table<-table(results)
  results$FTHG<-sapply(results$FTHG, min, 4)
  results$FTAG<-sapply(results$FTAG, min, 4)
  wresults<-aggregate(by=list(results$FTHG, results$FTAG), x = result_weights, FUN=sum)
  result_wfreq<-c(wresults$x/sum(wresults$x))
  result_matrix<-spMatrix(result_wfreq*100, i = wresults$Group.1+1, j=wresults$Group.2+1, ncol=5, nrow=5)
  dimnames(result_matrix) <- list(paste(hometeam, 0:4),paste(awayteam, 0:4))
  names(result_wfreq)<-paste(wresults$Group.1, wresults$Group.2)
  homewinprob = sum(lower.tri(result_table)*result_table)/sum(result_table)
  drawprob=sum(diag(result_table))/sum(result_table)
  awaywinprob = sum(upper.tri(result_table)*result_table)/sum(result_table)
  
  return(list(result_table=result_table, result_wfreq=result_matrix, result_scores=data.frame(H=homewinprob, D=drawprob, A=awaywinprob)))
}

ld<-loadData()
mask<-buildMask()
model<-buildModel(ld$teamresults)

newmatches<-ld$teamresults[ld$teamresults$where=='Home',c('team', 'otherTeam')]
prediction <- predictMatches(model, newmatches)

pred_freq<-apply(prediction$predoutcomes, 2, mean)
actual_results<-ld$games[,c('FTHG', 'FTAG')]
actual_results$FTHG<-sapply(actual_results$FTHG, min, 4)
actual_results$FTAG<-sapply(actual_results$FTAG, min, 4)
act_freq<-c(table(actual_results)/nrow(actual_results))
act_freq[act_freq==0]<-0.001
gap<-data.frame(pred_freq, act_freq, adj_factor=act_freq/pred_freq)

{print(str(x * gap$adj_factor))}

probs<-prediction$predoutcomes
adj_probs<-t(apply(prediction$predoutcomes, 1, function(x) x * gap$adj_factor))

cbind(rowSums(probs), rowSums(adj_probs), prediction$newmatches)
rbind(colMeans(probs), colMeans(adj_probs), act_freq)


plot(rowSums(probs), prediction$newmatches$lh)
plot(rowSums(probs), prediction$newmatches$la)
plot(rowSums(probs), apply(cbind(prediction$newmatches$lh, prediction$newmatches$la), 1, max))
plot(prediction$newmatches$lh, prediction$newmatches$la)

plot(rowSums(probs), rowSums(adj_probs))
abline(0,1)
summary(rowSums(probs)); summary(rowSums(adj_probs))


str(prediction$predoutcomes)
str(gap$adj_factor)
summary(prediction$newmatches)
summary(actual_results)

str(prediction$newmatches)
prediction$newmatches[500:600,]
ld$games[500:600,]

table(ld$games[,c('FTHG', 'FTAG')])
table(ld$games[ld$games$HomeTeam=='Bayern Munich',c('FTHG', 'FTAG')])
table(ld$games[ld$games$AwayTeam=='Bayern Munich',c('FTHG', 'FTAG')])



recommend(prediction)

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

nextmatches<-c(
  "Dortmund", "Ingolstadt",
  "Werder Bremen", "RB Leipzig",
  "Augsburg", "Freiburg", 
  "Wolfsburg",	"Darmstadt", 
  "FC Koln", "Hertha", 
  "Hoffenheim", "Leverkusen",
  "Ein Frankfurt", "Hamburg", 
  "Mainz", "Schalke 04", 
  "M'gladbach", "Bayern Munich"
)

nextmatches<-c(
  "Hertha", "Hoffenheim", 
  "Schalke 04","Dortmund",
  "RB Leipzig","Darmstadt", 
  "Bayern Munich", "Augsburg", 
  "Hamburg",  "FC Koln", 
  "Freiburg", "Werder Bremen", 
  "Ein Frankfurt","M'gladbach", 
  "Ingolstadt",  "Mainz", 
  "Leverkusen","Wolfsburg"
)

nextmatches<-c(
  "FC Koln", "Ein Frankfurt",
  "Werder Bremen", "Schalke 04",
  "Hoffenheim", "Bayern Munich", 
  "Dortmund",   "Hamburg",  
  "Darmstadt", "Leverkusen",
  "Mainz", "RB Leipzig",
  "Augsburg",   "Ingolstadt", 
    "M'gladbach", "Hertha", 
  "Wolfsburg", "Freiburg"
)

nextmatches<-c(
  "Ein Frankfurt","Werder Bremen",
  "Freiburg", "Mainz", 
  "RB Leipzig", "Leverkusen",
  "Schalke 04",  "Wolfsburg",
  "FC Koln",   "M'gladbach", 
  "Hamburg",  "Hoffenheim", 
  "Bayern Munich", "Dortmund",   
  "Hertha",   "Augsburg",   
  "Ingolstadt", "Darmstadt"
)

nextmatches<-c(
  "Mainz", "Hertha",
  "Augsburg",     "FC Koln",
  "Dortmund",   "Ein Frankfurt",
  "Wolfsburg","Ingolstadt",
  "Hoffenheim", "M'gladbach", 
  "RB Leipzig", "Freiburg", 
  "Leverkusen","Bayern Munich", 
  "Werder Bremen", "Hamburg",   
  "Darmstadt",   "Schalke 04"
)

nextmatches<-c(
  "FC Koln", "Hoffenheim",
  "Ingolstadt","Werder Bremen",
  "Hamburg",     "Darmstadt",   
  "Bayern Munich", "Mainz", 
  "Ein Frankfurt", "Augsburg",     
  "Hertha","Wolfsburg",
  "M'gladbach", "Dortmund",   
   "Freiburg", "Leverkusen",
  "Schalke 04", "RB Leipzig"
)

nextmatches<-c(
  "Leverkusen",  "Schalke 04", 
  "Werder Bremen", "Hertha",
  "Mainz", "M'gladbach", 
  "Darmstadt",   "Freiburg", 
  "Dortmund",   "FC Koln",
  "RB Leipzig", "Ingolstadt",
  "Wolfsburg",  "Bayern Munich", 
  "Augsburg",       "Hamburg",    
  "Hoffenheim","Ein Frankfurt"
)

nextmatches<-c(
  "FC Koln", "Werder Bremen", 
  "M'gladbach",   "Augsburg",
  "Ein Frankfurt",  "Wolfsburg",  
  "Ingolstadt",  "Leverkusen",  
  "Bayern Munich", "Darmstadt", 
  "Dortmund",   "Hoffenheim",
    "Hertha",   "RB Leipzig",
  "Hamburg",      "Mainz", 
  "Freiburg",   "Schalke 04"
)

nextmatches<-c(
  "Wolfsburg",    "M'gladbach",   
  "Freiburg",     "Ingolstadt",  
  "Schalke 04"  , "Hamburg",      
  "Darmstadt",   "Hertha",  
  "Werder Bremen",   "Hoffenheim",
  "Leverkusen",    "FC Koln", 
  "RB Leipzig",  "Bayern Munich", 
  "Mainz",   "Ein Frankfurt",  
  "Augsburg",  "Dortmund"
)

nextmatches<-c(
  "Bayern Munich",   "Freiburg",
  "Hoffenheim",  "Augsburg",  
  "Hertha",    "Leverkusen",    
  "M'gladbach",     "Darmstadt",
  "Dortmund",   "Werder Bremen",   
  "Hamburg",      "Wolfsburg",    
  "Ein Frankfurt",    "RB Leipzig",  
  "FC Koln",   "Mainz",   
  "Ingolstadt",  "Schalke 04"  
)


extractProbs4<-function(m) {
  data.frame(
    H=sum(lower.tri(m)*m),
    D=sum(diag(m)),
    A=sum(upper.tri(m)*m),
    GMAX0 = m[1,1],
    GMAX1 = sum(m[1:2,1:2]),
    GMAX2 = sum(m[1:3,1:3]),
    GMAX3 = sum(m[1:4,1:4]),
    H1 = m[2,1]+m[3,2]+m[4,3]+m[5,4],
    H2 = m[3,1]+m[4,2]+m[5,3],
    H3 = m[4,1]+m[5,2],
    A1 = m[1,2]+m[2,3]+m[3,4]+m[4,5],
    A2 = m[1,3]+m[2,4]+m[3,5],
    A3 = m[1,4]+m[2,5]
  )
}

nextmatches<-as.data.frame(matrix(nextmatches,ncol=2,byrow=TRUE))
colnames(nextmatches)<-c('team', 'otherTeam')

prediction <- predictMatches(model, nextmatches)
recommend(prediction)
cbind(recommend(prediction), maxExpectation(prediction$predoutcomes))

adj_probs<-t(apply(prediction$predoutcomes, 1, function(x) x * gap$adj_factor))
adj_prediction<-prediction
adj_prediction$predoutcomes<-adj_probs
recommend(adj_prediction)
cbind(recommend(adj_prediction), maxExpectation(adj_prediction$predoutcomes))

sum(maxExpectation(prediction$predoutcomes)$exp)
sum(maxExpectation(prediction$predoutcomes)$exp2)
sum(maxExpectation(prediction$predoutcomes)$exp3)

sum(maxExpectation(adj_prediction$predoutcomes)$exp)
sum(maxExpectation(adj_prediction$predoutcomes)$exp2)
sum(maxExpectation(adj_prediction$predoutcomes)$exp3)

x<-1; 
show_single_prediction (prediction$predoutcomes, mask, prediction$newmatches, x) 
show_single_prediction (adj_prediction$predoutcomes, mask, prediction$newmatches, x) 
res<-avg_match_results(nextmatches[x,1], nextmatches[x,2]); res; 
show_single_prediction (c(as.matrix(res$result_wfreq))/100, mask) 

rbind(pred=extractProbs4(matrix(prediction$predoutcomes[x,], nrow = 5, ncol=5)),
      adj=extractProbs4(matrix(adj_prediction$predoutcomes[x,], nrow = 5, ncol=5)),
      hist=extractProbs4(as.matrix(res$result_wfreq)/100))



table(ld$games[,c('FTHG', 'FTAG')])
act<-ld$games[,c('FTHG', 'FTAG')]
act$FTHG<-sapply(act$FTHG, min, 4)
act$FTAG<-sapply(act$FTAG, min, 4)
act_p<-c(table(act)/nrow(act))

show_single_prediction (act_p, mask) 


x<-9; 
teamgamesH<-ld$games[ld$games$HomeTeam==as.character(nextmatches[x,1]),c('FTHG', 'FTAG')]; nextmatches[x,1]; table(teamgamesH); table(teamgamesH)/nrow(teamgamesH)*100
teamgamesA<-ld$games[ld$games$AwayTeam==as.character(nextmatches[x,2]),c('FTHG', 'FTAG')]; nextmatches[x,2]; table(teamgamesA); table(teamgamesA)/nrow(teamgamesA)*100
both<-rbind(teamgamesH, teamgamesA); table(both); table(both)/nrow(both)*100

hometeam<-"M'gladbach"; awayteam<-"Bayern Munich"
avg_match_results(hometeam, awayteam)
avg_match_results(awayteam, hometeam)



actual_results<-ld$games[,c('FTHG', 'FTAG')]
actual_results$FTHG<-sapply(actual_results$FTHG, min, 4)
actual_results$FTAG<-sapply(actual_results$FTAG, min, 4)
act_freq<-c(table(actual_results)/nrow(actual_results))
act_freq[act_freq==0]<-0.001
gap<-data.frame(pred_freq, act_freq, adj_factor=act_freq/pred_freq)





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



hometeam<-nextmatches[x,1]
awayteam<-nextmatches[x,2]
weights<-(ld$games$season-min(ld$games$season)+1)*1.02^ld$games$spieltag 
hometeam<-as.character(hometeam)
awayteam<-as.character(awayteam)
matches <- ld$games$HomeTeam==hometeam | ld$games$AwayTeam==awayteam
results<- ld$games[matches,c('FTHG', 'FTAG')] 
result_weights<-weights[matches]
result_table<-table(results)
results$FTHG<-sapply(results$FTHG, min, 4)
results$FTAG<-sapply(results$FTAG, min, 4)
wresults<-aggregate(by=list(results$FTHG, results$FTAG), x = result_weights, FUN=sum)
result_wfreq<-c(wresults$x/sum(wresults$x))
result_matrix<-spMatrix(result_wfreq*100, i = wresults$Group.1+1, j=wresults$Group.2+1, ncol=5, nrow=5)
dimnames(result_matrix) <- list(paste(hometeam, 0:4),paste(awayteam, 0:4))
names(result_wfreq)<-paste(wresults$Group.1, wresults$Group.2)
homewinprob = sum(lower.tri(result_table)*result_table)/sum(result_table)
drawprob=sum(diag(result_table))/sum(result_table)
awaywinprob = sum(upper.tri(result_table)*result_table)/sum(result_table)

return(list(result_table=result_table, result_wfreq=result_matrix, result_scores=data.frame(H=homewinprob, D=drawprob, A=awaywinprob)))





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
