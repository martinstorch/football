setwd("~/LearningR/Bundesliga/")
library(rjags)


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
  return(list(result_table=result_table, result_wfreq=result_matrix))
}

ld<-loadData()
mask<-buildMask()
model<-buildModel(ld$teamresults)


teams<-levels(ld$games$HomeTeam)
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

nextmatches<-as.data.frame(matrix(nextmatches,ncol=2,byrow=TRUE))
colnames(nextmatches)<-c('team', 'otherTeam')

teams<-levels(ld$games$HomeTeam)
outputGames<-list(hometeam=match(nextmatches$team, teams), awayteam=match(nextmatches$otherTeam, teams))
#,
#                  p.HG=rep(NA, nrow(nextmatches)), p.AG=rep(NA, nrow(nextmatches)))
inputData<-ld$teamresults[ld$teamresults$where=='Home',c('goals', 'otherGoals')]
inputData$team<-match(ld$teamresults[ld$teamresults$where=='Home','team'], teams)
inputData$otherTeam<-match(ld$teamresults[ld$teamresults$where=='Home','otherTeam'], teams)

model_string <- "model{
for(i in 1:length(goals)) {
  goals[i] ~ dcat(p[])
  #otherGoals[i] ~ dpois(defense[team[i]]*attack[otherTeam[i]])
  #spieltag[i] ~ dunif(0,100)
  #season[i] ~ dunif(2013,2016)
  #weights[i] ~ dnorm(0,1)
}
for(i in 1:max(goals)) {
  p[i] ~ dexp(1)
  for(j in 1:2) {
    t[i,j,1] ~ dunif(0,1)
  }
}
"
model <- jags.model(textConnection(model_string), 
                    data = list(goals=inputData$goals+1),
                    n.chains = 3, n.adapt= 1000)
adapt(model, 2000)
update(model, 1000); # Burnin for 10000 samples
mcmc_static <- coda.samples(model, variable.names=c("p"), n.iter=2000)
summary(mcmc_static)

cov(inputData$goals, inputData$otherGoals)

model_string <- "model{
  #test <- pow(-0.3, 2)
  test ~ dgamma(2.5,1.8)

for (j in 0:6) {
  for (k in 0:6) {
    n_xbivterm[j+1,k+1] <- 
    n_xqlambda * j*k     
    + pow(n_xqlambda, 2)*j*k/2*(j-1)*(k-1)
    + pow(n_xqlambda, 3)*j*k/3*(j-1)*(k-1)/2*(j-2)*(k-2)
    + pow(n_xqlambda, 4)*j*k/4*(j-1)*(k-1)/3*(j-2)*(k-2)/2*(j-3)*(k-3)
    + pow(n_xqlambda, 5)*j*k/5*(j-1)*(k-1)/4*(j-2)*(k-2)/3*(j-3)*(k-3)/2*(j-4)*(k-4)
    + pow(n_xqlambda, 6)*j*k/6*(j-1)*(k-1)/5*(j-2)*(k-2)/4*(j-3)*(k-3)/3*(j-4)*(k-4)/2*(j-5)*(k-5)
    + 1
    } 
} 
n_xqlambda<- -0.1

}"
model <- jags.model(textConnection(model_string))
mcmc_test <- coda.samples(model, variable.names=c("test", "n_xbivterm"), n.iter=20)
mcmc_test[,1]
plot(mcmc_test[,1])
mcmc_test[,2]
summary(mcmc_test)

model_string <- "
data
{
  for(i in 1:length(team)) 
  {
    zeros[i] <- 0
  }
}
model
{
  for(i in 1:length(hometeam)) {
    for (j in 0:6) {
      for (k in 0:6) {
        pScore[i, j+1, k+1]<-exp(n_logLike[i, j+1, k+1])
  
#        n_logLike[i, j+1, k+1] <- log( #max(0.00001, 
#          (1-equals(j,k))*  (1-pi)*exp(n_poisLogLike[i, j+1, k+1]) + 
#             equals(j,k) *( (1-pi)*exp(n_poisLogLike[i, j+1, k+1]) + pi*exp(j*log(theta) - theta -  logfact(j))) 
#        )#)
  
        n_logLike[i, j+1, k+1] <- n_poisLogLike[i, j+1, k+1]

        n_poisLogLike[i, j+1, k+1] <- 
          j*log(n_lambda_h[i]) - n_lambda_h[i] -  logfact(j) 
        + k*log(n_lambda_a[i]) - n_lambda_a[i] -  logfact(k)         
#        - lambda0 
#        + log(n_xbivterm[i,j+1,k+1])
        + equals(j, 0)*equals(k, 0)*log(1-n_lambda_h[i]*n_lambda_a[i]*rho)
        + equals(j, 0)*equals(k, 1)*log(1+n_lambda_h[i]*rho)
        + equals(j, 1)*equals(k, 0)*log(1+n_lambda_a[i]*rho)
        + equals(j, 1)*equals(k, 1)*log(1-rho)
  
#        n_xbivterm[i,j+1,k+1] <- n_xbivterm1[i,j+1,k+1] + n_xbivterm2[i,j+1,k+1]  
#        n_xbivterm1[i,j+1,k+1] <- 
#          n_xqlambda[i] * k *j 
#          + pow(n_xqlambda[i], 2)*j*k/2*(j-1)*(k-1)
#          + pow(n_xqlambda[i], 3)*j*k/3*(j-1)*(k-1)/2*(j-2)*(k-2)
#          + pow(n_xqlambda[i], 4)*j*k/4*(j-1)*(k-1)/3*(j-2)*(k-2)/2*(j-3)*(k-3)
#          + 1
#        n_xbivterm2[i,j+1,k+1] <- 
#            pow(n_xqlambda[i], 5)*j*k/5*(j-1)*(k-1)/4*(j-2)*(k-2)/3*(j-3)*(k-3)/2*(j-4)*(k-4)
#          + pow(n_xqlambda[i], 6)*j*k/6*(j-1)*(k-1)/5*(j-2)*(k-2)/4*(j-3)*(k-3)/3*(j-4)*(k-4)/2*(j-5)*(k-5)
  
      } 
    } 
#    n_xqlambda[i] <- lambda0/(n_lambda_h[i]*n_lambda_a[i])
    n_lambda_h[i]<-attack[hometeam[i]]*homeadv[hometeam[i]]*defense[awayteam[i]]
    n_lambda_a[i]<-defense[hometeam[i]]*attack[awayteam[i]]
    
    # p.HG[i] ~ dpois(attack[hometeam[i]]*homeadv[hometeam[i]]*defense[awayteam[i]])
    # p.AG[i] ~ dpois(defense[hometeam[i]]*attack[awayteam[i]])
  }
  for(i in 1:length(team)) {
    zeros[i] ~ dpois(phi[i])
    phi[i] <- -logLike[i] + 2
  
#    logLike[i] <- log( 
#      (1-isDraw[i])*  (1-pi)*exp(poisLogLike[i]) + 
#         isDraw[i] *( (1-pi)*exp(poisLogLike[i]) + pi*exp(goals[i]*log(theta) - theta -  logfact(goals[i]))) 
#    )
    logLike[i] <- poisLogLike[i]

    poisLogLike[i] <-
        goals[i]*log(lambda_h[i]) - lambda_h[i] -  logfact(goals[i]) 
      + otherGoals[i]*log(lambda_a[i]) - lambda_a[i] -  logfact(otherGoals[i])
#      - lambda0 
#      + log(xbivterm[i])
        + equals(goals[i], 0)*equals(otherGoals[i], 0)*log(1-lambda_h[i]*lambda_a[i]*rho)
        + equals(goals[i], 0)*equals(otherGoals[i], 1)*log(1+lambda_h[i]*rho)
        + equals(goals[i], 1)*equals(otherGoals[i], 0)*log(1+lambda_a[i]*rho)
        + equals(goals[i], 1)*equals(otherGoals[i], 1)*log(1-rho)
  
#    xbivterm[i] <- 
#      1 
#      + xcoeff[i,1]*    xqlambda[i] 
#      + xcoeff[i,2]*pow(xqlambda[i], 2) / 2
#      + xcoeff[i,3]*pow(xqlambda[i], 3) / 6
#      + xcoeff[i,4]*pow(xqlambda[i], 4) / 24
# 
#    xqlambda[i] <- lambda0/(lambda_h[i]*lambda_a[i])

#    xcoeff[i,4] <- (goals[i]-3) * (otherGoals[i]-3) * xcoeff[i,3]
#    xcoeff[i,3] <- (goals[i]-2) * (otherGoals[i]-2) * xcoeff[i,2]
#    xcoeff[i,2] <- (goals[i]-1) * (otherGoals[i]-1) * xcoeff[i,1]
#    xcoeff[i,1] <- goals[i] * otherGoals[i]

    lambda_h[i] <- attack[team[i]]*homeadv[team[i]]*defense[otherTeam[i]]
    lambda_a[i] <- defense[team[i]]*attack[otherTeam[i]]

    isDraw[i] <- equals(goals[i], otherGoals[i])

    #spieltag[i] ~ dunif(0,100)
    #season[i] ~ dunif(2013,2016)
    #weights[i] ~ dnorm(0,1)
  }
  for(j in 1:21) {
    attack[j] ~ dgamma(2.5,1.8) I(0.3, )
    defense[j] ~ dgamma(2.5,1.8) I(0.3, )
    homeadv[j] ~ dnorm(1.5,1) I(0.7, )
  }
  pi ~ dunif(0,0.2)
  theta ~ dunif(0,5)
  lambda0 ~ dunif(-1, 1)
  rho ~ dunif(-1, 1)
  #pi<-0
  #theta<-1
  #lambda0 <- -0.1
}"
cat_inputData<-inputData
#cat_inputData$goals<-min(cat_inputData$goals+1, 4)
#cat_inputData$otherGoals<-min(cat_inputData$otherGoals+1, 4)

model <- jags.model(textConnection(model_string), 
            data = c(as.list(cat_inputData), outputGames),
            n.chains = 3, n.adapt= 100)
adapt(model, 200)
update(model, 100); # Burnin for 10000 samples
mcmc_samples <- coda.samples(model, variable.names=c("lambda0", "pi","theta", "pScore", "attack", "defense", "homeadv", "n_xqlambda"), n.iter=200)
# mcmc_samples <- coda.samples(model, variable.names=c("pi","theta", "p.HG", "p.AG", "attack", "defense", "homeadv"), n.iter=2000)
mcmc_samples <- coda.samples(model, variable.names=c("lambda0", "pi","theta"), n.iter=2000)
mcmc_samples <- coda.samples(model, variable.names=c("lambda0", "pi","theta", "attack", "defense", "homeadv", "xbivterm", "xcoeff",  "xqlambda", "lambda_h", "lambda_a", "logLike"), n.iter=200)
mcmc_samples <- coda.samples(model, variable.names=c("rho", "attack", "defense", "homeadv", "lambda_h", "lambda_a", "logLike"), n.iter=200)


plot(mcmc_samples[,c("rho")])
summary(mcmc_samples)

summary(mcmc_samples[,c("pi","theta")])
plot(mcmc_samples[,c("pi","theta")])
plot(mcmc_samples[,c("lambda0")])
plot(mcmc_samples[,c("pi")])
plot(mcmc_samples[,c("theta")])
plot(mcmc_samples[,c("xqlambda[3]")])
plot(mcmc_samples[,c("xqlambda[5]")])
summary(mcmc_samples[,c("xqlambda[5]")])
plot(mcmc_samples[,c("xbivterm[3]")])
plot(mcmc_samples[,c("xbivterm[5]")])
plot(mcmc_samples[,c("xcoeff[3,1]")])
plot(mcmc_samples[,c("xcoeff[3,2]")])
plot(mcmc_samples[,c("xcoeff[5,1]")])
plot(mcmc_samples[,c("lambda_h[3]")])
plot(mcmc_samples[,c("lambda_a[3]")])
plot(mcmc_samples[,c("lambda_h[3]", "lambda_a[3]")])
plot(mcmc_samples[,c("lambda_h[5]", "lambda_a[5]", "xbivterm[5]")])
plot(mcmc_samples[,c("lambda_h[2]", "lambda_a[2]")])
plot(mcmc_samples[,c("attack[1]", "defense[1]", "homeadv[1]")])
plot(mcmc_samples[,c("attack[2]", "defense[2]", "homeadv[2]")])
plot(mcmc_samples[,c("attack[3]", "defense[3]", "homeadv[3]")])
plot(mcmc_samples[,c()])
plot(mcmc_samples[,c("logLike[2]")])
plot(mcmc_samples[,c("logLike[3]")])
plot(mcmc_samples[,c("logLike[5]")])
plot(mcmc_samples[,c("n_xqlambda[2]", "n_xqlambda[4]", "n_xqlambda[8]", "n_xqlambda[9]")])

pi <- summary(mcmc_samples[,c("pi","theta")])$statistics[,1][["pi"]]
theta <- summary(mcmc_samples[,c("pi","theta")])$statistics[,1][["theta"]]
otherParams <-summary(mcmc_samples[,c("pi","theta")])$statistics[,1]

mcmc_means<-summary(mcmc_samples[,1:63])$statistics[,1]
estParams<-data.frame(attack=mcmc_means[1:21], defense=mcmc_means[22:42], homeadv=mcmc_means[43:63])
estResults<-data.frame(t1=outputGames$hometeam, 
                       t2=outputGames$awayteam,
                       t1name=teams[outputGames$hometeam], 
                       t2name=teams[outputGames$awayteam]
                       )
estResults$homegoals<-estParams$attack[estResults$t1]*estParams$homeadv[estResults$t1]*estParams$defense[estResults$t2]
estResults$awaygoals<-estParams$attack[estResults$t2]*estParams$defense[estResults$t1]
#estResults$sim_hg<-mcmc_means[varidxHome]
#estResults$sim_ag<-mcmc_means[varidxAway]
estResults

syntheticPrediction<-function(t1, t2, estTeamParams, otherParams, teams) {
  homegoals<-estTeamParams$attack[t1]*estTeamParams$homeadv[t1]*estTeamParams$defense[t2]
  awaygoals<-estTeamParams$attack[t2]*estTeamParams$defense[t1]
  pi <- otherParams[["pi"]]
  theta <- otherParams[["theta"]]
  probs<-expand.grid(HG=0:4, AG=0:4)
  probs$pHG<-dpois(x = probs$HG, lambda = homegoals)
  probs$pAG<-dpois(x = probs$AG, lambda = awaygoals)
  probs$pCombined<-probs$pHG*probs$pAG
  probs$pAdj<-probs$pCombined*(1-pi)
  probs$pAdj[probs$HG==probs$AG]<-probs$pAdj[probs$HG==probs$AG]+pi*dpois(x = 0:4, lambda = theta)
  m <- matrix(data = probs$pAdj*100, ncol=5, nrow=5, dimnames = list(paste(teams[t1], 0:4), paste(teams[t2], 0:4)))
  m
}

syntheticPredictionRho<-function(t1, t2, estTeamParams, otherParams, teams) {
  homegoals<-estTeamParams$attack[t1]*estTeamParams$homeadv[t1]*estTeamParams$defense[t2]
  awaygoals<-estTeamParams$attack[t2]*estTeamParams$defense[t1]
  rho <- otherParams[["rho"]]
  probs<-expand.grid(HG=0:4, AG=0:4)
  probs$pHG<-dpois(x = probs$HG, lambda = homegoals)
  probs$pAG<-dpois(x = probs$AG, lambda = awaygoals)
  probs$pCombined<-probs$pHG*probs$pAG
  probs$pAdj<-probs$pCombined
  probs$pAdj[1]<-probs$pAdj[1]*(1-rho*homegoals*awaygoals) # 0-0
  probs$pAdj[6]<-probs$pAdj[6]*(1+rho*homegoals) # 0-1
  probs$pAdj[2]<-probs$pAdj[2]*(1+rho*awaygoals) # 1-0
  probs$pAdj[7]<-probs$pAdj[7]*(1-rho) # 1-1
  print(paste(sum(probs$pCombined), sum(probs$pAdj)))
  m <- matrix(data = probs$pAdj*100, ncol=5, nrow=5, dimnames = list(paste(teams[t1], 0:4), paste(teams[t2], 0:4)))
  m
}
#otherParams <-summary(mcmc_samples[,c("pi", "theta")])$statistics[,1]
otherParams <-summary(mcmc_samples[,c("rho", "rho")])$statistics[,1]
mcmc_means<-summary(mcmc_samples[,1:63])$statistics[,1]
estParams<-data.frame(attack=mcmc_means[1:21], defense=mcmc_means[22:42], homeadv=mcmc_means[43:63])

predictGame<-function(i) {
  varidx<-grep(pattern = paste0("pScore\\[", i), x = varnames(mcmc_samples))
  matrix(data = summary(mcmc_samples[,varidx])$statistics[,1]*100, ncol=5, nrow=5, dimnames = list(0:4, 0:4))
}

i<-2; syntheticPredictionRho(outputGames$hometeam[i],outputGames$awayteam[i],estParams, otherParams, teams)
predictGame(i)

i<-2; syntheticPrediction(outputGames$hometeam[i],outputGames$awayteam[i],estParams, otherParams, teams)
predictGame(i)

i<-2; sum(syntheticPrediction(outputGames$hometeam[i],outputGames$awayteam[i],estParams, otherParams, teams))
sum(predictGame(i))


sum(predictGame(9))

check<-sapply(1:nrow(inputData), function(i) syntheticPrediction(inputData$team[i],inputData$otherTeam[i],estParams, otherParams, teams))
check<-t(check)

pred<-matrix(data = colSums(check)/nrow(check), ncol=7, nrow=7, dimnames = list(0:6, 0:6))
actual<-table(inputData$goals, inputData$otherGoals)*100/nrow(check)
pred[1:5, 1:5]/actual
pred[1:5, 1:5]
actual

actual<-table(inputData$goals, inputData$otherGoals)
actual<-cbind(actual, 0)
actual<-cbind(actual, 0)
actual<-rbind(actual, 0)
actual<-rbind(actual, 0)

chisq.test(pred, p = colSums(check)/nrow(check)/sum(colSums(check)/nrow(check)))
chisq.test(c(actual), p = colSums(check)/nrow(check)/sum(colSums(check)/nrow(check)))

p_pred<-pred/sum(pred)
predcounts<-c(p_pred)*nrow(check)
sum(predcounts)
chisq.test(x=(predcounts), p=p_pred)
chisq.test(x=(predcounts), y=c(actual))
chisq.test(x=round(predcounts), y=c(actual))


compare_counts<-cbind(pred=round(predcounts), act=c(actual))
compare_counts<-cbind(compare_counts, expand.grid(0:6,0:6))
compare_counts
sum(round(predcounts))
sum(actual)

chisq.test(x=compare_counts)


str(check)
inputData$team[133]


varidx<-grep(pattern = "pScore\\[5", x = varnames(mcmc_samples))
matrix(data = summary(mcmc_samples[,varidx])$statistics[,1]*100, ncol=5, nrow=5, dimnames = list(0:4, 0:4))
sum(summary(mcmc_samples[,varidx])$statistics[,1])


plot(mcmc_samples[,"pScore[5,1,3]"])




varidxHome<-grep(pattern = "p\\.HG", x = varnames(mcmc_samples))
varidxAway<-grep(pattern = "p\\.AG", x = varnames(mcmc_samples))
data.frame(t1name=teams[outputGames$hometeam], t2name=teams[outputGames$awayteam],
           goals = summary(mcmc_samples[,varidxHome])$statistics[,1],
           otherGoals = summary(mcmc_samples[,varidxAway])$statistics[,1])



predictGame<-function(i) {
  table(as.data.frame(as.matrix(mcmc_samples[,c(paste0("p.HG[", i, "]"), paste0("p.AG[", i, "]"))])))
}
predictGame(1)


summary(mcmc_samples[,c("homeadv[13]", "attack[13]", "defense[2]", "attack[2]", "defense[13]")])

summary(mcmc_samples[,c("hometeam[1]", "awayteam[1]", "hometeam[2]", "awayteam[2]", "defense[13]")])
summary(mcmc_samples[,c("team[1]", "otherTeam[1]", "team[2]", "otherTeam[2]", "defense[13]")])


i1<-grep(pattern = "^team\\[", x = varnames(mcmc_samples))
i2<-grep(pattern = "^otherTeam\\[", x = varnames(mcmc_samples))
teaminput<-cbind(summary(mcmc_samples[,i1])$statistics[,1], summary(mcmc_samples[,i2])$statistics[,1])
head(teaminput, 30)
table(teaminput[,1])
table(teaminput[,2])


summary(mcmc_samples[,1:18])


0.9919*1.0588*0.9433
0.9593*1.1116

plot(mcmc_samples[,1, drop=F])
densplot(mcmc_samples[,1, drop=F])
traceplot(mcmc_samples[,1, drop=F])
rejectionRate(mcmc_samples)
library(lattice)
qqmath(mcmc_samples[,1:5])
densityplot(mcmc_samples[,1:5])
xyplot(mcmc_samples[,1:5])
acfplot(mcmc_samples[,1:5])
geweke.diag(mcmc_samples)
geweke.plot(mcmc_samples)
heidel.diag(mcmc_samples)
HPDinterval(mcmc_samples)
gelman.diag(mcmc_samples)
gelman.plot(mcmc_samples)
cumuplot(mcmc_samples[,1:5])
crosscorr(mcmc_samples)
crosscorr.plot(mcmc_samples)
autocorr.diag(mcmc_samples)
autocorr.plot(mcmc_samples[,1:5])
raftery.diag(mcmc_samples)







model$state(internal=T)

model_test <- jags.model(textConnection("model{ q ~ dgamma(2.5,1.8)}"), 
                    data = list(),
                    n.chains = 3, n.adapt= 1000)
mcmc_test <- coda.samples(model_test, variable.names=c("q"), n.iter=2000)
plot(mcmc_test)
summary(mcmc_test)



plot(mcmc_samples[,46, drop=F])
plot(mcmc_samples[,45, drop=F])
plot(mcmc_samples[,2, drop=F])
plot(mcmc_samples[,1, drop=F])
plot(mcmc_samples[,24, drop=F])
plot(mcmc_samples[,25, drop=F])
x <- seq(0, 10, 0.01)
plot(x, dgamma(x , rate = 2.5, shape = 1.8))

summary(mcmc_samples[,45, drop=F])


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

x<-8; 
show_single_prediction (prediction$predoutcomes, mask, prediction$newmatches, x) 
show_single_prediction (adj_prediction$predoutcomes, mask, prediction$newmatches, x) 
res<-avg_match_results(nextmatches[x,1], nextmatches[x,2]); res; 
show_single_prediction (c(as.matrix(res$result_wfreq))/100, mask) 


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





nextmatches<-list(team=NA, otherTeam=NA)
teams<-levels(ld$games$HomeTeam)
outputGames<-list(hometeam=match(nextmatches$team, teams), awayteam=match(nextmatches$otherTeam, teams))
#,
#                  p.HG=rep(NA, nrow(nextmatches)), p.AG=rep(NA, nrow(nextmatches)))
inputData<-ld$teamresults[ld$teamresults$where=='Home',c('goals', 'otherGoals')]
inputData$team<-match(ld$teamresults[ld$teamresults$where=='Home','team'], teams)
inputData$otherTeam<-match(ld$teamresults[ld$teamresults$where=='Home','otherTeam'], teams)

model_string <- "
data
{
  for(i in 1:length(team)) 
  {
  zeros[i] <- 0
  }
}
model
{
  for(i in 1:length(hometeam)) {


    for (j in 0:6) {
      for (k in 0:6) {
        pScore[i, j+1, k+1]<-exp(n_logLike[i, j+1, k+1])
        
#        n_logLike[i, j+1, k+1] <- log( #max(0.00001, 
#        (1-equals(j,k))*  (1-pi)*exp(n_poisLogLike[i, j+1, k+1]) + 
#        equals(j,k) *( (1-pi)*exp(n_poisLogLike[i, j+1, k+1]) + pi * drawprob[j+1]/sum(drawprob[]))
#                                                                      # *exp(j*log(theta) - theta -  logfact(j))) 
#        )#)

        n_logLike[i, j+1, k+1] <- n_poisLogLike[i, j+1, k+1]

        n_poisLogLike[i, j+1, k+1] <- 
          j*log(n_lambda_h[i]) - n_lambda_h[i] -  logfact(j) 
        + k*log(n_lambda_a[i]) - n_lambda_a[i] -  logfact(k)         
#        - lambda0 
#        + log(max(n_xbivterm[i,j+1,k+1], 0.01)        )
        + equals(j, 0)*equals(k, 0)*log(1-n_lambda_h[i]*n_lambda_a[i]*rho)
        + equals(j, 0)*equals(k, 1)*log(1+n_lambda_h[i]*rho)
        + equals(j, 1)*equals(k, 0)*log(1+n_lambda_a[i]*rho)
        + equals(j, 1)*equals(k, 1)*log(1-rho)
        
        n_xbivterm[i,j+1,k+1] <- 
        1 +  k * j * n_xqlambda[i]
        + pow(n_xqlambda[i], 2)*j*k/2*(j-1)*(k-1)
        + pow(n_xqlambda[i], 3)*j*k/3*(j-1)*(k-1)/2*(j-2)*(k-2)
        + pow(n_xqlambda[i], 4)*j*k/4*(j-1)*(k-1)/3*(j-2)*(k-2)/2*(j-3)*(k-3)
        + pow(n_xqlambda[i], 5)*j*k/5*(j-1)*(k-1)/4*(j-2)*(k-2)/3*(j-3)*(k-3)/2*(j-4)*(k-4)
        + pow(n_xqlambda[i], 6)*j*k/6*(j-1)*(k-1)/5*(j-2)*(k-2)/4*(j-3)*(k-3)/3*(j-4)*(k-4)/2*(j-5)*(k-5)
      } 
    } 
  }
  for(i in 1:length(hometeam)) {
    n_xqlambda[i] <- lambda0/(n_lambda_h[i]*n_lambda_a[i])
    n_lambda_h[i]<-attack*homeadv*defense
    n_lambda_a[i]<-defense*attack
    
    # p.HG[i] ~ dpois(attack[hometeam[i]]*homeadv[hometeam[i]]*defense[awayteam[i]])
    # p.AG[i] ~ dpois(defense[hometeam[i]]*attack[awayteam[i]])
  }
  for(i in 1:length(team)) {
    zeros[i] ~ dpois(phi[i])
    phi[i] <- -logLike[i] + 2
    
#    logLike[i] <- log( 
#    (1-isDraw[i])*  (1-pi)* exp(poisLogLike[i]) + 
#    isDraw[i] *( (1-pi)*exp(poisLogLike[i]) + pi * drawprob[goals[i]+1]/sum(drawprob[]) ) # * exp(goals[i]*log(theta) - theta -  logfact(goals[i]))) 
#    )
    
    logLike[i] <- poisLogLike[i]

    poisLogLike[i] <-
      goals[i]     *log(lambda_h) - lambda_h -  logfact(goals[i]) 
    + otherGoals[i]*log(lambda_a) - lambda_a -  logfact(otherGoals[i])
#    - lambda0 
#    + log(xbivterm[i])
    + equals(goals[i], 0)*equals(otherGoals[i], 0)*log(1-lambda_h*lambda_a*rho)
    + equals(goals[i], 0)*equals(otherGoals[i], 1)*log(1+lambda_h*rho)
    + equals(goals[i], 1)*equals(otherGoals[i], 0)*log(1+lambda_a*rho)
    + equals(goals[i], 1)*equals(otherGoals[i], 1)*log(1-rho)
    
    xbivterm[i] <- 
    1 
    + xcoeff[i,1]*    xqlambda 
    + xcoeff[i,2]*pow(xqlambda, 2) / 2
    + xcoeff[i,3]*pow(xqlambda, 3) / 6
    + xcoeff[i,4]*pow(xqlambda, 4) / 24
    
    xcoeff[i,4] <- (goals[i]-3) * (otherGoals[i]-3) * xcoeff[i,3]
    xcoeff[i,3] <- (goals[i]-2) * (otherGoals[i]-2) * xcoeff[i,2]
    xcoeff[i,2] <- (goals[i]-1) * (otherGoals[i]-1) * xcoeff[i,1]
    xcoeff[i,1] <- goals[i] * otherGoals[i]
    
    isDraw[i] <- equals(goals[i], otherGoals[i])
    
    #spieltag[i] ~ dunif(0,100)
    #season[i] ~ dunif(2013,2016)
    #weights[i] ~ dnorm(0,1)
  }

  xqlambda <- lambda0/(lambda_h*lambda_a)
    
  lambda_h <- attack*homeadv*defense
  lambda_a <- defense*attack
  
  attack ~ dgamma(2.5,1.8) I(0.3, )
  defense ~ dgamma(2.5,1.8) I(0.3, )
  homeadv ~ dnorm(1.5,1) I(0.7, )

  rho <- 0
  # rho ~ dunif(-1,1)

  # pi ~ dunif(0,0.2)
  pi <- 0
  theta ~ dunif(0,5)
  lambda0 ~ dunif(-1, 1)

  drawprob[1] ~ dunif(0, 1)
  drawprob[2] ~ dunif(0, 1)
  drawprob[3] ~ dunif(0, 1)
  drawprob[4] ~ dunif(0, 1)
  drawprob[5] ~ dunif(0, 1)
  drawprob[6] ~ dunif(0, 1)
  drawprob[7] ~ dunif(0, 1)

  #pi<-0
  #theta<-1
  #lambda0 <- -0.1
}"

  
  
  
  
  
  
nextmatches<-list(team=NA, otherTeam=NA)
teams<-levels(ld$games$HomeTeam)
outputGames<-list(hometeam=match(nextmatches$team, teams), awayteam=match(nextmatches$otherTeam, teams))
#,
#                  p.HG=rep(NA, nrow(nextmatches)), p.AG=rep(NA, nrow(nextmatches)))
inputData<-ld$teamresults[ld$teamresults$where=='Home',c('goals', 'otherGoals')]
inputData$team<-match(ld$teamresults[ld$teamresults$where=='Home','team'], teams)
inputData$otherTeam<-match(ld$teamresults[ld$teamresults$where=='Home','otherTeam'], teams)
  
model_string <- "
data  {
  for(i in 1:length(team)) 
  {
    zeros[i] <- 0
  }
  resid_sum_zero <- 0
}

model  {
  for(i in 1:length(hometeam)) {
    for (j in 0:6) {
      for (k in 0:6) {
        pScore[i, j+1, k+1]<-exp(n_logLike[i, j+1, k+1]) 
            # * (1+residual_p[j+1,k+1])
        
        n_logLike[i, j+1, k+1] <- n_poisLogLike[i, j+1, k+1]
        
        n_poisLogLike[i, j+1, k+1] <- 
        j*log(n_lambda_h[i]) - n_lambda_h[i] -  logfact(j) 
        + k*log(n_lambda_a[i]) - n_lambda_a[i] -  logfact(k)         
        + log(alpha[j+1,k+1])
        - log(sum(p1))
      } 
    } 
  }
  for(i in 1:length(hometeam)) {
    n_lambda_h[i]<-attack*homeadv*defense
    n_lambda_a[i]<-defense*attack
  }
  for(i in 1:length(team)) {
    zeros[i] ~ dpois(phi[i])
    phi[i] <- -logLike[i] + 20
    
    logLike[i] <- poisLogLike[i]
    
    poisLogLike[i] <-
    goals[i]     *log(lambda_h) - lambda_h -  logfact(goals[i]) 
    + otherGoals[i]*log(lambda_a) - lambda_a -  logfact(otherGoals[i])
    + log(alpha[goals[i]+1, otherGoals[i]+1])
    - log(sum(p1))

 #   +log(1 + residual_p[goals[i]+1, otherGoals[i]+1])
  }
  p1sum <- sum(p1[,])
  for (j in 0:6) {
    for (k in 0:6) {
      p1[j+1,k+1] <- exp(
        j*log(lambda_h) - lambda_h -  logfact(j) 
        + k*log(lambda_a) - lambda_a -  logfact(k)         
        + log(alpha[j+1,k+1])
      )
    }
  }

  lambda_h <- attack*homeadv*defense
  lambda_a <- defense*attack
  
  attack ~ dgamma(2.5,1.8) I(0.3, )
  defense ~ dgamma(2.5,1.8) I(0.3, )
  homeadv ~ dnorm(1.5,1) I(0.7, )
  
  
##  resid_sum_zero ~ dpois(resid_phi)
##  resid_phi <- exp(pow(resid_sum, 6) + sum(pow(residual_p[,], 6))) * length(team) * 100 # -log(resid_sum) + 10
#  resid_sum <- sum(residual_p[,])

  for (j in 0:6) {
    for (k in 0:6) {
      alpha[j+1,k+1] ~ dunif(0,2)
    }
  }
#  for (j in 0:6) {
#    for (k in 0:6) {
#      residual_p[j+1,k+1] <- residual_raw[j+1,k+1] / sumrd - (1/49)
#    }
#  }
#  sumrd<-sum(residual_raw[,])
#  for (j in 0:6) {
#    for (k in 0:6) {
#      residual_raw[j+1,k+1] ~ dunif(0,1) 
#    }
#  }

}"
  
  
  
cat_inputData<-inputData
#cat_inputData$goals<-min(cat_inputData$goals+1, 4)
#cat_inputData$otherGoals<-min(cat_inputData$otherGoals+1, 4)

model <- jags.model(textConnection(model_string), 
            data = c(as.list(cat_inputData), outputGames),
            n.chains = 3, n.adapt= 100 )
adapt(model, 100)
update(model, 100); # Burnin for 10000 samples
# mcmc_samples <- coda.samples(model, variable.names=c("lambda0", "pi","theta", "attack", "defense", "homeadv", "drawprob"), n.iter=200)
# mcmc_samples <- coda.samples(model, variable.names=c("lambda0", "pi","theta", "attack", "defense", "homeadv", "pScore", "drawprob"), n.iter=1000)
# mcmc_samples <- coda.samples(model, variable.names=c("lambda0", "pi","theta", "attack", "defense", "homeadv", "pScore", "rho"), n.iter=1000)
mcmc_samples <- coda.samples(model, variable.names=c("p1sum", "attack", "defense", "homeadv", "pScore", "alpha", "p1"), n.iter=1000)

summary(mcmc_samples)
plot(mcmc_samples)

varidx<-grep(pattern = "pScore\\[", x = varnames(mcmc_samples))
predmatrix<-matrix(data = summary(mcmc_samples[,varidx])$statistics[,1]*100, ncol=7, nrow=7, dimnames = list(0:6, 0:6)); predmatrix

table(ld$games[,c('FTHG', 'FTAG')])
act<-ld$games[,c('FTHG', 'FTAG')]
act$FTHG<-sapply(act$FTHG, min, 6)
act$FTAG<-sapply(act$FTAG, min, 6)
act_p<-c(table(act)/nrow(act))
actmatrix<-matrix(data = act_p*100, ncol=7, nrow=7, dimnames = list(0:6, 0:6)); actmatrix

extractProbs<-function(m) {
  data.frame(
    H=sum(lower.tri(m)*m),
    D=sum(diag(m)),
    A=sum(upper.tri(m)*m),
    GMAX0 = m[1,1],
    GMAX1 = sum(m[1:2,1:2]),
    GMAX2 = sum(m[1:3,1:3]),
    GMAX3 = sum(m[1:4,1:4]),
    H1 = m[2,1]+m[3,2]+m[4,3]+m[5,4]+m[6,5]+m[7,6],
    H2 = m[3,1]+m[4,2]+m[5,3]+m[6,4]+m[7,5],
    H3 = m[4,1]+m[5,2]+m[6,3]+m[7,4],
    A1 = m[1,2]+m[2,3]+m[3,4]+m[4,5]+m[5,6]+m[6,7],
    A2 = m[1,3]+m[2,4]+m[3,5]+m[4,6]+m[5,7],
    A3 = m[1,4]+m[2,5]+m[3,6]+m[4,7]
  )
}
rbind(actual=extractProbs(actmatrix), pred=extractProbs(predmatrix))



varidx_gap<-grep(pattern = "alpha\\[", x = varnames(mcmc_samples))
gapmatrix<-matrix(data = summary(mcmc_samples[,varidx_gap])$statistics[,1]*100, ncol=7, nrow=7, dimnames = list(0:6, 0:6)); gapmatrix


data.frame(H=sum(lower.tri(actmatrix)*actmatrix),
           D=sum(diag(actmatrix)),
           A=sum(upper.tri(actmatrix)*actmatrix),
           GMAX0 = actmatrix[1,1],
           GMAX1 = actmatrix[1,1]+actmatrix[2,1]+actmatrix[1,2]+actmatrix[2,2],
           GMAX2 = actmatrix[1,1]+actmatrix[2,1]+actmatrix[1,2]+actmatrix[2,2]+actmatrix[2,3]+actmatrix[3,2]+actmatrix[3,3],
           H1 = actmatrix[2,1]+actmatrix[3,2]+actmatrix[4,3]+actmatrix[5,4]+actmatrix[6,5]+actmatrix[7,6],
           H2 = actmatrix[3,1]+actmatrix[4,2]+actmatrix[5,3]+actmatrix[6,4]+actmatrix[7,5],
           A1 = actmatrix[1,2]+actmatrix[2,3]+actmatrix[3,4]+actmatrix[4,5]+actmatrix[5,6]+actmatrix[6,7],
           A2 = actmatrix[1,3]+actmatrix[2,4]+actmatrix[3,5]+actmatrix[4,6]+actmatrix[5,7]
)

data.frame(H=sum(lower.tri(predmatrix)*predmatrix),
           D=sum(diag(predmatrix)),
           A=sum(upper.tri(predmatrix)*predmatrix))
sum(predmatrix)
predmatrix[1,1]+predmatrix[2,1]+predmatrix[1,2]+predmatrix[2,2]
predmatrix[1,1]+predmatrix[2,1]+predmatrix[1,2]+predmatrix[2,2]+predmatrix[2,3]+predmatrix[3,2]+predmatrix[3,3]


compare<-data.frame(act=act_p*100, pred=100.0*summary(mcmc_samples[,varidx])$statistics[,1])
compare$r <- compare$pred / compare$act
compare$score <- expand.grid(0:6, 0:6)
compare

sum(compare$pred)
sum(compare$act)

plot(mcmc_samples[,c("pi", "rho")])
plot(mcmc_samples[,c("attack", "defense", "homeadv")])
plot(mcmc_samples[,c("p1sum", "pScore[1,1,1]", "alpha[1,1]")])
plot(mcmc_samples[,c("p1[2,2]", "pScore[1,2,2]", "alpha[2,2]")])
plot(mcmc_samples[,c("p1[4,4]", "pScore[1,4,4]", "alpha[4,4]")])
plot(mcmc_samples[,c("p1[2,3]", "pScore[1,2,3]", "alpha[2,3]")])


plot(mcmc_samples[,"drawprob[1]"])
plot(mcmc_samples[,"drawprob[2]"])
plot(mcmc_samples[,"drawprob[3]"])
plot(mcmc_samples[,"drawprob[4]"])
plot(mcmc_samples[,"drawprob[5]"])
plot(mcmc_samples[,"drawprob[6]"])
plot(mcmc_samples[,"drawprob[7]"])
plot(mcmc_samples[,c("drawprob[1]", "drawprob[2]", "drawprob[3]")])
plot(mcmc_samples[,c("drawprob[1]", "drawprob[2]")])
summary(mcmc_samples[,c("pi", "drawprob[1]", "drawprob[2]", "drawprob[3]", "drawprob[4]", "drawprob[5]", "drawprob[6]")])

plot(mcmc_samples[,"pScore[1,1,1]"])
plot(mcmc_samples[,"pScore[1,2,1]"])
plot(mcmc_samples[,"pScore[1,3,1]"])
plot(mcmc_samples[,"pScore[1,2,2]"])
plot(mcmc_samples[,"pScore[1,1,2]"])
plot(mcmc_samples[,"pScore[1,4,6]"])
plot(mcmc_samples[,c("pScore[1,1,1]", "pScore[1,2,2]", "pScore[1,1,2]")])



mean(inputData$goals)
var(inputData$goals)

mean(inputData$otherGoals)
var(inputData$otherGoals)

mean(inputData$goals+inputData$otherGoals)
var(inputData$goals+inputData$otherGoals)


mean(ld$games$FTHG)
var(ld$games$FTHG)

mean(ld$games$FTAG)
var(ld$games$FTAG)

mean(ld$games$FTHG+ld$games$FTAG)
var(ld$games$FTHG+ld$games$FTAG)


gr<-expand.grid(0:6, 0:6, 0:6)
cbind(gr, choose(gr$Var2, gr$Var1)*choose(gr$Var3, gr$Var1)*factorial(gr$Var1))

summary(mcmc_samples[,c("pScore[1,2,2]")])


"

n_xbivterm1[i,1,1] <- 1
n_xbivterm1[i,2,1] <- 1
n_xbivterm1[i,3,1] <- 1
n_xbivterm1[i,4,1] <- 1
n_xbivterm1[i,5,1] <- 1
n_xbivterm1[i,6,1] <- 1
n_xbivterm1[i,7,1] <- 1
n_xbivterm1[i,1,2] <- 1 
n_xbivterm1[i,1,3] <- 1 
n_xbivterm1[i,1,4] <- 1 
n_xbivterm1[i,1,5] <- 1 
n_xbivterm1[i,1,6] <- 1 
n_xbivterm1[i,1,7] <- 1 
n_xbivterm1[i,2,2] <- 1 + n_xqlambda[i]
n_xbivterm1[i,2,3] <- 1 + 2*n_xqlambda[i]
n_xbivterm1[i,2,4] <- 1 + 3*n_xqlambda[i]
n_xbivterm1[i,2,5] <- 1 + 4*n_xqlambda[i]
n_xbivterm1[i,2,6] <- 1 + 5*n_xqlambda[i]
n_xbivterm1[i,2,7] <- 1 + 6*n_xqlambda[i]
n_xbivterm1[i,3,2] <- 1 + 2*n_xqlambda[i]
n_xbivterm1[i,4,2] <- 1 + 3*n_xqlambda[i]
n_xbivterm1[i,5,2] <- 1 + 4*n_xqlambda[i]
n_xbivterm1[i,6,2] <- 1 + 5*n_xqlambda[i]
n_xbivterm1[i,7,2] <- 1 + 6*n_xqlambda[i]
n_xbivterm1[i,3,3] <- 1 + 4*n_xqlambda[i] + 2*pow(n_xqlambda[i], 2)
n_xbivterm1[i,4,3] <- 1 + 6*n_xqlambda[i] + 6*pow(n_xqlambda[i], 2)
n_xbivterm1[i,3,4] <- 1 + 6*n_xqlambda[i] + 6*pow(n_xqlambda[i], 2)
n_xbivterm1[i,5,3] <- 1 + 8*n_xqlambda[i] + 12*pow(n_xqlambda[i], 2)
n_xbivterm1[i,3,5] <- 1 + 8*n_xqlambda[i] + 12*pow(n_xqlambda[i], 2)
n_xbivterm1[i,6,3] <- 1 + 10*n_xqlambda[i] + 20*pow(n_xqlambda[i], 2)
n_xbivterm1[i,3,6] <- 1 + 10*n_xqlambda[i] + 20*pow(n_xqlambda[i], 2)
n_xbivterm1[i,7,3] <- 1 + 12*n_xqlambda[i] + 30*pow(n_xqlambda[i], 2)
n_xbivterm1[i,3,7] <- 1 + 12*n_xqlambda[i] + 30*pow(n_xqlambda[i], 2)
n_xbivterm1[i,4,4] <- 1 + 9*n_xqlambda[i] + 18*pow(n_xqlambda[i], 2) + 6*pow(n_xqlambda[i], 3)
n_xbivterm1[i,5,4] <- 1 + 12*n_xqlambda[i] + 36*pow(n_xqlambda[i], 2) + 24*pow(n_xqlambda[i], 3)
n_xbivterm1[i,4,5] <- 1 + 12*n_xqlambda[i] + 36*pow(n_xqlambda[i], 2) + 24*pow(n_xqlambda[i], 3)
n_xbivterm1[i,6,4] <- 1 + 15*n_xqlambda[i] + 60*pow(n_xqlambda[i], 2) + 60*pow(n_xqlambda[i], 3)
n_xbivterm1[i,4,6] <- 1 + 15*n_xqlambda[i] + 60*pow(n_xqlambda[i], 2) + 60*pow(n_xqlambda[i], 3)
n_xbivterm1[i,7,4] <- 1 + 18*n_xqlambda[i] + 90*pow(n_xqlambda[i], 2) + 120*pow(n_xqlambda[i], 3)
n_xbivterm1[i,4,7] <- 1 + 18*n_xqlambda[i] + 90*pow(n_xqlambda[i], 2) + 120*pow(n_xqlambda[i], 3)
n_xbivterm1[i,5,5] <- 1 + 16*n_xqlambda[i] + 72*pow(n_xqlambda[i], 2) + 96*pow(n_xqlambda[i], 3) + 24*pow(n_xqlambda[i], 4)
n_xbivterm1[i,5,6] <- 1 + 20*n_xqlambda[i] + 120*pow(n_xqlambda[i], 2) + 240*pow(n_xqlambda[i], 3) + 120*pow(n_xqlambda[i], 4)
n_xbivterm1[i,6,5] <- 1 + 20*n_xqlambda[i] + 120*pow(n_xqlambda[i], 2) + 240*pow(n_xqlambda[i], 3) + 120*pow(n_xqlambda[i], 4)
n_xbivterm1[i,5,7] <- 1 + 24*n_xqlambda[i] + 180*pow(n_xqlambda[i], 2) + 480*pow(n_xqlambda[i], 3) + 360*pow(n_xqlambda[i], 4)
n_xbivterm1[i,7,5] <- 1 + 24*n_xqlambda[i] + 180*pow(n_xqlambda[i], 2) + 480*pow(n_xqlambda[i], 3) + 360*pow(n_xqlambda[i], 4)
n_xbivterm1[i,6,6] <- 1 + 25*n_xqlambda[i] + 200*pow(n_xqlambda[i], 2) + 600*pow(n_xqlambda[i], 3) + 600*pow(n_xqlambda[i], 4) + 120*pow(n_xqlambda[i], 5)
n_xbivterm1[i,7,6] <- 1 + 30*n_xqlambda[i]*(1 + n_xqlambda[i]*(10 + 4*n_xqlambda[i]*(10 + 6*n_xqlambda[i] + 2.4*pow(n_xqlambda[i], 2))))
n_xbivterm1[i,6,7] <- 1 + 30*n_xqlambda[i] + 300*pow(n_xqlambda[i], 2) + 1200*pow(n_xqlambda[i], 3) + 1800*pow(n_xqlambda[i], 4) + 720*pow(n_xqlambda[i], 5)
n_xbivterm1[i,7,7] <- 1 + 36*n_xqlambda[i] + 450*pow(n_xqlambda[i], 2) + 2400*pow(n_xqlambda[i], 3) + 5400*pow(n_xqlambda[i], 4) + 4320*pow(n_xqlambda[i], 5) + 720*pow(n_xqlambda[i], 6)


"