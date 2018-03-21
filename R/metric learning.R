download.file("http://www.football-data.co.uk/mmz4281/1617/D1.csv", "BL2016.csv")
download.file("http://www.football-data.co.uk/mmz4281/1516/D1.csv", "BL2015.csv")
download.file("http://www.football-data.co.uk/mmz4281/1415/D1.csv", "BL2014.csv")

library(dplyr)
require(expm)

data1<-read.csv("BL2016.csv")
data1$season<-2016
data2<-read.csv("BL2015.csv")
data2$season<-2015
data3<-read.csv("BL2014.csv")
data3$season<-2014
data<-rbind(data3[,colnames(data1)], data2, data1)
teams <- unique(data$HomeTeam)
results <- data[,c('HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'season', 'Date', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',  'HR',  'AR' )]
results$season<-as.factor(results$season)
table(results$FTR , results$season)
results$spieltag <- floor((9:(nrow(results)+8))/9)
results$round <- ((results$spieltag-1) %% 34) +1
results$Date<-as.Date(results$Date, "%d/%m/%y")
results$dayofweek<-weekdays(results$Date)


teamresults <- data.frame(team=results$HomeTeam, otherTeam=results$AwayTeam, 
                          goals=results$HTHG, otherGoals=results$HTAG, where="Home", spieltag=results$spieltag, season=results$season)
teamresults <- rbind(data.frame(team=results$AwayTeam, otherTeam=results$HomeTeam, 
                                goals=results$HTAG, otherGoals=results$HTHG, where="Away", spieltag=results$spieltag, season=results$season),
                     teamresults)

teamresults <- data.frame(team=results$HomeTeam, otherTeam=results$AwayTeam, 
                          goals=results$FTHG, otherGoals=results$FTAG, where="Home", round=results$round, season=results$season)
teamresults <- rbind(data.frame(team=results$AwayTeam, otherTeam=results$HomeTeam, 
                                goals=results$FTAG, otherGoals=results$FTHG, where="Away", round=results$round, season=results$season),
                     teamresults)

# teamresults[teamresults$team=="Bayern Munich" & teamresults$season==2016,]
teamresults<-teamresults[order(teamresults$season, teamresults$round, teamresults$team),]

teamresults$points<-ifelse(
  sign(teamresults$goals - teamresults$otherGoals)==1, 3, 
  ifelse(teamresults$goals == teamresults$otherGoals, 1, 0))

teamresults$otherPoints<-ifelse(
  sign(teamresults$goals - teamresults$otherGoals)==1, 0, 
  ifelse(teamresults$goals == teamresults$otherGoals, 1, 3))

# ranking<-aggregate(points ~ team+season, teamresults, cumsum)
# 
# teampoints<-lapply(1:nrow(ranking), function(i) expand.grid(team=ranking[i,1], season=ranking[i,2], points=ranking[i,3][[1]]))
# tp<-data.frame()
# for (i in 1:length((teampoints))) {tp<-rbind(tp, teampoints[[i]])}
# 
# data.frame(tp$team, tp$season, tp$points)
# cn<-colnames(tp)
# attributes(tp)<-NULL
# names(tp)<-cn
# str(tp)
# data.frame(tp)
# 
# str(tp)
# 
# 
# teamresults$tablepoints<-sapply(1:nrow(teamresults), function(i) {
#   ranking[ranking$team==teamresults[i, 'team'] & ranking$season==teamresults[i, 'season'],'points'][[1]][teamresults[i, 'round']-1][1]  })
# teamresults$tablepoints[is.na(teamresults$tablepoints)]<-0
# teamresults$othertablepoints<-sapply(1:nrow(teamresults), function(i) {
#   ranking[ranking$team==teamresults[i, 'otherTeam'] & ranking$season==teamresults[i, 'season'],'points'][[1]][teamresults[i, 'round']-1][1]  })
# teamresults$othertablepoints[is.na(teamresults$othertablepoints)]<-0
# 
# results$HP<-sapply(1:nrow(results), function(i) {
#   ranking[ranking$team==results[i, 'HomeTeam'] & ranking$season==results[i, 'season'],'points'][[1]][results[i, 'round']-1][1]  })
# results$HP[is.na(results$HP)]<-0
# results$AP<-sapply(1:nrow(results), function(i) {
#   ranking[ranking$team==results[i, 'AwayTeam'] & ranking$season==results[i, 'season'],'points'][[1]][results[i, 'round']-1][1]  })
# results$AP[is.na(results$AP)]<-0
# 


teamresults<-
  teamresults %>%
  group_by(season, team) %>%
  arrange(round) %>%
  mutate(tablepoints = cumsum(points)-points) %>%
  mutate(scoredgoals = cumsum(goals)-goals) %>%
  mutate(receivedgoals = cumsum(otherGoals)-otherGoals) %>%
  group_by(season, otherTeam) %>%
  arrange(round) %>%
  mutate(othertablepoints = cumsum(otherPoints)-otherPoints) %>% 
  mutate(otherscoredgoals = cumsum(otherGoals)-otherGoals) %>%
  mutate(otherreceivedgoals = cumsum(goals)-goals) %>%
  ungroup()

teamresults<-
  teamresults %>%
  group_by(season, round) %>%
  mutate(rank = rank(-tablepoints, ties.method="min"),
         otherrank = rank(-othertablepoints, ties.method="min")) %>% 
  arrange(season, round, team) %>%
  ungroup() %>%  
  mutate(rankdiff = -rank+otherrank,
         pointdiff = tablepoints-othertablepoints,
         goaldiff = scoredgoals - receivedgoals,
         othergoaldiff = otherscoredgoals - otherreceivedgoals
  )
  
print.data.frame(tail(teamresults, 30))

m.team<-glm(formula = goals ~ (team+otherTeam)+where+tablepoints+rank+season+round+pointdiff+rankdiff, data=teamresults, family = poisson)
summary(m.team)

ranking <- 
  teamresults %>% 
  group_by(season, round, team) %>% 
  select (tablepoints, rank, scoredgoals, receivedgoals) %>% 
  ungroup()

results <-
  results %>% 
  select(-HTR, -FTR) %>%
  rename(team=HomeTeam) %>% left_join(ranking, by = c("season", "round", "team")) %>% rename(HomeTeam=team, HRank=rank, HP=tablepoints, HSG=scoredgoals, HRG=receivedgoals)%>% 
  rename(team=AwayTeam) %>% left_join(ranking, by = c("season", "round", "team")) %>% rename(AwayTeam=team, ARank=rank, AP=tablepoints, ASG=scoredgoals, ARG=receivedgoals)%>%
  mutate(PointDiff=HP-AP, RankDiff=ARank-HRank,  FTR=sign(FTHG-FTAG), HTR=sign(HTHG-HTAG))


outputs <- results %>%
  select(FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR )

inputs <- results %>%
  select(HomeTeam, AwayTeam, season, round, dayofweek, HP, AP  , PointDiff, HRank, ARank ,  RankDiff, HSG, HRG, ASG, ARG)

x2<-scale(outputs, scale=FALSE, center = TRUE)
str(x2)
summary(x2)
sigma<-cov(x2)
sigma_inv<-solve(sigma)
sigma_inv_sqrt<-sqrtm(sigma_inv)
white<-t(sigma_inv_sqrt %*% t(x2))
cov(white)
colnames(white)<-colnames(outputs)



km<-kmeans(white, centers = 10)

km<-kmeans(x3, centers = 10)


outputs2 <- results %>%
  select(FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR )%>%
  mutate(FTHG2 = FTHG-HTHG, FTAG2 = FTAG-HTAG, FTGDiff=FTHG-FTAG, HT2GDiff=FTHG2-FTAG2, HTGDiff=HTHG-HTAG) 

factanal(x = outputs, factors=9)

factanal(x = white, factors=5)

cov(outputs2)


library(RANN)
x3<-x2
x3[,1]<-outputs[,1]*3
x3[,2]<-outputs[,2]*3
x3[,3]<-outputs[,3]*5
nn<-nn2(x3)
nn<-nn2(white)
summary(nn$nn.dists[, 2:10])

rbind(outputs[nn$nn.idx[,1],], outputs[nn$nn.idx[,2],], outputs[nn$nn.idx[,3],])[ c(0, nrow(outputs), 2*nrow(outputs))+rep(1:nrow(outputs), each=3), ]

matrix(ncol=10, data=outputs[nn$nn.idx[,1:10],], byrow = FALSE)

matrix(ncol=10, data=km$cluster[nn$nn.idx[,1:10]], byrow = FALSE)
km$cluster[nn$nn.idx[1,1:10]]
nn$nn.idx[2,1:10]


table(outputs$FTHG,outputs$FTAG, km$cluster)

sum(km$withinss)+km$betweenss
km
km$size

km$centers
summary(dist(km$centers))
boxplot(dist(km$centers))
hist(dist(km$centers))

shift <- attr(x2,"scaled:center")
t(t(x2)+shift)

unwhiten<-function(x) t(sigma %*% sigma_inv_sqrt %*% t(x)+shift)
unwhiten(white)

cmeans<-t(t(km$centers)+shift)
colnames(cmeans)<-names(shift)
cmeans
dist(cmeans)


unwhiten(km$centers)

outputs[km$cluster==10,]

neglinks <- matrix(1, ncol = length(km$size), nrow=length(km$size), byrow = TRUE)
diag(neglinks)<-0

library(dummies)
inputs_df <- dummy.data.frame(inputs, names=c("HomeTeam", "AwayTeam", "season", "dayofweek"), sep="_")

inputs_df2 <- dummy.data.frame(data = data.frame(Team=inputs[, "HomeTeam"]), names="Team", sep="_")
inputs_df3 <- dummy.data.frame(data = data.frame(Team=inputs[, "AwayTeam"]), names="Team", sep="_")

inputs_df4 <- dummy.data.frame( data.frame(inputs_df2-inputs_df3, inputs[,-c(1,2)]), names=c("season", "dayofweek"), sep="_")


str(inputs_df)

dcaData <- dca(data = inputs_df4, chunks = km$cluster, neglinks=neglinks)
dcaData <- dca(data = inputs_df, chunks = km$cluster, neglinks=neglinks)
str(dcaData)
dcaData$newData
dcamatrix<-dcaData$DCA
colnames(dcamatrix)<-colnames(inputs_df)
dcamatrix

distmatrix<-as.matrix(dist(dcaData$newData))
str(distmatrix)
hist(distmatrix, prob = T)

hist(1/distmatrix)
hist(exp(-distmatrix))
summary(exp(-distmatrix))

plot(sort(exp(-distmatrix[1,2:909])))

i<-302
plot(sort(exp(-distmatrix[i,-i]^4.5)))
s1<-aggregate(exp(-distmatrix[i,-i]^1.5), by=list(outputs$FTHG[-i], outputs$FTAG[-i]), sum)
s1$x<-s1$x/sum(s1$x)*100
s1
paste(outputs$FTHG[i], outputs$FTAG[i])


library(class)
knnclass <- knn(train = dcaData$newData, test = dcaData$newData, km$cluster, k = 10, l = 0, prob = TRUE, use.all = TRUE)
table(knnclass, km$cluster)

summary(attr(knnclass, "prob"))

library(RANN)
nn<-nn2(dcaData$newData)
summary(nn$nn.dists[,2:10])

matrix(ncol=10, data=km$cluster[nn$nn.idx[,1:10]], byrow = FALSE)
km$cluster[nn$nn.idx[1,1:10]]
nn$nn.idx[2,1:10]

str(knnclass)


## rf
library(randomForest)
x<-inputs
x$dayofweek<-as.factor(x$dayofweek)
x$HPQ<-x$HP/x$round
x$APQ<-x$AP/x$round
x$HSGQ<-x$HSG/x$round
x$HRGQ<-x$HRG/x$round
x$ASGQ<-x$ASG/x$round
x$ARGQ<-x$ARG/x$round


y<-as.factor(paste(outputs$FTHG,outputs$FTAG, sep=":"))
y<-outputs$FTHG - outputs$FTAG
model_rf <- randomForest(x = x, y=y, importance=TRUE)
model_rf
summary(model_rf)
varImpPlot(model_rf)
data.frame(predict(model_rf, newdata = x), y)
cov(predict(model_rf, newdata = x), y)
cor(predict(model_rf, newdata = x), y)
plot(predict(model_rf, newdata = x), y)
qqplot(predict(model_rf, newdata = x), y)
table(round(predict(model_rf, newdata = x)), y)
table(round(predict(model_rf, newdata = x)) - y)
table(sign(round(predict(model_rf, newdata = x))), sign(y))
table(sign(round(predict(model_rf, newdata = x))) - sign(y))

nrow(x)
x
tail(x[1:801,])

train_idx<-1:801
cv_idx<-801:nrow(x)

model_rf <- randomForest(x = x[train_idx,], y=y[train_idx], importance=TRUE)
model_rf
summary(model_rf)
varImpPlot(model_rf)
x<-x[cv_idx,]
y<-y[cv_idx]
data.frame(predict(model_rf, newdata = x), y)
cov(predict(model_rf, newdata = x), y)
cor(predict(model_rf, newdata = x), y)
plot(predict(model_rf, newdata = x), y)
qqplot(predict(model_rf, newdata = x), y)
table(round(predict(model_rf, newdata = x)), y)
table(round(predict(model_rf, newdata = x)) - y)
table(sign(round(predict(model_rf, newdata = x))), sign(y))
table(sign(round(predict(model_rf, newdata = x))) - sign(y))
table(sign(round(predict(model_rf, newdata = x))) == sign(y))

sum(sign(round(predict(model_rf, newdata = x))) == sign(y))/length(y)*100 


str(inputs)
varImpPlot(model_rf)

as.factor(paste(outputs$FTHG,outputs$FTAG, sep=":"))

table(outputs$FTHG,outputs$FTAG, km$cluster)


pred<-sapply(1:nrow(outputs), function(i) 
  aggregate(exp(-distmatrix[i,]^1.5), by=list(outputs$FTHG, outputs$FTAG), sum)$x
  , simplify = TRUE)
pred<-t(pred)
str(pred)

scores<-aggregate(exp(-distmatrix[1,]), by=list(outputs$FTHG, outputs$FTAG), sum)[,1:2]
ownscore<-t(sapply(1:nrow(outputs), function(i) outputs$FTHG[i]==scores[,1] & outputs$FTAG[i]==scores[,2]))
pred2<-pred-ownscore
pred3<-pred2/rowSums(pred2)
colnames(pred3)<-paste(scores[,1], scores[,2])
pred3

summary(pred3[ownscore])

scores_0<-table(outputs$FTHG, outputs$FTAG)/nrow(outputs)

data.frame(meanhitprob=mean(pred3[ownscore]), standardhitprob=sum(scores_0^2))

#####################

pred<-sapply(1:nrow(outputs), function(i) 
  aggregate(exp(-distmatrix[i,]^1.5), by=list(outputs$FTR), sum)$x
  , simplify = TRUE)
pred<-t(pred)
str(pred)

scores<-aggregate(exp(-distmatrix[1,]), by=list(outputs$FTR), sum)[,1:2]
ownscore<-t(sapply(1:nrow(outputs), function(i) outputs$FTR[i]==scores[,1]))
pred2<-pred-ownscore
pred3<-pred2/rowSums(pred2)
colnames(pred3)<-paste(scores[,1])
pred3

summary(pred3[ownscore])

scores_0<-table(outputs$FTR)/nrow(outputs)

data.frame(meanhitprob=mean(pred3[ownscore]), standardhitprob=sum(scores_0^2))

#####################

pred<-sapply(1:nrow(outputs), function(i) 
  aggregate(exp(-distmatrix[i,]^4.5), by=list(outputs$FTHG-outputs$FTAG), sum)$x
  , simplify = TRUE)
pred<-t(pred)
str(pred)

scores<-aggregate(exp(-distmatrix[1,]), by=list(outputs$FTHG-outputs$FTAG), sum)[,1:2]
ownscore<-t(sapply(1:nrow(outputs), function(i) outputs$FTHG[i]-outputs$FTAG[i]==scores[,1]))
pred2<-pred-ownscore
pred3<-pred2/rowSums(pred2)
colnames(pred3)<-paste(scores[,1])
pred3

summary(pred3[ownscore])

scores_0<-table(outputs$FTHG-outputs$FTAG)/nrow(outputs)

data.frame(meanhitprob=mean(pred3[ownscore]), standardhitprob=sum(scores_0^2))

#####################






paste(scores[,1], scores[,2])

pred[2,]
pred2[2,]
scores[ownscore[2,],]

  aggregate(exp(-distmatrix[i,]), by=list(outputs$FTHG, outputs$FTAG), sum)$x
  , simplify = TRUE)


pred2<-matrix(nrow=nrow(outputs), byrow = TRUE, data = unlist(pred))

sapply(pred, length)

str(pred[[2]])

unlist(pred, recursive = FALSE)

table(outputs$FTHG, outputs$FTAG)

table(paste(outputs$FTHG, outputs$FTAG))


dist(km$centers)
dist(unwhiten(km$centers)[,"FTR"])

str(df)

inputs

require(mclust)
mcl<-Mclust(scale((outputs), scale=FALSE))
summary(mcl)
plot(mcl)


mahalanobis()


require(graphics)

ma <- cbind(1:6, 1:3)
(S <-  var(ma))
mahalanobis(c(0, 0), 1:2, S)

x <- matrix(rnorm(100*3), ncol = 3)
stopifnot(mahalanobis(x, 0, diag(ncol(x))) == rowSums(x*x))
##- Here, D^2 = usual squared Euclidean distances

Sx <- cov(x)
D2 <- mahalanobis(x, colMeans(x), Sx)
plot(density(D2, bw = 0.5),
     main="Squared Mahalanobis distances, n=100, p=3") ; rug(D2)
qqplot(qchisq(ppoints(100), df = 3), D2,
       main = expression("Q-Q plot of Mahalanobis" * ~D^2 *
                           " vs. quantiles of" * ~ chi[3]^2))
abline(0, 1, col = 'gray')




outputs %>% mutate(cluster=km$cluster) %>% arrange(cluster) %>% 
  group_by(cluster) %>% 
  select(table(FTHG,FTAG))
  print.data.frame()


X <- matrix(nrow = 2, ncol=5, data=rnorm(10))
t(X)%*%X
plot(X)
X%*%t(X)
i<-1
j<-5
e <- matrix(nrow = 5, ncol=1, data=c(1,0,0,0,-1))
t(e) %*% t(X) %*%X %*% e
t(X[,1]-X[,5]) %*% (X[,1]-X[,5])

require (dml)

## Not run:
set.seed(123)
require(MASS) # generate synthetic Gaussian data
k = 100 # sample size of each class
n = 3 # specify how many class
N = k * n # total sample number
x1 = mvrnorm(k, mu = c(-2, 1), matrix(c(10, 4, 4, 10), ncol = 2))
x2 = mvrnorm(k, mu = c(0, 0), matrix(c(10, 4, 4, 10), ncol = 2))
x3 = mvrnorm(k, mu = c(2, -1), matrix(c(10, 4, 4, 10), ncol = 2))
data = as.data.frame(rbind(x1, x2, x3))
# The fully labeled data set with 3 classes
plot(data$V1, data$V2, bg = c("#E41A1C", "#377EB8", "#4DAF4A")[gl(n, k)],
     pch = c(rep(22, k), rep(21, k), rep(25, k)))
Sys.sleep(3)
# Same data unlabeled; clearly the classes' structure is less evident
plot(data$V1, data$V2)
Sys.sleep(3)
chunk1 = sample(1:100, 5)
chunk2 = sample(setdiff(1:100, chunk1), 5)
chunk3 = sample(101:200, 5)
chunk4 = sample(setdiff(101:200, chunk3), 5)
chunk5 = sample(201:300, 5)
chks = list(chunk1, chunk2, chunk3, chunk4, chunk5)
chunks = rep(-1, 300)
# positive samples in the chunks
for (i in 1:5) {
  for (j in chks[[i]]) {
    chunks[j] = i
  }
}
# define the negative constrains between chunks
neglinks = matrix(c(
  0, 0, 1, 1, 1,
  0, 0, 1, 1, 1,
  1, 1, 0, 0, 0,
  1, 1, 0, 0, 1,
  1, 1, 1, 1, 0),
  ncol = 5, byrow = TRUE)
# 4 GdmDiag
dcaData = dca(data = data, chunks = chunks, neglinks = neglinks)$newData
# plot DCA transformed data
plot(dcaData[, 1], dcaData[, 2], bg = c("#E41A1C", "#377EB8", "#4DAF4A")[gl(n, k)],
     pch = c(rep(22, k), rep(21, k), rep(25, k))
     )
#,     xlim = c(-15, 15), ylim = c(-15, 15))
## End(Not run)








## Not run:
set.seed(602)
library(MASS)
library(scatterplot3d)
# generate simulated Gaussian data
k = 100
m <- matrix(c(1, 0.5, 1, 0.5, 2, -1, 1, -1, 3), nrow =3, byrow = T)
x1 <- mvrnorm(k, mu = c(1, 1, 1), Sigma = m)
x2 <- mvrnorm(k, mu = c(-1, 0, 0), Sigma = m)
data <- rbind(x1, x2)
# define similar constrains
simi <- rbind(t(combn(1:k, 2)), t(combn((k+1):(2*k), 2)))
temp <- as.data.frame(t(simi))
tol <- as.data.frame(combn(1:(2*k), 2))
# define disimilar constrains
dism <- t(as.matrix(tol[!tol %in% simi]))
# transform data using GdmDiag
result <- GdmDiag(data, simi, dism)
newData <- result$newData
# plot original data
color <- gl(2, k, labels = c("red", "blue"))
par(mfrow = c(2, 1), mar = rep(0, 4) + 0.1)
scatterplot3d(data, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Original Data")
# plot GdmDiag transformed data
scatterplot3d(newData, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Transformed Data")
## End(Not run)



## Not run:
set.seed(123)
library(MASS)
library(scatterplot3d)
# generate simulated Gaussian data
k = 100
m <- matrix(c(1, 0.5, 1, 0.5, 2, -1, 1, -1, 3), nrow =3, byrow = T)
x1 <- mvrnorm(k, mu = c(1, 1, 1), Sigma = m)
x2 <- mvrnorm(k, mu = c(-1, 0, 0), Sigma = m)
data <- rbind(x1, x2)
# define similar constrains
simi <- rbind(t(combn(1:k, 2)), t(combn((k+1):(2*k), 2)))
temp <- as.data.frame(t(simi))
tol <- as.data.frame(combn(1:(2*k), 2))
# define disimilar constrains
dism <- t(as.matrix(tol[!tol %in% simi]))
# transform data using GdmFull
result <- GdmFull(data, simi, dism)
newData <- result$newData
# plot original data
color <- gl(2, k, labels = c("red", "blue"))
par(mfrow = c(2, 1), mar = rep(0, 4) + 0.1)
scatterplot3d(data, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Original Data")
# plot GdmFull transformed data
scatterplot3d(newData, color = color, cex.symbols = 0.6,
              xlim = range(data[, 1], newData[, 1]),
              ylim = range(data[, 2], newData[, 2]),
              zlim = range(data[, 3], newData[, 3]),
              main = "Transformed Data")
## End(Not run)



par(mfrow = c(1, 1))



set.seed(1234)
require(MASS) # generate synthetic Gaussian data
k = 100 # sample size of each class
n = 3 # specify how many class
N = k * n # total sample number
x1 = mvrnorm(k, mu = c(-10, 6), matrix(c(10, 4, 4, 10), ncol = 2))
x2 = mvrnorm(k, mu = c(0, 0), matrix(c(10, 4, 4, 10), ncol = 2))
x3 = mvrnorm(k, mu = c(10, -6), matrix(c(10, 4, 4, 10), ncol = 2))
x = as.data.frame(rbind(x1, x2, x3))
x$V3 = gl(n, k)
# The fully labeled data set with 3 classes
plot(x$V1, x$V2, bg = c("#E41A1C", "#377EB8", "#4DAF4A")[x$V3],
     pch = c(rep(22, k), rep(21, k), rep(25, k)))
Sys.sleep(3)
# Same data unlabeled; clearly the classes' structure is less evident
plot(x$V1, x$V2)
Sys.sleep(3)
chunk1 = sample(1:100, 5)
chunk2 = sample(setdiff(1:100, chunk1), 5)
chunk3 = sample(101:200, 5)
chunk4 = sample(setdiff(101:200, chunk3), 5)
chunk5 = sample(201:300, 5)
chks = x[c(chunk1, chunk2, chunk3, chunk4, chunk5), ]
chunks = list(chunk1, chunk2, chunk3, chunk4, chunk5)
# The chunklets provided to the RCA algorithm
plot(chks$V1, chks$V2, col = rep(c("#E41A1C", "#377EB8",
                                   "#4DAF4A", "#984EA3", "#FF7F00"), each = 5),
     pch = rep(0:4, each = 5), ylim = c(-15, 15))
Sys.sleep(3)
# Whitening transformation applied to the chunklets
chkTransformed = as.matrix(chks[ , 1:2]) %*% rca(x[ , 1:2], chunks)$A
plot(chkTransformed[ , 1], chkTransformed[ , 2], col = rep(c(
  "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"), each = 5),
  pch = rep(0:4, each = 5), ylim = c(-15, 15))
Sys.sleep(3)
# The origin data after applying the RCA transformation
plot(rca(x[ , 1:2], chunks)$newX[, 1], rca(x[ , 1:2], chunks)$newX[, 2],
     bg = c("#E41A1C", "#377EB8", "#4DAF4A")[gl(n, k)],
     pch = c(rep(22, k), rep(21, k), rep(25, k)))
# The RCA suggested transformation of the data, dimensionality reduced
rca(x[ , 1:2], chunks)$A
# The RCA suggested Mahalanobis matrix
rca(x[ , 1:2], chunks)$B
## End(Not run)




