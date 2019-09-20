library(glmnet)
library(glmnetUtils)
library(colorspace)

#setwd("c:/git/football/TF/data")
setwd("d:/gitrepository/Football/football/TF/data")

lfda.result <- read.csv("lfda_data.csv")
data <- read.csv("all_features.csv")
labels <- read.csv("all_labels.csv")
rawdata <- read.csv("full_data.csv")

is_new <- data$Predict=="True"
foldid<-as.integer(factor(rep(rawdata$Season, each=2)))

par(mfrow=c(5,4))
for (i in 1:20) {
  plot(lfda.result[,i], main=colnames(lfda.result[i]))
}
par(mfrow=c(1,1))

pmin_pmax_clip <- function(x, a, b) {x2<-pmax(a, pmin(x, b) ); x2-min(x2)+1}
mylabel<-pmin_pmax_clip((labels$T1_GFT-labels$T2_GFT), -3, 6) 
mylabel<-pmin_pmax_clip(sign(labels$T1_GFT-labels$T2_GFT), -1, 1) 
#colors <- heat_hcl(length(unique(mylabel)))
colors <- rainbow_hcl(length(unique(mylabel)))
plot(seq_along(colors), col=colors)
cl <- colors[mylabel]
cl[is_new] <- "black"
names(cl)<-mylabel
for (i in 1:9) {
  plot(lfda.result[,c(i, i+1)], col=cl)
  plot(lfda.result[,c(i+10, i+11)], col=cl)
}
barplot(table(mylabel), col=colors)




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

str(cvfit$fit.preval)

pp<-cvfit$fit.preval[1:6246, 1:3, 30]

table(apply(pp, 1, which.max))
mean(apply(pp, 1, which.max)== as.integer(label))
table(apply(pp[seq(1, 6246, 2),], 1, which.max))
table(apply(pp[seq(2, 6246, 2),], 1, which.max))

table(apply(pp, 1, which.max), label)
table(apply(pp[seq(1, 6246, 2),], 1, which.max), label[seq(1, 6246, 2)])
table(apply(pp[seq(2, 6246, 2),], 1, which.max), label[seq(2, 6246, 2)])


train_idx<-seq(2448, 6191)
test_idx<-seq(46, 2447)
pred_idx<-seq(6192, 6245)

model <- glmnet(label~., data=cbind(lfda.result, label=label)[train_idx,], family = family, alpha=1.0)
plot(model, label = TRUE )
plot(model, xvar = "dev")
plot(model, xvar = "lambda")

s <- cvfit$lambda.min
s<- exp(-4)
idx<-test_idx
table(apply(predict(model, s=s, newdata = cbind(lfda.result, label=label)[idx,]), 1, which.max), label[idx])
mean(apply(predict(model, s=s, newdata = cbind(lfda.result, label=label)[idx,]), 1, which.max) == as.integer(label[idx]))


