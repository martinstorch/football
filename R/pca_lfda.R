setwd("c:/git/football/TF/data")
setwd("d:/gitrepository/Football/football/TF/data")

library(lfda)
library(caret)
library(dplyr)
library(tidyimpute)
library(probFDA)
library(MASS)
 

data1 <- read.csv("all_features.csv")
labels1 <- read.csv("all_labels.csv")
teamdata1 <- read.csv("teams_onehot.csv")

is_new <- data1$Predict=="True"
is_new[1:2447]<-TRUE # OOB  seasons "0910", "1011", "1112", "1213"

# collect labels and features from previous match as additional labels: 
mh1 <- data1%>%bind_cols(labels1)%>%group_by(Team1, Team1_index)%>%mutate_at(vars(-group_cols()), .funs = dplyr::lag)%>%ungroup()%>%impute_mean_all()
mh2 <- data1%>%bind_cols(labels1)%>%group_by(Team2, Team2_index)%>%mutate_at(vars(-group_cols()), .funs = dplyr::lag)%>%ungroup()%>%impute_mean_all()
mh12 <- data1%>%bind_cols(labels1)%>%group_by(Team1, Team1_index, Team2, Team2_index)%>%mutate_at(vars(-group_cols()), .funs = dplyr::lag)%>%ungroup()%>%impute_mean_all()

data1<- data1%>%
  bind_cols(mh1%>%select(-Team1,-Team2,-Team1_index,-Team2_index,-Predict))%>%
  bind_cols(mh2%>%select(-Team1,-Team2,-Team1_index,-Team2_index,-Predict))%>%
  bind_cols(mh12%>%select(-Team1,-Team2,-Team1_index,-Team2_index,-Predict))

#dim(data1)

data1 <- data1[,setdiff(colnames(data1), c("Team1","Team2","Team1_index","Team2_index","Predict" ))]
#data1 <- cbind(data1, teamdata1)

trans = preProcess(data1, c("BoxCox", "center", "scale"))
data1 <- data.frame(trans = predict(trans, data1))

labels<-labels1[!is_new,]
data<-data1[!is_new,]
#colnames(data1)
model.pca <- prcomp(data1, center = TRUE, scale. = TRUE)
# summary(model.pca)
# plot(model.pca$sdev)
# plot(summary(model.pca)$importance[2,])
# plot(summary(model.pca)$importance[3,])

data1.sc <- scale(data1, center= model.pca$center)
data1.pca <- data1.sc %*% model.pca$rotation
#data1.pca <- data1.pca[,1:116]
data1.pca <- data1.pca[,1:588]
#data1.pca <- data1.pca[,1:150]


metric = c("orthonormalized", "plain", "weighted")

lfda.labels<-labels1[!is_new,]
lfda.data<-data1.pca[!is_new,]

target_label1 <- lfda.labels$T1_GFT-labels$T2_GFT;
target_label1<-ifelse(target_label1<(-3), -3, ifelse(target_label1>3, 3, target_label1))
lfda.model1 <- lfda(lfda.data, target_label1, r=10,  metric = metric, knn = 5)
#lfda.model1 <- lda(lfda.data, target_label1)
# lfda.model1 <- self(lfda.data, target_label1, r=10,  metric = metric, kNN=5, beta = 0.3)
# plot(lfda.model1)

lfda.data1<-predict(lfda.model1, data1.pca)

target_label2 <- lfda.labels$T1_GFT+labels$T2_GFT;
target_label2<-ifelse(target_label2>5, 5, target_label2)
lfda.model2 <- lfda(lfda.data, target_label2, r=10,  metric = metric, knn = 5)
#lfda.model2 <- lda(lfda.data, target_label2)
# lfda.model2 <- self(lfda.data, target_label2, r=10,  metric = metric, kNN = 5, beta=0.3)
# plot(lfda.model2)

lfda.data2<-predict(lfda.model2, data1.pca)

lfda.result<-data.frame(lfda.data1, lfda.data2)

write.csv(lfda.result, "lfda_data.csv", row.names=FALSE)

# for (i in 1:20) {
#   plot(lfda.result[,i], main=colnames(lfda.result[i]))
# }

# table(target_label1)
# 
# table(predict(lfda.model1, newdata=lfda.data)$class, target_label1)
# table(predict(lfda.model1, newdata=data1.pca[is_new,])$class, (lfda.labels$T1_GFT-labels$T2_GFT)[is_new])
# 
# table(predict(lfda.model2)$class)
# 
# lfda.model1 <- lda(lfda.data, grouping=target_label1)
# 
# table(sign(as.integer(predict(lfda.model1, newdata=lfda.data)$class)-4), sign(target_label1))
# mean(sign(as.integer(predict(lfda.model1, newdata=lfda.data)$class)-4) == sign(target_label1))
# 
# table(sign(as.integer(predict(lfda.model1, newdata=data1.pca[is_new,])$class)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# mean(sign(as.integer(predict(lfda.model1, newdata=data1.pca[is_new,])$class)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# 
# str(predict(lfda.model1, newdata=lfda.data, type = "class"))
# 
# probFDA()
# 
# target_label1<-factor(target_label1)
# lfda.model1 <- pfda(lfda.data, target_label1, model=c('DB'), crit = "bic", cv.fold = 10, kernel = "rbf", display = TRUE)
# #str(lfda.model1)
# 
# predtrain <- predict(lfda.model1, X=lfda.data)
# table(predtrain$cls)
# table(target_label1)
# 
# table(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4), sign(as.integer(target_label1)-4))
# mean(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4) == sign(as.integer(target_label1)-4))
# 
# table(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# mean(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# table(predict(lfda.model1, X=lfda.data)$cls)#, target_label1)
# 
# as.integer(predict(lfda.model1, X=data1.pca[6193:6228,])$cls)-9
# 
# 
# str(lfda.model1)
# lfda.model1$prms$prop
# 
# 
# model = "DBk"
# 
# -1   0   1
# -1 767 310 314
# 0  339 290 334
# 1  312 309 770
# > mean(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4) == sign(as.integer(target_label1)-4))
# [1] 0.4878505
# > table(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# -1   0   1
# -1 506 269 249
# 0  163 125 163
# 1  249 271 506
# > mean(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# [1] 0.4546182
# 
# * Selected model: AkjBk 
# > #str(lfda.model1)
#   > 
#   > table(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4), sign(as.integer(target_label1)-4))
# 
# -1   0   1
# -1 840 322 325
# 0  252 266 251
# 1  326 321 842
# > mean(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4) == sign(as.integer(target_label1)-4))
# [1] 0.5201602
# > 
#   > table(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# -1   0   1
# -1 519 272 252
# 0  147 123 147
# 1  252 270 519
# > mean(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# [1] 0.4642143
# 
# -1   0   1
# -1 834 323 326
# 0  260 261 257
# 1  324 325 835
# > mean(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4) == sign(as.integer(target_label1)-4))
# [1] 0.5153538
# > 
#   > table(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# -1   0   1
# -1 517 274 250
# 0  151 118 148
# 1  250 273 520
# > mean(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# [1] 0.4618153
# > 
#  
#   lfda.model1 <- pfda(lfda.data, target_label1, model=c('DB','AB'), crit = "bic", cv.fold = 10, kernel = "", display = TRUE)
# DB       AB 
# -3705478 -3706712 
# * Selected model: DB 
# > table(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4), sign(as.integer(target_label1)-4))
# 
# -1   0   1
# -1 864 345 357
# 0  199 217 198
# 1  355 347 863
# > mean(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4) == sign(as.integer(target_label1)-4))
# [1] 0.5190921
# > 
#   > table(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# -1   0   1
# -1 541 284 263
# 0  113  99 114
# 1  264 282 541
# > mean(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# [1] 0.4722111
# 
# 
# > lfda.model1 <- pfda(lfda.data, target_label1, model=c('DkBk','AkB'), crit = "bic", cv.fold = 10, kernel = "", display = TRUE)
# DkBk      AkB 
# -3682388 -3703797 
# * Selected model: DkBk 
# > #str(lfda.model1)
#   > 
#   > table(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4), sign(as.integer(target_label1)-4))
# 
# -1   0   1
# -1 841 320 320
# 0  258 268 256
# 1  319 321 842
# > mean(sign(as.integer(predict(lfda.model1, X=lfda.data)$cls)-4) == sign(as.integer(target_label1)-4))
# [1] 0.5209613
# > 
#   > table(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4), sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# 
# -1   0   1
# -1 517 271 250
# 0  151 124 149
# 1  250 270 519
# > mean(sign(as.integer(predict(lfda.model1, X=data1.pca[is_new,])$cls)-4) == sign(labels1$T1_GFT-labels1$T2_GFT)[is_new])
# [1] 0.4638145
# >  
# 
