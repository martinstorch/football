setwd("D:/gitrepository/Football/football/TF/data")

library(lfda)
library(caret)

data1 <- read.csv("all_features.csv")
labels1 <- read.csv("all_labels.csv")
teamdata1 <- read.csv("teams_onehot.csv")

is_new <- data1$Predict=="True"
data1 <- data1[,setdiff(colnames(data1), c("Team1","Team2","Team1_index","Team2_index","Predict" ))]
data1 <- cbind(data1, teamdata1)

labels<-labels1[!is_new,]
data<-data1[!is_new,]

model.pca <- prcomp(data1, center = TRUE, scale. = TRUE)
# summary(model.pca)
# plot(data.pca$sdev)
# plot(summary(data.pca)$importance[2,])
# plot(summary(data.pca)$importance[3,])

data1.sc <- scale(data1, center= model.pca$center)
data1.pca <- data1.sc %*% model.pca$rotation
data1.pca <- data1.pca[,1:173]


metric = c("orthonormalized", "plain", "weighted")
# trans = preProcess(data, c("BoxCox", "center", "scale"))
# transdata <- data.frame(trans = predict(trans, data))

lfda.labels<-labels1[!is_new,]
lfda.data<-data1.pca[!is_new,]

target_label1 <- lfda.labels$T1_GFT-labels$T2_GFT;
target_label1<-ifelse(target_label1<(-3), -3, ifelse(target_label1>3, 3, target_label1))
lfda.model1 <- lfda(lfda.data, target_label1, r=10,  metric = metric, knn = 5)

lfda.data1<-predict(lfda.model1, data1.pca)

target_label2 <- lfda.labels$T1_GFT+labels$T2_GFT;
target_label2<-ifelse(target_label2>5, 5, target_label2)
lfda.model2 <- lfda(lfda.data, target_label2, r=10,  metric = metric, knn = 5)

lfda.data2<-predict(lfda.model2, data1.pca)

lfda.result<-data.frame(lfda.data1, lfda.data2)

write.csv(lfda.result, "lfda_data.csv", row.names=FALSE)






