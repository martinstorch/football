setwd("~/LearningR/Bundesliga")
f<-"rnn_candidate_kernel.csv"
f<-"rnn_gates_kernel.csv"
f<-"mapping.csv"


w <- read.csv(f, sep = " ", header = FALSE)
str(w)
summary(w)

boxplot(w)
abline(h=0, col="red")
boxplot(t(w))
abline(h=0, col="red")

plot(sort(colMeans(w)))

plot(sort(rowMeans(w)))

hist(colMeans(w), breaks = 25)
hist(rowMeans(w), breaks = 25)

w0<-as.matrix(w)
dim(w0)<-NULL
summary(w0)
hist(w0, breaks = 50)

heatmap(as.matrix(w))


summary(colMeans(w))
summary(rowMeans(w))
heatmap(var(w))

heatmap(var(t(w)))

plot(w[,1])
sqrt(var(w[,1]))
w[,1]

