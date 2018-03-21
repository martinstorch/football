setwd("~/LearningR/Bundesliga")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)
library(RColorBrewer)

point_type<-"z_points"
cut_off_level_low<-1.6
cut_off_level_high<-2.5
human_level<-405/9/27
human_level_median <- 372 / 9 / 27

point_type<-"p_points"
cut_off_level_low<-0.7
cut_off_level_high<-1.1
human_level<-285/9/27
human_level_median <- 196 / 9 / 27

point_type<-"s_points"
cut_off_level_low<-1.0
cut_off_level_high<-1.8
human_level<-274/6/27
human_level_median <- 251 / 6 / 27

#input_file<-"D:\\Models\\model_map_1718_2/results_df.csv"

input_file<-"D:\\Models\\model_skimmed_1718_2//results_df.csv"
input_file<-"D:\\Models\\model_skimmed_1718_2//results_df_2.csv"
input_file2<-"D:\\Models\\model_skimmed_1718_2//results_df - Copy (3).csv"
season<-"2017/18"

input_file     <-"D:\\Models\\rolling_skimmed_1418/results_df.csv"


input_file<-"D:\\Models\\model_map_1415/results_df.csv"
season<-"2014/15"

input_file<-"D:\\Models\\model_map_1516/results_df.csv"
season<-"2015/16"

human_level<-521/9/34
human_level_median <- 470 / 9 / 34
input_file<-"D:\\Models\\model_map_1617/results_df.csv"
season<-"2016/17"





boxplot(Train~Measure, point_data, main="Train Scores")
abline(h=human_level, col="red")
abline(h=human_level_median, col="red")

boxplot(Test~Measure, point_data, main="Test Scores")
abline(h=human_level, col="red")
abline(h=human_level_median, col="red")

boxplot(Test/Train~Measure, point_data, main="Train-Test gap")
#boxplot(Test-Train~Measure, point_data, main="Train-Test gap")

plot(Test ~ Train, data=point_data, col=Measure, pch=as.integer(Measure))#, main=paste(season, "test / train scores with mean and stddev"))
abline(h=human_level, col="red")
abline(h=human_level_median, col="red")
legend("topright", levels(point_data$Measure), pch = 1:nlevels(point_data$Measure), col=1:nlevels(point_data$Measure))
points(TestMean ~ TrainMean, data=data_summary, col=Measure, pch=as.integer(Measure), cex=2, lwd=3)
draw.ellipse(x=data_summary$TrainMean, y=data_summary$TestMean, a = data_summary$TrainStddev, b = data_summary$TestStddev,
             border = 1:nlevels(point_data$Measure), lwd=2, angle = 0)


train_test<-point_data %>% filter(Measure=="sp") %>% dplyr::select(Train, Test) %>% filter(!is.na(Test))
plot(train_test)
abline(lm(Test~Train, data=train_test))
cor(train_test)
plot(lm(Test~Train, data=train_test))  

car::dataEllipse(x=point_data$Train, y=point_data$Test, 
                 groups = point_data$Measure, group.labels = levels(point_data$Measure), 
                 col=brewer.pal(nlevels(point_data$Measure),"Paired"),
                 pch=1:nlevels(point_data$Measure),
                 add=FALSE,
                 levels = c(0.7),
                 fill=T)
legend("topright", levels(point_data$Measure), pch = 1:nlevels(point_data$Measure), col=brewer.pal(nlevels(point_data$Measure),"Paired"))

display.brewer.all()


plot(sp ~ p7, seq_data_test[,-1])
abline(lm(sp ~ p7, seq_data_test[,-1]))

plot(sp-p7 ~ p7, seq_data_train[,-1])
lm(sp ~ p7, seq_data_train[,-1])

cor(seq_data_test[,-1])
plotcorr(cor(seq_data_test[,-1]))

heatmap(cor(seq_data_test[,-1]))
heatmap(cov(seq_data_test[,-1]))
heatmap(as.matrix(seq_data_test[,-1]))


#####################################################

load_data<-function(filename, season){
  inp_data <- read.csv(filename)
  inp_data$season<-season
  return(inp_data)
}
data <- data.frame()
data <- rbind(data, load_data("D:\\Models\\model_map_1415/results_df.csv", "2014/15"))
data <- rbind(data, load_data("D:\\Models\\model_map_1516/results_df.csv", "2015/16"))
data <- rbind(data, load_data("D:\\Models\\model_map_1617/results_df.csv", "2016/17"))
data <- rbind(data, load_data("D:\\Models\\model_map_1718_2/results_df.csv", "2017/18"))
season<-"2014/15/16/17/18"



