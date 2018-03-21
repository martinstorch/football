setwd("~/LearningR/Bundesliga")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)

input_file<-"D:\\Models\\model_skimmed_1718_2/new_predictions_df_2.csv"
season<-"2017/18"
data <- read.csv(input_file)

matchdata <- data[data$X %in% 16:17,]
matchdata1 <- data[data$X == 16,]
matchdata2 <- data[data$X==17,]

table(matchdata1$pred)
table(matchdata2$pred)

table(data$pred)
m <- function(x) round(mean(x, na.rm = T), digits = 2)
data %>% group_by(Team1, Team2, pred) %>% summarise(count=n(), Win=m(win), Draw=m(draw), Loss=m(loss), WinPt=m(winPt), DrawPt=m(drawPt), LossPt=m(lossPt)) %>% arrange(Team1, Team2, count) %>% print.data.frame()  
#data %>% filter(Team1=='Bayern Munich' | Team2=='Bayern Munich') %>% 
#setOption("max.print")
data %>% filter(Prefix %in% c("sp", "pspt", "pgpt")) %>% 
  group_by(Team1, Team2, Prefix, pred) %>% summarise(count=n(), Win=m(win), Draw=m(draw), Loss=m(loss), WinPt=m(winPt), DrawPt=m(drawPt), LossPt=m(lossPt)) %>% arrange(Team1, Team2, Prefix, count) %>% print.data.frame()  

data %>% filter(Team1=='Bayern Munich')
