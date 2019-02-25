setwd("~/LearningR/Bundesliga")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)
library(RColorBrewer)
library(ggplot2)

point_type<-"z_points"
cut_off_level_low<-1.5
cut_off_level_high<-2.5
human_level<-461/9/31
human_level_median <- 449 / 9 / 31

point_type<-"p_points"
point_type<-"z_points"
cut_off_level_low<-0.7
cut_off_level_high<-1.4
human_level<-320/9/31
human_level_median <- 219 / 9 / 31
predictions_file<-"D:\\Models\\conv1_auto_pistor//new_predictions_df.csv"

point_type<-"s_points"
point_type<-"z_points"
cut_off_level_low<-1.15
cut_off_level_high<-2.8
human_level<-297/6/30
human_level_median <- 282 / 6 / 30
predictions_file<-"D:\\Models\\rnn1_sky/new_predictions_df.csv"
predictions_file<-"D:\\Models\\conv1_auto6_sky/new_predictions_df.csv"
predictions_file<-"D:\\Models\\conv1_auto6_sky/new_predictions_df_offset15.csv"
predictions_file<-"D:\\Models\\conv1_auto6_sky/new_predictions_df_offset25.csv"
predictions_file<-"D:\\Models\\conv1_auto6_sky/new_predictions_df_offset35.csv"


season<-"2018/19"
predictions_data <- read.csv(predictions_file)


table(predictions_data$pred)
table(predictions_data$pred, predictions_data$Prefix)
summary(predictions_data$Date)
print(unique(predictions_data$Date))
table(predictions_data$Prefix)
table(predictions_data$global_step)



boxplot(train-test~Prefix, data=predictions_data)
boxplot(train~Prefix, data=predictions_data)
boxplot(test~Prefix, data=predictions_data)

# line plot of steps
plot(test~global_step, data=predictions_data%>%filter(), col=Prefix, pch=as.integer(Prefix))
legend("topright", legend = levels(predictions_data$Prefix), horiz = T, text.col = as.integer(predictions_data$Prefix),
       pch = as.integer(predictions_data$Prefix), col = as.integer(predictions_data$Prefix))
linedata<-  predictions_data%>%select(global_step, Prefix, test)%>%dcast(global_step~Prefix, value.var="test", fun.aggregate = mean)
linedata[is.na(linedata)]<-0
#linedata<-linedata%>%select(-global_step)
matplot(x = linedata[,1], y=linedata[,2:ncol(linedata)]  , type = c("b"), col=1:ncol(linedata), pch=1:ncol(linedata), add=T)

point_data <- predictions_data %>% group_by(global_step, Prefix)%>% summarise(train=mean(train), test=mean(test))
# ranking of strategies
rankdata<-point_data %>% group_by(global_step)  %>% mutate(rank=rank(-test, ties.method = "min"))
plot(-rank~global_step, data=rankdata, col=Prefix, pch=as.integer(Prefix))
legend("bottomright", legend = levels(predictions_data$Prefix), horiz = T, text.col = as.integer(predictions_data$Prefix),
       pch = as.integer(predictions_data$Prefix), col = as.integer(predictions_data$Prefix))
linedata<-  rankdata%>%select(global_step, Prefix, rank)%>%mutate(rank=-rank)%>%dcast(global_step~Prefix, value.var="rank", fun.aggregate = mean)
linedata[is.na(linedata)]<-0
#linedata<-linedata%>%select(-step)
matplot(x = linedata[,1], y=linedata[,2:ncol(linedata)]   , type = c("b"), col=1:ncol(linedata), pch=1:ncol(linedata), add=T)
boxplot(-rank~Prefix, data=rankdata)

theme_set(theme_classic())
# Histogram on a Continuous (Numeric) Variable
g <- ggplot(rankdata, aes(rank)) + scale_fill_brewer(palette = "Spectral")
g + geom_histogram(aes(fill=Prefix), 
                   binwidth = 1.0, 
                   col="black", 
                   size=1.0) +  # change binwidth
  labs(title="Relative strategy ranking", 
       subtitle="1=best, 6=worst")  

g <- ggplot(point_data, aes(test)) + scale_fill_brewer(palette = "Spectral")
g + geom_histogram(aes(fill=Prefix),
                   bins=20,
                   #binwidth = 0.01, 
                   col="black"
                   #,size=0.01
) +  # change binwidth
  labs(title="Test scores", 
       subtitle="")  

g <- ggplot(point_data, aes(test))
g + geom_density(aes(fill=Prefix), alpha=0.5) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Test scores",
       fill="Prefix")

# http://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html

# test ~ train  - points and ellipsis summary
pp<-point_data %>%filter(!is.na(test), !is.na(train), Prefix!="cp1")%>%mutate(Prefix=factor(Prefix))
plot(test~train, data=pp, col=Prefix, pch=as.integer(Prefix))
legend("bottomright", legend = levels(pp$Prefix), horiz = T, text.col = as.integer(pp$Prefix),
       pch = as.integer(pp$Prefix), col = as.integer(pp$Prefix))
palette<-brewer.pal(nlevels(pp$Prefix),"Paired")
palette<-seq(nlevels(pp$Prefix))
with (pp, 
      car::dataEllipse(x=train, y=test, 
                       groups = Prefix, #group.labels = levels(Prefix)[-2], 
                       col=palette, lwd = 2,
                       add=T, robust = FALSE,
                       levels = c(0.8), 
                       fill=T, fill.alpha = 0.05
      ))

data_summary <- predictions_data %>% group_by(Prefix) %>% summarise(TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) )
print(data_summary%>%arrange(-TestMean))

hometeams<-unique(predictions_data[predictions_data$Where=="Home","Team1"])
print(hometeams)

t <- 1
evaluate<-function(t){
  onematch1 <- predictions_data %>% filter(Team1==hometeams[t] & !is.na(train) & !is.na(test)) %>% dplyr::select(Team1, Team2, Prefix, pred, train, test)
  onematch2 <- predictions_data %>% filter(Team2==hometeams[t] & !is.na(train) & !is.na(test)) %>% dplyr::select(Team1=Team2, Team2=Team1, Prefix, pred, train, test) %>% mutate(pred=stringi::stri_reverse(pred))
  onematch <- rbind(onematch1, onematch2)
  #onematch <- onematch %>% filter(Prefix %in% c("ens", "pgpt", "sp", "ps", "pghb", "pspt", "smpt"))
  #onematch <- onematch %>% filter(Prefix %in% c("ens", "cp", "sp", "smpt", "pspt", "pgpt", "cp2"))
  onematch <- onematch %>% filter(Prefix %in% c("av", "cp", "sp", "smpt", "pspt", "pgpt"))
  #onematch <- onematch %>% filter(Prefix %in% c("sm", "smpt", "pspt", "pgpt", "smpi", "cp")) # "sp", 
  # eliminate groups with only one sample
  onematch <- onematch %>% group_by(pred, Prefix) %>% mutate(N=n()) %>% filter(N > 0) %>% ungroup() %>% mutate(legend_text=factor(paste(Prefix, pred, "-", N) )) 
  onematch <- onematch %>% group_by(pred) %>% mutate(N=n()) %>% filter(N > 2) %>% ungroup() %>% mutate(legend_text=factor(pred)) 
  matchname <- unique(paste(onematch$Team1, onematch$Team2, sep=" - "))
  levels(onematch$legend_text)
  nlevels(onematch$legend_text)
  print.data.frame(onematch %>% group_by(pred, Prefix) %>% summarise(N=n(), TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) ) %>% arrange(-TestMean) )
  print(onematch %>% group_by(pred) %>% summarise(N=n(), TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) ) %>% arrange(-TestMean) )
  palette<-brewer.pal(nlevels(onematch$legend_text),"Paired")
  palette<-seq(nlevels(onematch$legend_text))
  with (onematch , 
        car::dataEllipse(x=train, y=test, 
                         groups = onematch$legend_text, group.labels = levels(legend_text), 
                         col=palette, lwd = 2,
                         add=FALSE, robust = FALSE,
                         levels = c(0.6), 
                         fill=T, fill.alpha = 0.03*(log(N)),
                         main=matchname))
  legend("topleft", legend=levels(onematch$legend_text), pch = 1:nlevels(onematch$legend_text), col=palette)
  abline(h=human_level, col="red")
  abline(h=human_level_median, col="red")
}
evaluate(1)
evaluate(2)
evaluate(3)
evaluate(4)
evaluate(5)
evaluate(6)
evaluate(7)
evaluate(8)
evaluate(9)

