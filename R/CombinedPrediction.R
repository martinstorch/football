setwd("~/LearningR/Bundesliga")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)

point_type<-"z_points"
cut_off_level_low<-1.5
cut_off_level_high<-2.5
human_level<-461/9/31
human_level_median <- 449 / 9 / 31

point_type<-"p_points"
cut_off_level_low<-0.7
cut_off_level_high<-1.1
human_level<-320/9/31
human_level_median <- 219 / 9 / 31

point_type<-"s_points"
cut_off_level_low<-1.0
cut_off_level_high<-1.8
human_level<-297/6/30
human_level_median <- 282 / 6 / 30

#input_file<-"D:\\Models\\model_map_1718_2/results_df.csv"

result_file     <-"D:\\Models\\model_skimmed_1718_2/results_df_2.csv"
predictions_file<-"D:\\Models\\model_skimmed_1718_2/new_predictions_df_2.csv"
result_file     <-"D:\\Models\\model_skimmed_1718_2/results_df_test.csv"
predictions_file<-"D:\\Models\\model_skimmed_1718_2/new_predictions_df_test.csv"
result_file     <-"D:\\Models\\model_skimmed_1718_2/results_df.csv"
predictions_file<-"D:\\Models\\model_skimmed_1718_2/new_predictions_df.csv"

result_file     <-"D:\\Models\\rolling_skimmed_1418/results_df.csv"
predictions_file<-"D:\\Models\\rolling_skimmed_1418/new_predictions_df.csv"

result_file     <-"D:\\Models\\model_mb_1718_2/results_df.csv"
predictions_file<-"D:\\Models\\model_mb_1718_2/new_predictions_df.csv"

result_file     <-"D:\\Models\\model_cp_1718/results_df.csv"
predictions_file<-"D:\\Models\\model_cp_1718/new_predictions_df.csv"

season<-"2017/18"
predictions_data <- read.csv(predictions_file)
result_data <- read.csv(result_file)
#predictions_data <- cbind(predictions_data, read.csv("D:\\Models\\model_skimmed_1718_2/new_predictions_2.csv"))
#result_data <- cbind(result_data, read.csv("D:\\Models\\model_skimmed_1718_2/results_df_2.csv"))

table(predictions_data$pred)
table(predictions_data$pred, predictions_data$Prefix)
summary(predictions_data$Date)
rows_per_iteration<-nrow(predictions_data) / length(unique(predictions_data$Date))
iterations<-nrow(predictions_data) / rows_per_iteration
predictions_data$step <- rep(1:iterations, each=rows_per_iteration)

point_data <- result_data[grep(point_type, result_data$Measure),]
point_data <- point_data[grep("summary", point_data$Measure, invert=TRUE),]
point_data$Measure <- gsub( "//", "/", as.character(point_data$Measure))
point_data$Measure <- factor(gsub( paste0("/",point_type), "", as.character(point_data$Measure)))
point_data$step <- rep(1:(nrow(point_data)/nlevels(point_data$Measure)), each=nlevels(point_data$Measure))
point_data$Test[point_data$Train < cut_off_level_low]<-NA
point_data$Test[point_data$Train > cut_off_level_high]<-NA
point_data$Train[point_data$Train < cut_off_level_low]<-NA
point_data$Train[point_data$Train > cut_off_level_high]<-NA
point_data <- point_data %>% dplyr::select(Prefix=Measure, Train, Test, step)
data_summary <- point_data %>% group_by(Prefix) %>% summarize(TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) )

seq_data_test <- dcast(point_data, step ~ Prefix, sum, value.var="Test") 
seq_data_train <- dcast(point_data, step ~ Prefix, sum, value.var="Train") 

combined_data <- predictions_data%>%inner_join(point_data, by = c("Prefix", "step"))
combined_data %>% group_by(Team1, Team2, pred, Prefix) %>% 
  summarize(N=n(), TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), 
            TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) ) %>% print.data.frame()
hometeams<-unique(combined_data[combined_data$Where=="Home","Team1"])

t <- 1
evaluate<-function(t){
onematch1 <- combined_data %>% filter(Team1==hometeams[t] & !is.na(Train) & !is.na(Test)) %>% dplyr::select(Team1, Team2, Prefix, pred, Train, Test)
onematch2 <- combined_data %>% filter(Team2==hometeams[t] & !is.na(Train) & !is.na(Test)) %>% dplyr::select(Team1=Team2, Team2=Team1, Prefix, pred, Train, Test) %>% mutate(pred=stringi::stri_reverse(pred))
onematch <- rbind(onematch1, onematch2)
#onematch <- onematch %>% filter(Prefix %in% c("ens", "pgpt", "sp", "ps", "pghb", "pspt", "smpt"))
onematch <- onematch %>% filter(Prefix %in% c("sm", "smpt", "pspt", "pgpt", "smpi", "cp")) # "sp", 
# eliminate groups with only one sample
onematch <- onematch %>% group_by(pred, Prefix) %>% mutate(N=n()) %>% filter(N > 6) %>% ungroup() %>% mutate(legend_text=factor(paste(Prefix, pred, "-", N) )) 
onematch <- onematch %>% group_by(pred) %>% mutate(N=n()) %>% filter(N > 2) %>% ungroup() %>% mutate(legend_text=factor(pred)) 
matchname <- unique(paste(onematch$Team1, onematch$Team2, sep=" - "))
levels(onematch$legend_text)
nlevels(onematch$legend_text)
print(onematch %>% group_by(pred, Prefix) %>% summarize(N=n(), TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) ) %>% arrange(-TestMean) )
print(onematch %>% group_by(pred) %>% summarize(N=n(), TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) ) %>% arrange(-TestMean) )
palette<-brewer.pal(nlevels(onematch$legend_text),"Paired")
palette<-seq(nlevels(onematch$legend_text))
with (onematch , 
  car::dataEllipse(x=Train, y=Test, 
                 groups = onematch$legend_text, group.labels = levels(legend_text), 
                 col=palette, lwd = 2,
                 add=FALSE, robust = FALSE,
                 levels = c(0.6), 
                 fill=T, fill.alpha = 0.05*(log(N)),
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
