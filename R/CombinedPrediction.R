setwd("~/LearningR/Bundesliga")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)
library(RColorBrewer)

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
result_file     <-"D:\\Models\\model_gen2_pistor_1819//results_df.csv"
predictions_file<-"D:\\Models\\model_gen2_pistor_1819/new_predictions_df.csv"
result_file     <-"D:\\Models\\simple_pistor_1819//results_df.csv"
predictions_file<-"D:\\Models\\simple_pistor_1819/new_predictions_df.csv"
#result_file     <-"D:\\Models\\model_gen_pistor_1819_2//results_df\ -\ Copy.csv"
#predictions_file<-"D:\\Models\\model_gen_pistor_1819_2/new_predictions_df\ -\ Copy.csv"

point_type<-"s_points"
point_type<-"z_points"
cut_off_level_low<-1.15
cut_off_level_high<-2.8
human_level<-297/6/30
human_level_median <- 282 / 6 / 30
result_file     <-"D:\\Models\\model_gen2_sky_1819//results_df.csv"
predictions_file<-"D:\\Models\\model_gen2_sky_1819/new_predictions_df.csv"
result_file     <-"D:\\Models\\simple_sky_1819//results_df.csv"
predictions_file<-"D:\\Models\\simple_sky_1819/new_predictions_df.csv"



season<-"2018/19"
predictions_data <- read.csv(predictions_file)
result_data <- read.csv(result_file)


#predictions_data <- cbind(predictions_data, read.csv("D:\\Models\\model_skimmed_1718_2/new_predictions_2.csv"))
#result_data <- cbind(result_data, read.csv("D:\\Models\\model_skimmed_1718_2/results_df_2.csv"))

table(predictions_data$pred)
table(predictions_data$pred, predictions_data$Prefix)
summary(predictions_data$Date)
print(unique(predictions_data$Date))
rows_per_iteration<-nrow(predictions_data) / length(unique(predictions_data$Date))
iterations<-nrow(predictions_data) / rows_per_iteration
predictions_data$step <- rep(1:iterations, each=rows_per_iteration)

# eliminate predictions_data with step 11
#predictions_data<-predictions_data%>%filter(step!=11)
#predictions_data$step[predictions_data$step>11]<-predictions_data$step[predictions_data$step>11]-1


point_data <- result_data[grep(point_type, result_data$Measure),]
point_data <- point_data[grep("summary", point_data$Measure, invert=TRUE),]
point_data$Measure <- gsub( "//", "/", as.character(point_data$Measure))
point_data$Measure <- factor(gsub( paste0("/",point_type), "", as.character(point_data$Measure)))
point_data$step <- rep(1:(nrow(point_data)/nlevels(point_data$Measure)), each=nlevels(point_data$Measure))
point_data$Test[point_data$Train < cut_off_level_low]<-NA
point_data$Test[point_data$Train > cut_off_level_high]<-NA
point_data$Train[point_data$Train < cut_off_level_low]<-NA
point_data$Train[point_data$Train > cut_off_level_high]<-NA
point_data$Test[point_data$Test < cut_off_level_low]<-NA
point_data$Test[point_data$Test > cut_off_level_high]<-NA
point_data$Train[point_data$Test < cut_off_level_low]<-NA
point_data$Train[point_data$Test > cut_off_level_high]<-NA
point_data <- point_data %>% dplyr::select(Prefix=Measure, Train, Test, step)
data_summary <- point_data %>% group_by(Prefix) %>% summarise(TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) )

seq_data_test <- dcast(point_data, step ~ Prefix, sum, value.var="Test") 
seq_data_train <- dcast(point_data, step ~ Prefix, sum, value.var="Train") 

table(predictions_data$Prefix)
table(predictions_data$step)
table(point_data$step)

combined_data <- predictions_data%>%inner_join(point_data, by = c("Prefix", "step"))
combined_data %>% group_by(Team1, Team2, pred, Prefix) %>% 
  summarise(N=n(), TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), 
            TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) ) %>% print.data.frame()
hometeams<-unique(combined_data[combined_data$Where=="Home","Team1"])

boxplot(Train-Test~Prefix, data=point_data)
boxplot(Train~Prefix, data=point_data)
boxplot(Test~Prefix, data=point_data)


# line plot of steps
plot(Test~step, data=point_data%>%filter(), col=Prefix, pch=as.integer(Prefix))
legend("topright", legend = levels(point_data$Prefix), horiz = T, text.col = as.integer(point_data$Prefix),
       pch = as.integer(point_data$Prefix), col = as.integer(point_data$Prefix))
linedata<-  point_data%>%select(step, Prefix, Test)%>%dcast(step~Prefix, value.var="Test")
linedata[is.na(linedata)]<-0
linedata<-linedata%>%select(-step)
matplot(linedata  , type = c("b"), col=1:ncol(linedata), pch=1:ncol(linedata), add=T)


# test ~ train  - points and ellipsis summary
pp<-point_data %>%filter(!is.na(Test), !is.na(Train), Prefix!="cp1")%>%mutate(Prefix=factor(Prefix))
plot(Test~Train, data=pp, col=Prefix, pch=as.integer(Prefix))
legend("bottomright", legend = levels(pp$Prefix), horiz = T, text.col = as.integer(pp$Prefix),
       pch = as.integer(pp$Prefix), col = as.integer(pp$Prefix))
palette<-brewer.pal(nlevels(pp$Prefix),"Paired")
palette<-seq(nlevels(pp$Prefix))
with (pp, 
      car::dataEllipse(x=Train, y=Test, 
                       groups = Prefix, #group.labels = levels(Prefix)[-2], 
                       col=palette, lwd = 2,
                       add=T, robust = FALSE,
                       levels = c(0.8), 
                       fill=T, fill.alpha = 0.05
                       ))

print(data_summary%>%arrange(-TestMean))

all_data<-combined_data

combined_data%>%select(Team1, Team2)%>%unique()
combined_data%>%group_by(Team1, Team2)%>%count()

combined_data%>%mutate(ii=row_number()-1, z=ii%%54)%>%filter(z==0, ii<=8*54)%>%select(ii, z, Team1, Team2, Prefix)

combined_data<-
rbind(all_data%>%filter(Prefix=="ens")%>%mutate(ii=row_number()-1, z=ii%%27)%>%filter(z<9)
, all_data%>%filter(Prefix!="ens")%>%mutate(ii=row_number()-1, z=ii%%54)%>%filter(z<18)
)
combined_data<-
  rbind(all_data%>%filter(Prefix=="ens")%>%mutate(ii=row_number()-1, z=ii%%27)%>%filter(z>=9, z<18)
        , all_data%>%filter(Prefix!="ens")%>%mutate(ii=row_number()-1, z=ii%%54)%>%filter(z>=18,z<36)
  )
combined_data<-
  rbind(all_data%>%filter(Prefix=="ens")%>%mutate(ii=row_number()-1, z=ii%%27)%>%filter(z>=18)
        , all_data%>%filter(Prefix!="ens")%>%mutate(ii=row_number()-1, z=ii%%54)%>%filter(z>=36)
  )

t <- 1
evaluate<-function(t){
  onematch1 <- combined_data %>% filter(Team1==hometeams[t] & !is.na(Train) & !is.na(Test)) %>% dplyr::select(Team1, Team2, Prefix, pred, Train, Test)
  onematch2 <- combined_data %>% filter(Team2==hometeams[t] & !is.na(Train) & !is.na(Test)) %>% dplyr::select(Team1=Team2, Team2=Team1, Prefix, pred, Train, Test) %>% mutate(pred=stringi::stri_reverse(pred))
  onematch <- rbind(onematch1, onematch2)
  #onematch <- onematch %>% filter(Prefix %in% c("ens", "pgpt", "sp", "ps", "pghb", "pspt", "smpt"))
  onematch <- onematch %>% filter(Prefix %in% c("ens", "cp", "sp", "smpt", "pspt", "pgpt", "cp2"))
  #onematch <- onematch %>% filter(Prefix %in% c("sm", "smpt", "pspt", "pgpt", "smpi", "cp")) # "sp", 
  # eliminate groups with only one sample
  onematch <- onematch %>% group_by(pred, Prefix) %>% mutate(N=n()) %>% filter(N > 0) %>% ungroup() %>% mutate(legend_text=factor(paste(Prefix, pred, "-", N) )) 
  onematch <- onematch %>% group_by(pred) %>% mutate(N=n()) %>% filter(N > 2) %>% ungroup() %>% mutate(legend_text=factor(pred)) 
  matchname <- unique(paste(onematch$Team1, onematch$Team2, sep=" - "))
  levels(onematch$legend_text)
  nlevels(onematch$legend_text)
  print.data.frame(onematch %>% group_by(pred, Prefix) %>% summarise(N=n(), TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) ) %>% arrange(-TestMean) )
  print(onematch %>% group_by(pred) %>% summarise(N=n(), TrainMean=mean(Train, na.rm = T), TestMean=mean(Test, na.rm = T), TrainStddev=sd(Train, na.rm = T), TestStddev=sd(Test, na.rm = T) ) %>% arrange(-TestMean) )
  palette<-brewer.pal(nlevels(onematch$legend_text),"Paired")
  palette<-seq(nlevels(onematch$legend_text))
  with (onematch , 
        car::dataEllipse(x=Train, y=Test, 
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

