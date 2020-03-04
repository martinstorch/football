#setwd("~/LearningR/Bundesliga")
library(dplyr)
library(plotrix)
library(graphics)
library(reshape2)
library(RColorBrewer)
library(ggplot2)

model_list<-c(
  "d:\\Models\\model_1920_pistor/new_predictions_df.csv",
  "D:\\Models\\model_1920_gd///new_predictions_df.csv",
  "d:\\Models\\model_1920_sky//new_predictions_df.csv",
  "d:\\Models\\model_1920_pistor_verify/new_predictions_df.csv",
  "D:\\Models\\model_1920_gd_verify///new_predictions_df.csv",
  "d:\\Models\\model_1920_sky_verify//new_predictions_df.csv",
  
  "d:\\Models\\model_spi_pistor_verify2/new_predictions_df.csv",
  "D:\\Models\\model_spi_gd_verify///new_predictions_df.csv",
  "d:\\Models\\model_spi_sky_verify//new_predictions_df.csv"
)

# predictions_file<-"d:\\Models\\model_1920_pistor/new_predictions_df.csv"
# predictions_file<-"D:\\Models\\model_1920_gd///new_predictions_df.csv"
# predictions_file<-"d:\\Models\\model_1920_sky//new_predictions_df.csv"
# predictions_file<-"d:\\Models\\model_1920_pistor_verify/new_predictions_df.csv"
# predictions_file<-"D:\\Models\\model_1920_gd_verify///new_predictions_df.csv"
# predictions_file<-"d:\\Models\\model_1920_sky_verify//new_predictions_df.csv"
for (m in model_list[7:9])
  create_outputs(m)

create_outputs<-function(predictions_file){

if (grepl("pistor", predictions_file)){
  point_type<-"p_points"
  cut_off_level_low<-0.70
  cut_off_level_high<-1.1
  human_level<-320/9/31
  human_level_median <- 219 / 9 / 31
}

if (grepl("sky", predictions_file)){
  point_type<-"s_points"
  cut_off_level_low<-1.24
  cut_off_level_high<-1.7
  human_level<-297/6/30
  human_level_median <- 282 / 6 / 30
}

if (grepl("gd", predictions_file)){
  point_type<-"d_points"
  cut_off_level_low<-4.1
  cut_off_level_high<-8.8
  human_level<-297/6/30
  human_level_median <- 282 / 6 / 30
}

predictions_data <- read.csv(predictions_file)
#predictions_data<-predictions_data%>%filter(global_step>271600)

predictions_data<-predictions_data%>%filter(train>cut_off_level_low & test>cut_off_level_low)
predictions_data<-predictions_data%>%filter(train<cut_off_level_high & test<cut_off_level_high)

print(range(as.character(predictions_data$Date)))
table(predictions_data$pred)
table(predictions_data$pred, predictions_data$Prefix)
#summary(predictions_data$Date)
table(predictions_data$Prefix)
unique(predictions_data$global_step)

filename <- paste(basename(dirname(predictions_file)), Sys.Date(), ".pdf", sep="_")
pdf(filename, onefile = TRUE, width = 18, height = 7, title = point_type)

print(predictions_data%>%dplyr::select(Prefix, train, test)%>%tidyr::gather(type, score, train:test)%>%
  ggplot(aes(x=Prefix, y=score, fill=type))+geom_boxplot()+ggtitle(dirname(predictions_file))
)

# test ~ train  - points and ellipsis summary
point_data <- predictions_data %>% group_by(global_step, Prefix)%>% summarise(train=mean(train), test=mean(test))
pp<-point_data %>%filter(!is.na(test), !is.na(train), Prefix!="cp1") # %>%mutate(Prefix=factor(Prefix))
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

hometeams<-unique(predictions_data[predictions_data$Where=="Home",c("Team1", "Team2")])
print(hometeams)

data_summary <- predictions_data %>% group_by(Prefix) %>% summarise(TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) )
print(data_summary%>%arrange(-TestMean))

evaluate2<-function(t){
  onematch1 <- predictions_data %>% filter(Team1==hometeams$Team1[t] & Team2==hometeams$Team2[t] & !is.na(train) & !is.na(test)) %>% dplyr::select(Team1, Team2, Prefix, pred, train, test)
  onematch2 <- predictions_data %>% filter(Team2==hometeams$Team1[t] & Team1==hometeams$Team2[t] & !is.na(train) & !is.na(test)) %>% dplyr::select(Team1=Team2, Team2=Team1, Prefix, pred, train, test) %>% mutate(pred=stringi::stri_reverse(pred))
  onematch <- rbind(onematch1, onematch2)
  #onematch <- onematch %>% filter(!Prefix %in% c("xpt"))
  #onematch <- onematch %>% filter(Prefix %in% c("av", "cp", "sp", "pgpt" )) # 
  #onematch <- onematch %>% filter(Prefix %in% c("cbsp", "cpmx", "pgpt", "avmx", "xpt")) # 
  # eliminate groups with only one sample
  onematch <- onematch %>% group_by(pred, Prefix) %>% mutate(N=n()) %>% filter(N > 0) %>% ungroup() %>% mutate(legend_text=factor(paste(Prefix, pred, "-", N) )) 
  onematch <- onematch %>% group_by(pred) %>% mutate(N=n()) %>% filter(N > 2) %>% ungroup() %>% mutate(legend_text=factor(pred)) 
  matchname <- unique(paste(onematch$Team1, onematch$Team2, sep=" - "))
  levels(onematch$legend_text)
  nlevels(onematch$legend_text)
  print.data.frame(onematch %>% group_by(Prefix, pred) %>% summarise(N=n(), TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) ) %>% arrange(-TestMean) )
  print(matchname)
  print(onematch %>% group_by(pred) %>% summarise(N=n(), TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) ) %>% arrange(-TestMean) )
  ggplot(onematch, aes(x=pred, y=test, fill=pred))+
    geom_violin(scale = "count", draw_quantiles = c(0.25, 0.5, 0.75))+
    facet_wrap(~Prefix, nrow=2)+  
    stat_summary(fun.y = "mean", geom = "point", shape = 8, size = 3, color = "midnightblue") +
    #stat_summary(fun.y = "median", geom = "point", shape = 2, size = 3, color = "black")+
    geom_dotplot(binaxis="y", stackdir="center", position="dodge", dotsize=0.5, binwidth=0.002)+
    ggtitle(matchname)+ theme(legend.position="top")
    
}

for (i in 1:9)
  print(evaluate2(i))
dev.off()

}
# 
# evaluate2(1)
# evaluate2(2)
# evaluate2(3)
# evaluate2(4)
# evaluate2(5)
# evaluate2(6)
# evaluate2(7)
# evaluate2(8)
# evaluate2(9)
# 
# evaluate2(10)
# evaluate2(11)
# evaluate2(12)
# evaluate2(13)
# evaluate2(14)
# evaluate2(15)
# evaluate2(16)
# evaluate2(17)
# evaluate2(18)
# 
