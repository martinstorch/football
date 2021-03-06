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
cut_off_level_low<-0.70
cut_off_level_high<-1.1
human_level<-320/9/31
human_level_median <- 219 / 9 / 31
predictions_file<-"d:\\Models\\model_1920_pistor/new_predictions_df.csv"

point_type<-"s_points"
cut_off_level_low<-1.24
cut_off_level_high<-1.7
human_level<-297/6/30
human_level_median <- 282 / 6 / 30
predictions_file<-"d:\\Models\\model_1920_sky//new_predictions_df.csv"


point_type<-"d_points"
cut_off_level_low<-4.1
cut_off_level_high<-8.8
human_level<-297/6/30
human_level_median <- 282 / 6 / 30
predictions_file<-"D:\\Models\\model_1920_gd///new_predictions_df.csv"

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



boxplot(train-test~Prefix, data=predictions_data)
boxplot(train~Prefix, data=predictions_data)
boxplot(test~Prefix, data=predictions_data)

#predictions_data%>%filter(Prefix=="cp")%>%ggplot(aes(x=global_step, y=test))+geom_line()+geom_text(aes(label=test))

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

t <- 1
evaluate<-function(t){
  onematch1 <- predictions_data %>% filter(Team1==hometeams$Team1[t] & Team2==hometeams$Team2[t] & !is.na(train) & !is.na(test)) %>% dplyr::select(Team1, Team2, Prefix, pred, train, test)
  onematch2 <- predictions_data %>% filter(Team2==hometeams$Team1[t] & Team1==hometeams$Team2[t] & !is.na(train) & !is.na(test)) %>% dplyr::select(Team1=Team2, Team2=Team1, Prefix, pred, train, test) %>% mutate(pred=stringi::stri_reverse(pred))
  onematch <- rbind(onematch1, onematch2)
  #onematch <- onematch %>% filter(!Prefix %in% c("pg2", "xpt"))
  #onematch <- onematch %>% filter(Prefix %in% c("av", "cp", "sp", "pgpt" )) # 
  #onematch <- onematch %>% filter(Prefix %in% c("cbsp", "cpmx", "pgpt", "avmx")) # 
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
  print(onematch %>% group_by(pred) %>% summarise(N=n(), TrainMean=mean(train, na.rm = T), TestMean=mean(test, na.rm = T), TrainStddev=sd(train, na.rm = T), TestStddev=sd(test, na.rm = T) ) %>% arrange(-TestMean) )
  ggplot(onematch, aes(x=pred, y=test, fill=pred))+
    geom_violin(scale = "count", draw_quantiles = c(0.25, 0.5, 0.75))+
    facet_wrap(~Prefix, nrow=2)+  
    stat_summary(fun.y = "mean", geom = "point", shape = 8, size = 3, color = "midnightblue") +
    #stat_summary(fun.y = "median", geom = "point", shape = 2, size = 3, color = "black")+
    geom_dotplot(binaxis="y", stackdir="center", position="dodge", dotsize=0.5, binwidth=0.002)+
    ggtitle(matchname)+ theme(legend.position="top")
    
}
evaluate2(1)
evaluate2(2)
evaluate2(3)
evaluate2(4)
evaluate2(5)
evaluate2(6)
evaluate2(7)
evaluate2(8)
evaluate2(9)

evaluate2(10)
evaluate2(11)
evaluate2(12)
evaluate2(13)
evaluate2(14)
evaluate2(15)
evaluate2(16)
evaluate2(17)
evaluate2(18)

v<-predictions_data[predictions_data$Where=="Home" & predictions_data$Prefix!="pgpt" & predictions_data$Prefix!="av",]
v$match<-paste(v$Team1, v$Team2, sep=" - ")
ggplot(v, aes(x=win, y=loss, col=pred, shape=Prefix))+geom_point()
ggplot(v, aes(x=winPt, y=lossPt, col=pred, shape=Prefix))+geom_point()
ggplot(v, aes(x=est1, y=est2, col=pred, shape=Prefix))+geom_point()
ggplot(v, aes(x=est1, y=est2, col=pred, shape=match, alpha=test))+geom_point() + scale_shape_manual(values=seq(0,8)) + scale_color_brewer(palette = "Spectral") + xlab("Schätzwert Heimtore") + ylab("Schätzwert Auswärtstore")


v$pred2 <- factor(v$pred, ordered = TRUE, 
                  levels = c("0:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"))
v$act2 <- factor(v$act, ordered = TRUE, 
                  levels = c("0:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"))
ggplot(v, aes(x=est1, y=est2, col=pred2, shape=Prefix))+geom_point(alpha=0.4) + 
  scale_color_brewer(palette = "Spectral") + 
  xlab("Schätzwert Heimtore") + ylab("Schätzwert Auswärtstore")

v<-predictions_data[predictions_data$Where=="Home" ,]
v$match<-paste(v$Team1, v$Team2, sep=" - ")
v$pred2 <- factor(v$pred, ordered = TRUE, 
                  levels = c("0:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"))
v$act2 <- factor(v$act, ordered = TRUE, 
                 levels = c("0:3", "0:2", "1:2", "0:1", "0:0", "1:1", "1:0", "2:1", "2:0", "3:1", "3:0"))
ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2, col=pred2))+geom_point(alpha=0.4) + 
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2, col=act2))+geom_point(alpha=0.4) + 
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Tatsächliches Ergebnis",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

v$pGS<-strtoi(substr(as.character(v$pred),1,1))
v$pGC<-strtoi(substr(as.character(v$pred),3,3))
table(v$pGS)
table(v$pGC)
v$GS<-strtoi(substr(as.character(v$act),1,1))
v$GC<-strtoi(substr(as.character(v$act),3,3))
v$pdiff<-factor(v$pGS-v$pGC)
v$adiff<-factor(sapply(sapply(v$GS-v$GC, max, -5), min, 5))
v$pmin<-factor(apply(v[,c("pGS", "pGC")], 1, min))
v$amin<-factor(-apply(v[,c("GS", "GC")], 1, min))
v$ptend<-factor(sign(v$pGS-v$pGC))
v$atend<-factor(sign(v$GS-v$GC))
v$ptendr<-sign(v$pGS-v$pGC)
summary(v$pmin)
table(v$amin)
summary(v$adiff)

ggplot(v[v$Prefix=="sp",], aes(x=est1, y=est2, col=pdiff, shape=pmin))+geom_point(alpha=0.8) + 
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="sp",], aes(x=est1, y=est2, col=adiff, shape=amin))+geom_point(alpha=0.8) + 
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Tatsächliches Ergebnis",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")


ggplot(v[v$Prefix=="sp",], aes(x=est1, y=est2, col=ptend))+geom_point(alpha=0.8) + 
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="sp",], aes(x=est1, y=est2, col=atend, shape=amin))+geom_point(alpha=0.8) + 
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Tatsächliches Ergebnis",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")



g <- ggplot(v[v$Prefix=="sp",], aes(x=est1-est2))
g + geom_density(aes(fill=ptend), alpha=0.5) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")

g <- ggplot(v[v$Prefix=="pgpt",], aes(x=est1-est2))
g + geom_density(aes(fill=atend), alpha=0.5) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")

g <- ggplot(v[v$Prefix=="pg2" & v$dataset=="train",], aes(x=est1-est2))
g + geom_density(aes(fill=atend), alpha=0.5) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")

g <- ggplot(v[v$Prefix=="pgpt" & v$dataset=="test",], aes(x=est1-est2))
#g + geom_density(aes(fill=atend), alpha=0.5) + 
g + geom_histogram(aes(fill=atend), bins=20) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")


ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2))+geom_density_2d(aes(colour = ptend))+
  #scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2))+geom_density_2d(aes(colour = atend))+
  #scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2))+geom_density_2d(aes(colour = amin, size=stat(exp(level))))+
  scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="pgpt" & v$amin %in% c("0", "1", "2"),], aes(x=est1, y=est2))+geom_density_2d(aes(colour = amin))+
  #scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

g <- ggplot(v[v$Prefix=="pgpt" & v$dataset=="train",], aes(x=est1+est2))
#g + geom_density(aes(fill=amin), alpha=0.5) + 
g + geom_histogram(aes(fill=amin), bins=25) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")

g <- ggplot(v[v$Prefix=="pgpt" & v$dataset=="test",], aes(x=est1+est2))
#g + geom_density(aes(y=..density.., fill=amin), alpha=0.5) + 
g + geom_histogram(aes(fill=amin), bins=25) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")



g <- ggplot(v[v$Prefix=="pgpt" & v$dataset=="train",], aes(x=est1-est2))
g + geom_density(aes(fill=adiff), alpha=0.5) + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")

g <- ggplot(v[v$Prefix=="sp" & v$dataset=="test",], aes(x=est1-est2))
g + geom_histogram(aes(fill=adiff), bins=25) + 
  scale_fill_brewer(palette = "Spectral") + 
#g + geom_density(aes(fill=adiff), alpha=0.2, bw="SJ", position="stack") + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Heimtore minus Auswärtstore",
       fill="Prognose")


ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2))+geom_density_2d(aes(colour = ptend))+
  #scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

ggplot(v[v$Prefix=="pgpt",], aes(x=est1, y=est2))+geom_density_2d(aes(colour = atend))+
  #scale_color_brewer(palette = "Spectral") + 
  labs(col = "Prognose",  x="Schätzwert Heimtore", y="Schätzwert Auswärtstore")

#g <- ggplot(v[v$Prefix=="pgpt" & v$dataset=="test",], aes(x=est1-est2))
g <- ggplot(v[v$Prefix=="pgpt" & v$dataset=="test",], aes(x=est1-est2))
g + geom_histogram(aes(fill=act2), bins=25) + 
  scale_fill_brewer(palette = "Spectral") 
  #g + geom_density(aes(fill=adiff), alpha=0.2, bw="SJ", position="stack") + 
  labs(title="Density plot", 
       subtitle="Test scores",
       caption="Source: Tensorflow",
       x="Schätzwert Gewinnwahrscheinlichkeit",
       fill="Prognose")
