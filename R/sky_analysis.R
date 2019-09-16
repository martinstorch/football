setwd("~/LearningR/Bundesliga/Analysis")
setwd("c:/users/marti")

library(metR)
library(ggExtra)
library(ggplot2)
library(ggpmisc)
library(ggmosaic)

library(dplyr)
library(reshape2)
library(lfda)
library(caret)
library(MASS)
library(tidyimpute)
library(na.tools)
library(lubridate)
library(tidyr)
library(TTR)

newdatafile<-"D:/gitrepository/Football/football/TF/sky_user_tipps.csv"
data<-read.csv(newdatafile, sep = ",", encoding = "utf-8")

nrow(data)

data<-unique(data)

nrow(data)

plot((data$Userid[data$Userid<100000]))

data$Userid<-factor(data$Userid)
#data$DateTo<-dmy(data$DateTo)
summary(data$Userid[data$Userid<100000])
hist(data$Userid[data$Userid<100000])

print(nlevels(data$Userid))

tail(levels(data$Userid), 600)

all_user_data<-data
all_user_data<-all_user_data%>%mutate(FTG=factor(paste(FTHG, FTAG, sep=":")), 
                    uBet=factor(paste(uFTHG, uFTAG, sep=":")),
                    FTR=factor(sign(FTHG-FTAG)+2),
                    uFTR=factor(sign(uFTHG-uFTAG)+2))
levels(all_user_data$FTR)<-c("A","D","H")
levels(all_user_data$uFTR)<-c("A","D","H")
all_user_data$uBet[all_user_data$uBet=="NA:NA"]<-NA
all_user_data<-droplevels(all_user_data)

all_user_data<-all_user_data%>%mutate(uBetOrder=uFTHG-uFTAG+0.01*sign(uFTHG-uFTAG)*uFTHG, uBet=reorder(uBet, uBetOrder),
       FTGOrder=FTHG-FTAG+0.01*sign(FTHG-FTAG)*FTHG, FTG=reorder(FTG, FTGOrder),
       uDiff=uFTHG-uFTAG,
       FTDiff=FTHG-FTAG)%>%droplevels()
summary(all_user_data)

n<-all_user_data
all_user_data<-all_user_data%>%group_by(Userid, Username)%>%mutate(Points=sum(uPoints), n=n())%>%ungroup()%>%mutate(Rank=rank(-Points)/n)
all_user_data<-all_user_data%>%group_by(Userid, Username, Round)%>%mutate(DayPoints=sum(uPoints), n=n())%>%ungroup()%>%group_by(Round)%>%mutate(DayRank=rank(-DayPoints)/n)%>%ungroup()
all_user_data%>%dplyr::select(Username, Points, Rank)%>%unique()
all_user_data%>%dplyr::select(Username, Round, DayPoints, DayRank)%>%unique()

user_points<-all_user_data%>%group_by(Userid, Username)%>%summarise(uPoints=sum(uPoints), b=sum(is.na(uFTHG)))
nrow(user_points)
table(user_points$b/6)
hist(user_points$b/6, breaks=30)
summary(user_points)
summary(user_points%>%filter(b==0))
user_fullpoints<-user_points%>%filter(b==0)
hist(user_points$uPoints, breaks=50)
plot(sort(user_points$uPoints))
abline(h=(user_points%>%filter(Username=="TCSNet"))$uPoints, col="red")

hist(user_fullpoints$uPoints, breaks=50)
plot(sort(user_fullpoints$uPoints))
abline(h=(user_points%>%filter(Username=="TCSNet"))$uPoints, col="red")

all_user_data%>%filter(Username=="TCSNet")%>%dplyr::select(uFTR)%>%table()/(132+22+242)

132+22+242 

tgf<-all_user_data%>%
  group_by(Userid, Username)%>%
  summarise(tendency=mean(uPoints>0, na.rm=T),
            full=mean(uPoints>2, na.rm=T)) 
smoothScatter(tgf$tendency, tgf$full)
with(tgf%>%filter(Username=="TCSNet"), points(tendency, full, col="red"))

ggplot(tgf, aes(x=tendency, y=full, col=full))+geom_point()+scale_color_continuous(high = "red", low="green")



ggplot(tgf, aes(x=tendency, y=full, col=full))+geom_point()+geom_jitter(height = 0.01, width=0.01)+scale_color_continuous(high = "red", low="green")

ggplot(tgf, aes(x=tendency))+geom_histogram(bins=50)
ggplot(tgf, aes(x=full))+geom_histogram(bins=30)
table(cut(tgf$tendency, breaks=50))

tgf%>%arrange(tendency, full)%>%print.data.frame()




summary(all_user_data)
all_user_data6 <- all_user_data%>%filter(uFTHG+uFTAG<=6)%>%droplevels()

all_user_data0 <- all_user_data%>%mutate(n=nchar(as.character(uBet)))%>%filter(n==3)%>%droplevels()
  
gridExtra::grid.arrange(
  ggplot(all_user_data6)+
    geom_mosaic(aes(x=product(uBet), fill=factor(uPoints), weight=uPoints), offset=0.003)+
    scale_fill_manual(values = c('darkblue', 'orange',  'red2')),
  ggplot(all_user_data6)+
    geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
    scale_fill_manual(values = c('darkblue', 'orange',  'red2')),
  ncol = 1, nrow = 2)

with(all_user_data0, table(uBet))
with(all_user_data0, table(uBet, uPoints))
with(all_user_data6%>%filter(Username=="TCSNet"), table(FTG))
with(all_user_data6%>%filter(Username=="TCSNet"), table(uBet))
with(all_user_data6%>%filter(Username=="TCSNet"), table(uBet, uPoints))
with(all_user_data, table(FTG, uPoints))

tbids<-with(all_user_data, table(uBet))
str(tbids)
tbids<-as.data.frame(tbids)
tFTG<-with(all_user_data0, table(FTG))
str(tFTG)
tFTG<-as.data.frame(tFTG)
tFTG<-tFTG%>%mutate(t="act", Freq=Freq/sum(Freq))

tbids<-tbids%>%mutate(t="pred", FTG=uBet, Freq=Freq/sum(Freq))%>%dplyr::select(-uBet)

t1<-rbind(tbids, tFTG)
t1$FTG<-reorder(t1$FTG, 1.01*strtoi(substr(t1$FTG,1,1))-strtoi(substr(t1$FTG,3,3)))
t1$FTG<-droplevels(t1$FTG)

ggplot(t1, aes(x=FTG, fill=t))+geom_bar(aes(weight=Freq), position="dodge")

######
pointdist<-all_user_data%>%mutate(color=uPoints, mpoints=uPoints, Name=Username)
pointdist_missed<-all_user_data%>%mutate(color=0, mpoints=5.0-uPoints, Name=Username)
pointdist<-rbind(pointdist, pointdist_missed)

gridExtra::grid.arrange(
ggplot(pointdist%>%mutate(fill=factor(ifelse(uPoints==0, sign(FTHG-FTAG)-3, color))))+
  geom_mosaic(aes(x=product(uFTAG, uFTHG), 
                  fill=fill, 
                  weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c("bisque", 'darkseagreen2', 'lightblue2', 'white', 'orange', 'red2'))
,
ggplot(pointdist%>%filter(Username=='TCSNet')%>%mutate(fill=factor(ifelse(uPoints==0, sign(FTHG-FTAG)-3, color))))+
  geom_mosaic(aes(x=product(uFTAG, uFTHG), 
                  fill=fill, 
                  weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c("bisque", 'darkseagreen2', 'lightblue2', 'white', 'orange', 'red2'))
, ncol = 1, nrow = 2)


ggplot(pointdist%>%filter(Username%in%c('TCSNet', 'jwoli')))+
facet_wrap(aes(paste(Username, Userid, Points)))+
geom_mosaic(aes(x=product(FTG), fill=factor(color), weight=mpoints), offset=0.003)+
scale_fill_manual(values = c('white', 'orange',  'red2'))

ggplot(pointdist%>%filter(Username%in%c('TCSNet', 'jwoli', "Sandra Ba")|Rank<=7))+
  facet_wrap(aes(paste(Points, Name, Userid, Rank)))+
  geom_mosaic(aes(x=product(FTAG, FTHG), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))

ggplot(pointdist%>%filter(Username%in%c('TCSNet', 'jwoli', "Sandra Ba")|Rank<=7))+
  facet_wrap(aes(paste(Points, Name, Userid, Rank)))+
  geom_mosaic(aes(x=product(uFTAG, uFTHG), 
                  fill=factor(ifelse(uPoints==0, sign(FTHG-FTAG)-3, color)), 
                  weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c("bisque", 'darkseagreen2', 'lightblue2', 'white', 'orange', 'red2'))


ggplot(pointdist%>%filter(Username%in%c('TCSNet', 'jwoli', "Sandra Ba")|Rank<=7))+
  facet_wrap(aes(paste(Points, Name, Userid, Rank)))+
  geom_mosaic(aes(x=product(FTR), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))

ggplot(pointdist%>%filter(Username%in%c('TCSNet', 'jwoli', "Sandra Ba")|Rank<=7))+
  facet_wrap(aes(paste(Points, Name, Userid, Rank)))+
  geom_mosaic(aes(x=product(FTDiff), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))

ggplot(pointdist%>%filter(!is.na(uFTHG)))+
  facet_wrap(aes(Username%in%c('TCSNet')))+
  geom_mosaic(aes(x=product(FTDiff), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))

ggplot(pointdist%>%filter(!is.na(uFTHG)))+
  facet_wrap(aes(Username%in%c('TCSNet')))+
  geom_mosaic(aes(x=product(FTR), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))

ggplot(pointdist%>%filter(!is.na(uFTHG)))+
  facet_wrap(aes(Username%in%c('TCSNet')))+
  geom_mosaic(aes(x=product(FTAG, FTHG), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))
  
ggplot(pointdist%>%filter(!is.na(uFTHG)))+
  facet_wrap(aes(Username%in%c('TCSNet')))+
  geom_mosaic(aes(x=product(FTG), fill=factor(color), weight=mpoints), offset=0.003, color="black")+
  scale_fill_manual(values = c('white', 'orange', 'red2'))



str(all_user_data)

ggplot(all_user_data%>%filter(!is.na(uFTHG))%>%group_by(Userid, Round)%>%mutate(total=sum(uPoints)))+
  facet_wrap(aes(Round))+
  geom_histogram(aes(x=total, fill=factor(uPoints)), breaks=seq(0, 30, by=1))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))+
  geom_vline(data = all_user_data%>%filter(Username=='TCSNet')%>%group_by(Round)%>%summarise(uPoints=sum(uPoints)), 
                 aes(xintercept=uPoints+0.5), col="red")

ggplot(all_user_data6%>%filter(Round==33))+
  facet_wrap(aes(HomeTeam, AwayTeam, result=FTG))+
  geom_bar(aes(x=uBet, fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))
  


gridExtra::grid.arrange(
  ggplot(all_user_data)+geom_mosaic(aes(x=product(uFTR), fill=factor(uPoints)))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data)+geom_mosaic(aes(x=product(uFTR), fill=factor(uPoints), weight=uPoints))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ncol = 1, nrow = 2)

gridExtra::grid.arrange(
  ggplot(all_user_data6)+geom_bar(aes(x=uBet, fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))
,
ggplot(all_user_data0)+geom_bar(aes(x=FTG, fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))
,  ncol = 1, nrow = 2)


ggplot(all_user_data6%>%filter(Rank<=2))+
  geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))


ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"))+
  geom_mosaic(aes(x=product(uFTR), fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"))+
  geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_bar(aes(x=uBet, fill=factor(uPoints), weight=ifelse(Username=="TCSNet", 1, 1/length((-1+unique(Userid))))))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_bar(aes(x=uBet, fill=factor(uPoints), weight=uPoints*ifelse(Username=="TCSNet", 1, 1/length((-1+unique(Userid))))))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_bar(aes(x=uFTHG-uFTAG, fill=factor(uPoints), weight=ifelse(Username=="TCSNet", 1, 1/length((-1+unique(Userid))))))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_bar(aes(x=uFTHG-uFTAG, fill=factor(uPoints), weight=uPoints*ifelse(Username=="TCSNet", 1, 1/(-1+length(unique(Userid))))))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(uFTHG-uFTAG), nrow=1)+
  geom_bar(aes(x=Username=="TCSNet", fill=factor(uPoints), weight=uPoints*ifelse(Username=="TCSNet", 1, 1/(-1+length(unique(Userid))))))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Rank>=500|Username=="TCSNet"))+facet_wrap(aes(uFTR), nrow=1)+
  geom_bar(aes(x=Username=="TCSNet", fill=factor(uPoints), weight=uPoints*ifelse(Username=="TCSNet", 1, 1/(-1+length(unique(Userid))))))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_mosaic(aes(x=product(uFTR, FTR), fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+#facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_mosaic(aes(x=product(uBet, FTG), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Username=="TCSNet"))+
  geom_mosaic(aes(x=product(uBet, FTG), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Username=="TCSNet"))+
  geom_mosaic(aes(x=product(uBet, FTG), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6)+facet_wrap(aes(Username=="TCSNet"), ncol=1)+
  geom_mosaic(aes(x=product(uDiff, FTDiff), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))


all_user_data6%>%filter(Username=="TCSNet")%>%dplyr::select(FTG:uFTR)%>%group_by(uBet)%>%count()

ggplot(all_user_data6)+facet_wrap(aes(round(log(Rank))))+
  geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(X<=6|Userid%in%c(218206, 10)))+facet_wrap(aes(paste(Rank, Name, Points)))+
  geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Rank<=10|Userid%in%c(218206, 10)))+facet_wrap(aes(paste(Rank, Name, Points)))+
  geom_mosaic(aes(x=product(uFTR), fill=factor(uPoints)), offset=0.003)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Rank<=10|Userid%in%c(218206, 10))%>%droplevels())+facet_wrap(aes(uFTR), ncol = 1)+
  geom_mosaic(aes(x=product(Rank), fill=factor(uPoints)), offset=0.005)+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Rank<=10|Userid%in%c(218206, 10))%>%droplevels())+facet_wrap(aes(paste(Rank, Name, Points)))+
  geom_bar(aes(x=uBet, fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))

ggplot(all_user_data6%>%filter(Userid%in%c(218206, 10, 201945, 202994, 32382))%>%droplevels())+facet_wrap(aes(paste(Rank, Name, Points)))+
  geom_bar(aes(x=uBet, fill=factor(uPoints)))+
  scale_fill_manual(values = c('darkblue', 'orange', 'red2'))


gridExtra::grid.arrange(
  ggplot(all_user_data6%>%filter(Rank==1))+facet_wrap(aes(Userid))+
    geom_mosaic(aes(x=product(uBet), fill=factor(uPoints), weight=uPoints), offset=0.003)+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data6%>%filter(Rank==2))+
    geom_mosaic(aes(x=product(uBet), fill=factor(uPoints), weight=uPoints), offset=0.003)+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ncol = 1, nrow = 2)

gridExtra::grid.arrange(
  ggplot(all_user_data6%>%filter(Rank<=2))+
    geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data6%>%filter(Rank==2))+
    geom_mosaic(aes(x=product(uBet), fill=factor(uPoints)), offset=0.003)+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ncol = 1, nrow = 2)


gridExtra::grid.arrange(
  ggplot(all_user_data)+geom_mosaic(aes(x=product(myBet), fill=factor(myPoints), weight=myPoints))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data)+geom_mosaic(aes(x=product(psBet), fill=factor(psPoints), weight=psPoints))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ncol = 1, nrow = 2)

gridExtra::grid.arrange(
  ggplot(all_user_data)+geom_mosaic(aes(x=product(myFTR), fill=factor(myPoints)))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data)+geom_mosaic(aes(x=product(myFTR), fill=factor(myPoints), weight=myPoints))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data)+geom_mosaic(aes(x=product(psFTR), fill=factor(psPoints)))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data)+geom_mosaic(aes(x=product(psFTR), fill=factor(psPoints), weight=psPoints))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ncol = 2, nrow = 2)


gridExtra::grid.arrange(
  ggplot(all_user_data)+geom_bar(aes(x=myBet, fill=factor(myPoints)))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ggplot(all_user_data)+geom_bar(aes(x=psBet, fill=factor(psPoints)))+
    scale_fill_manual(values = c('darkblue', 'orange', 'red2')),
  ncol = 1, nrow = 2)


