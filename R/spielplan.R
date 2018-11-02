setwd("D:/Models")
data<-read.csv("spielplan1819.csv")

library(plyr)
library(lubridate)

str(data)

from <- as.character(head(data$Heimmannschaft, 18))
to <- c("Bayern Munich", "Dortmund", "M'gladbach", "Hertha", "Werder Bremen",
        "Freiburg", "Mainz", "Wolfsburg", "Fortuna Dusseldorf", "Schalke 04", 
        "Hoffenheim", "Leverkusen", "RB Leipzig", "Stuttgart", "Ein Frankfurt", 
        "Augsburg", "Hannover", "Nurnberg" )
cbind(from, to)

out<-with(data, data.frame(Date=as.character(Datum), 
      HomeTeam=mapvalues(data$Heimmannschaft, from = from, to = to),
      AwayTeam=mapvalues(data$Gastmannschaft, from = from, to = to),                           
      ID=Spieltag))
out$Date<-as.character(out$Date)
out$Date[1]<-"25.08.2018"
out$Date<-gsub("15:30   (Sat)", "", out$Date, fixed=T)
out$Date<-trimws(out$Date)
out$Date<-sapply(out$Date, function(x) substring(x, nchar(x)-9, nchar(x))) # use last 8 characters only
out$Date<-dmy(out$Date)
out$Date[1:(306-18)]<-out$Date[1:(306-18)]-days(1)
out$Date<-as.character(out$Date, "%d/%m/%y") 
write.csv(out, "all_new_games.csv", row.names = F)
write.csv(out[out$ID==3,], "NewGames.csv", row.names = F)

str(out)
dmy(out$Date)
