# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:44 2018

@author: 811087
"""
print("hello!")

#import bs4
from bs4 import BeautifulSoup
import pandas as pd
from urllib import request
#url = 'http://coordinated-tray.surge.sh'
url='https://sports.bwin.com/de/sports/4/43/wetten/bundesliga#leagueIds=43&sportId=4'
html_content = request.urlopen(url).read()

with open("bwin.html", "wb+") as f:
  f.write(html_content)
 
#text_content1 = html_content.decode('unicode_escape')  # Converts bytes to unicode
#with open("books_utf8.html", "w+") as f:
#  f.write(text_content1)

soup = BeautifulSoup(html_content, 'html.parser')
quotes = soup.find_all('table', attrs={'class':'marketboard-event-without-header__markets-list'})

hometeams=[]
awayteams=[]
dates=[]
dows=[]
quote_home=[]
quote_draw=[]
quote_away=[]

for q in quotes:
  pdiv = q.find_parent('div', attrs={'class':'marketboard-event-group__item--sub-group'})
  span = pdiv.find('span', attrs={'class':'marketboard-event-group__header-content marketboard-event-group__header-content--level-3'})
  matchdate = span.get_text().strip()
  dow, mdate = matchdate.split(' - ')
  tr = q.find('tr')
  td102 = tr.find_all('td')
  for td,hda in zip(td102, ['H','D','A']):
    bt = td.find('button')
    team = bt.find('div', attrs={'class':['mb-option-button__option-name mb-option-button__option-name--odds-4', 'mb-option-button__option-name mb-option-button__option-name--odds-5']}).get_text()
    teamquote = bt.find('div', attrs={'class':'mb-option-button__option-odds'}).get_text()
    
    #print(dow, mdate, team, teamquote)

    if hda=='H':
      dates.append(mdate)
      dows.append(dow)
      hometeams.append(team)
      quote_home.append(teamquote)
    elif hda=='D':
      quote_draw.append(teamquote)
    elif hda=='A':
      awayteams.append(team)
      quote_away.append(teamquote)
    else:
      raise "HDA"+hda

quotes_bwin = pd.DataFrame({
  'Date':dates,
  'HomeTeam':hometeams,
  'AwayTeam':awayteams,
  'BWH':quote_home,
  'BWD':quote_draw,
  'BWA':quote_away,
  'DOW':dows
})
quotes_bwin.to_csv("quotes_bwin.csv", encoding = "utf-8")
quotes_bwin.iloc[0:9].to_csv("quotes_bwin.csv", encoding = "utf-8")
print(quotes_bwin)

####################################################################################################


##url='https://www.bet365.com/#/AC/B1/C1/D13/E109/F16/'
#url='https://s5.sir.sportradar.com/bet365/de/1/season/55017'
#html_content = request.urlopen(url).read()
#
#with open("b365.html", "wb+") as f:
#  f.write(html_content)
# 
##text_content1 = html_content.decode('unicode_escape')  # Converts bytes to unicode
##with open("books_utf8.html", "w+") as f:
##  f.write(text_content1)
#
#soup = BeautifulSoup(html_content, 'html.parser')
#quotes = soup.find_all('table', attrs={'div':'sl-MarketHeaderLabel_Date'})
#
#for q in quotes:
#  print(q.get_text())
#
#hometeams=[]
#awayteams=[]
#dates=[]
#dows=[]
#quote_home=[]
#quote_draw=[]
#quote_away=[]
#
#for q in quotes:
#  pdiv = q.find_parent('div', attrs={'class':'marketboard-event-group__item--sub-group'})
#  span = pdiv.find('span', attrs={'class':'marketboard-event-group__header-content marketboard-event-group__header-content--level-3'})
#  matchdate = span.get_text().strip()
#  dow, mdate = matchdate.split(' - ')
#  tr = q.find('tr')
#  td102 = tr.find_all('td')
#  for td,hda in zip(td102, ['H','D','A']):
#    bt = td.find('button')
#    team = bt.find('div', attrs={'class':['mb-option-button__option-name mb-option-button__option-name--odds-4', 'mb-option-button__option-name mb-option-button__option-name--odds-5']}).get_text()
#    teamquote = bt.find('div', attrs={'class':'mb-option-button__option-odds'}).get_text()
#    
#    #print(dow, mdate, team, teamquote)
#
#    if hda=='H':
#      dates.append(mdate)
#      dows.append(dow)
#      hometeams.append(team)
#      quote_home.append(teamquote)
#    elif hda=='D':
#      quote_draw.append(teamquote)
#    elif hda=='A':
#      awayteams.append(team)
#      quote_away.append(teamquote)
#    else:
#      raise "HDA"+hda
#
#quotes_bwin = pd.DataFrame({
#  'Date':dates,
#  'HomeTeam':hometeams,
#  'AwayTeam':awayteams,
#  'BWH':quote_home,
#  'BWD':quote_draw,
#  'BWA':quote_away,
#  'DOW':dows
#})
#quotes_bwin.to_csv("quotes_bwin.csv")
#print("Done")
#
#
