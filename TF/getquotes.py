# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:44 2018

@author: 811087
"""
from bs4 import BeautifulSoup
import pandas as pd
from urllib import request

import ssl

context = ssl.create_default_context()
der_certs = context.get_ca_certs(binary_form=True)
pem_certs = [ssl.DER_cert_to_PEM_cert(der) for der in der_certs]

with open('wincacerts.pem', 'w') as outfile:
    for pem in pem_certs:
        outfile.write(pem + '\n')

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

import ssl

context = ssl.create_default_context()
der_certs = context.get_ca_certs(binary_form=True)
pem_certs = [ssl.DER_cert_to_PEM_cert(der) for der in der_certs]

with open('wincacerts.pem', 'w') as outfile:
    for pem in pem_certs:
        outfile.write(pem + '\n')

#print(pem_certs)
        
url = "https://www.betbrain.de/football/germany/"
#html_content = request.urlopen(url, context=ssl.SSLContext()).read()
html_content = request.urlopen(url).read()

with open("betbrain.html", "wb+") as f:
  f.write(html_content)
 
#text_content1 = html_content.decode('unicode_escape')  # Converts bytes to unicode
#with open("books_utf8.html", "w+") as f:
#  f.write(text_content1)

soup = BeautifulSoup(html_content, 'html.parser')

over = []
under = []
overavg = []
underavg = []
comparison = []
hometeams = []
awayteams = []
urls = []
matchdates = []
matchtimes = []

sections = soup.find_all(string="Unter")
for s in sections:
  header = s.find_parent('h3', class_="TheDayTitle")
  matchlist = header.find_next_sibling("ol", class_="TheMatches")
  
  matches= matchlist.find_all('ol', attrs={'class':'ThreeWay'})

  for m in matches:
    out1 = m.find("li", attrs={'class':'Outcome1'}).find("a")
    q_over = out1.find("span", attrs={'class':'Odds'}).get_text()
    q_overavg = out1.find("span", attrs={'class':'AverageOdds'}).get_text()[1:-1]
      
    out3 = m.find("li", attrs={'class':'Outcome3'}).find("a")
    q_under = out3.find("span", attrs={'class':'Odds'}).get_text()
    q_underavg = out3.find("span", attrs={'class':'AverageOdds'}).get_text()[1:-1]
  
    out2 = m.find("li", attrs={'class':'Outcome2'})
    q_comparison= out2.find("span", attrs={'class':'Param'}).get_text()
  
    over.append(q_over.replace(",", "."))
    under.append(q_under.replace(",", "."))
    overavg.append(q_overavg.replace(",", "."))
    underavg.append(q_underavg.replace(",", "."))
    comparison.append(q_comparison)

    md = out1.find_parent("ol", class_="ThreeWay").find_previous_sibling("div", class_="MatchDetails")
    q_match = md.div.a.span.text.split(" - ")
    hometeams.append(q_match[0])
    awayteams.append(q_match[1])
    urls.append(md.a["href"])
    matchdate = md.find("span", attrs={"class":"DateTime", "itemprop":"startDate"}).get_text()
    md = matchdate.split(' ')
    matchdates.append(md[0])
    matchtimes.append(md[1])
    
overunder_df = pd.DataFrame({
  "Date": matchdates,
  "Time": matchtimes,
  "HomeTeam": hometeams,
  "AwayTeam": awayteams,
  "Over": over,
  "Param": comparison,
  "Under": under,
  "OverAvg": overavg,
  "UnderAvg": underavg,
  "URL": urls,
})
print(overunder_df)

overunder_df["margin"] = 1/overunder_df.Over.astype(float)+1/overunder_df.Under.astype(float)
overunder_df["marginavg"] = 1/overunder_df.OverAvg.astype(float)+1/overunder_df.UnderAvg.astype(float)


