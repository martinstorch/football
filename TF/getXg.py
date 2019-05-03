# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:44 2018

@author: 811087
"""
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from urllib import request

season_list=[2014, 2015, 2016, 2017, 2018]

def load_expected_goals(season):
  season=str(season)
  url='https://understat.com/league/Bundesliga/'+season
  print(url)
  html_content = request.urlopen(url).read()
  
  with open("xG"+season+".html", "wb+") as f:
    f.write(html_content)
   
  #text_content1 = html_content.decode('unicode_escape')  # Converts bytes to unicode
  #with open("books_utf8.html", "w+") as f:
  #  f.write(text_content1)
  
  soup = BeautifulSoup(html_content, 'html.parser')
  xgoals= soup.find_all('script')[1].get_text()
  
  jsontext = re.search(r'\(\'''(.*)\'''\)', xgoals).groups()[0]
  decoded_string = bytes(jsontext, "utf-8").decode("unicode_escape")
  jsonobj = json.loads(decoded_string)
  
  seasons=[]
  hometeams=[]
  awayteams=[]
  FTHGs=[]
  FTAGs=[]
  FTxHGs=[]
  FTxAGs=[]
  dates=[]
  fcH=[]
  fcD=[]
  fcA=[]
  for j in jsonobj:
    if not j['xG']['h'] is None:
      hometeams.append(j['h']['title'])
      awayteams.append(j['a']['title'])
      FTHGs.append(j['goals']['h'])
      FTAGs.append(j['goals']['a'])
      FTxHGs.append(j['xG']['h'])
      FTxAGs.append(j['xG']['a'])
      dates.append(j['datetime'])
      fcH.append(j['forecast']['w'])
      fcD.append(j['forecast']['d'])
      fcA.append(j['forecast']['l'])
      seasons.append(season)
  
  rxgdf = pd.DataFrame({
      'Date':dates,
      'season':seasons,
      'HomeTeam':hometeams,
      'AwayTeam':awayteams,
      'FTHG':FTHGs,
      'FTAG':FTAGs,
      'xHG':FTxHGs,
      'xAG':FTxAGs,
      'pH': fcH,
      'pD': fcD,
      'pA': fcA,
      })
  return rxgdf

xgdf=pd.DataFrame()
for s in season_list:
  xg = load_expected_goals(s)
  print(xg.shape)
  xgdf = pd.concat([xgdf, xg], ignore_index=True)
xgdf.to_csv('xgoals.csv', index=False)  
print(xgdf)

