# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:44 2018

@author: 811087
"""
from bs4 import BeautifulSoup
import pandas as pd
#import numpy as np
import re
import json
from urllib import request
import os 
import ssl
from datetime import datetime, date, time
from requests_html import HTMLSession
import re

dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

skip_download = False


def load_expected_goals(season):
  season=str(season)
  url='https://understat.com/league/Bundesliga/'+season
  print(url)
  html_content = request.urlopen(url).read()
  
#  with open("xG"+season+".html", "wb+") as f:
#    f.write(html_content)
   
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

def load_bwin_quotes():
  context = ssl.create_default_context()
  der_certs = context.get_ca_certs(binary_form=True)
  pem_certs = [ssl.DER_cert_to_PEM_cert(der) for der in der_certs]
  
  with open(dir_path+'/wincacerts.pem', 'w') as outfile:
      for pem in pem_certs:
          outfile.write(pem + '\n')
  
  url='https://sports.bwin.com/de/sports/4/43/wetten/bundesliga#leagueIds=43&sportId=4'
  #html_content = request.urlopen(url).read()
  
  #text_content1 = html_content.decode('unicode_escape')  # Converts bytes to unicode
  #with open("books_utf8.html", "w+") as f:
  #  f.write(text_content1)

  session = HTMLSession()
  r = session.get(url)
  r.html.render()
  #print(r.html.html)
  print(re.search("gladbach", r.html.html))
  
  with open("bwin.html", "w+", encoding="utf-8") as f:
    f.write(r.html.html)
  with open("bwin.html", "r", encoding="utf-8") as f:
    html_content = f.read()    
  soup = BeautifulSoup(html_content, 'html.parser')
  quotes = soup.find_all('a', attrs={'class':'grid-event-wrapper'})
  
  hometeams=[]
  awayteams=[]
  dates=[]
  times=[]
  dows=[]
  quote_home=[]
  quote_draw=[]
  quote_away=[]
  
  for q in quotes:
    matchtimer = q.find("ms-prematch-timer") #, attr={"class":"timer-badge"})
    mdate, timestr = matchtimer.get_text().strip().split()

    pdiv = q.find_previous('ms-date-header', attrs={'class':'date-group'})
    span = pdiv.find('div')
    matchdate = span.get_text().strip()
    dow, mdate = matchdate.split(' - ')
    tr = q.find('ms-option-group')
    pa = q.find_all('div', attrs={'class':'participant'})
    hteam = pa[0].get_text().strip()    
    ateam = pa[1].get_text().strip()    
    hometeams.append(hteam)
    awayteams.append(ateam)
    dates.append(mdate)
    dows.append(dow)
    times.append(timestr)
    td102 = tr.find_all('div', attrs={"class":"option"})
    for td,hda in zip(td102, ['H','D','A']):
      teamquote = td.get_text()
      if hda=='H':
        quote_home.append(teamquote)
      elif hda=='D':
        quote_draw.append(teamquote)
      elif hda=='A':
        quote_away.append(teamquote)
      else:
        raise "HDA"+hda
  
  quotes_bwin = pd.DataFrame({
    'Date':dates,
    'Time':times,
    'HomeTeam':hometeams,
    'AwayTeam':awayteams,
    'BWH':quote_home,
    'BWD':quote_draw,
    'BWA':quote_away,
    'DOW':dows
  })
  return quotes_bwin

def download_data(model_dir, season, skip_download):
    """Maybe downloads training data and returns train and test file names."""
    file_name = dir_path + "/" + season + ".csv"
    print(file_name)
    if (not skip_download):    
  #    urllib.request.urlretrieve(
  #        "http://217.160.223.109/mmz4281/"+season+"/D1.csv",
  #        file_name)  # pylint: disable=line-too-long
      
      url = "http://www.football-data.co.uk/mmz4281/"+season+"/D1.csv"
      print("Downloading %s" % url)
      request.urlretrieve(
          url,
          file_name)  # pylint: disable=line-too-long
      print("Data is downloaded to %s" % file_name)
    data = pd.read_csv(
      file_name,
      skipinitialspace=True,
      engine="python",
      skiprows=0)
    data["Season"]= season
    return data

def getFiveThirtyEightData(skip_download):
    url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
    file_name = dir_path + "/spi_matches.csv"
    print(file_name)
    if skip_download:
        print("Skipping %s" % url)
    else:
        print("Downloading %s" % url)
        request.urlretrieve(
              url,
              file_name)  # pylint: disable=line-too-long
        print("Data is downloaded to %s" % file_name)
    data = pd.read_csv(
      file_name,
      skipinitialspace=True,
      engine="python",
      skiprows=0)
    #data["Season"]= season
    return data
    

if skip_download:
    quotes_bwin = pd.read_csv(dir_path+"/all_quotes_bwin.csv", encoding = "utf-8")
else:
    quotes_bwin = load_bwin_quotes()
    quotes_bwin.to_csv(dir_path+"/all_quotes_bwin.csv", encoding = "utf-8", index=False)
    quotes_bwin.iloc[0:9].to_csv(dir_path+"/quotes_bwin.csv", encoding = "utf-8", index=False)
print(quotes_bwin)
if len(quotes_bwin)==0:
    exit()
    
data538 = getFiveThirtyEightData(skip_download)

if skip_download:
    xgdf = pd.read_csv(dir_path+'/xgoals.csv')  
else:
    xg_season_list=[2014, 2015, 2016, 2017, 2018, 2019]
    xgdf=pd.DataFrame()
    for s in xg_season_list:
      xg = load_expected_goals(s)
      print(xg.shape)
      xgdf = pd.concat([xgdf, xg], ignore_index=True)
    xgdf.to_csv(dir_path+'/xgoals.csv', index=False)  
print(xgdf)

all_seasons = ["0910", "1011", "1112", "1213", "1314","1415", "1516", "1617", "1718", "1819", "1920"]
all_data = []
for s in all_seasons:
  sdata = download_data(dir_path, s, skip_download=skip_download) 
  #print(sdata.columns.values)
  sdata["Predict"]=False
  all_data.append(sdata)
all_data = pd.concat(all_data, ignore_index=True)
all_data = all_data[['Season' ,'Predict',"Date","Time","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HTHG","HTAG","HTR","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR","BWH","BWD","BWA","B365H","B365D","B365A"]]
print(all_data.HomeTeam.unique())
print(all_data.iloc[-9:,list(range(22))+[24, 25, 26]])
#print(all_data.columns.values)
#print(set(all_data.columns.values)-set(sdata.columns.values))

# standardize team names
team_mapping= pd.read_csv(dir_path+"/xg_team_mapping.csv").drop(columns="Unnamed: 0")
xgdf= pd.read_csv(dir_path+"/xgoals.csv")
xgdf= xgdf.merge(team_mapping, left_on="HomeTeam", right_on="xgTeam", how="left")
xgdf = xgdf.merge(team_mapping, left_on="AwayTeam", right_on="xgTeam", how="left")
xgdf = xgdf.drop(columns=["xgTeam_x", "xgTeam_y"]).rename(columns=
                   {"HomeTeam":"HomeTeam_xg", "AwayTeam":"AwayTeam_xg",
                    "stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam"})
xgdf["Time"]=pd.to_datetime(xgdf.Date).apply(lambda x: x.strftime('%H:%M'))
xgdf["Date"]=pd.to_datetime(xgdf.Date).apply(lambda x: x.strftime('%d.%m.%Y'))
print(xgdf[-9:])

# standardize team names
spi_team_mapping= pd.read_csv(dir_path+"/spi_team_mapping.csv").drop(columns="Unnamed: 0")
spidf= pd.read_csv(dir_path+"/spi_matches.csv")
spidf= spidf.loc[spidf.league=="German Bundesliga"].drop(columns=["league_id", "league"])
spidf= spidf.merge(spi_team_mapping, left_on="team1", right_on="spiTeam", how="left")
spidf = spidf.merge(spi_team_mapping, left_on="team2", right_on="spiTeam", how="left")
spidf = spidf.drop(columns=["spiTeam_x", "spiTeam_y"]).rename(columns=
                   {"HomeTeam":"HomeTeam_spi", "AwayTeam":"AwayTeam_spi",
                    "stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam", 
                    "score1":"HG", "score2":"AG", 
                    "adj_score1":"HGFTa", "adj_score2":"AGFTa",
                    "spi1":"Hspi", "spi2":"Aspi",
                    "xg1":"Hxsg", "xg2":"Axsg",
                    "nsxg1":"Hxnsg", "nsxg2":"Axnsg",
                    "importance1":"Himp", "importance2":"Aimp",
                    "proj_score1":"HGFTe", "proj_score2":"AGFTe",
                    "prob1":"ppH", "prob2":"ppA", "probtie":"ppD"})
spidf["Date"]=pd.to_datetime(spidf.date).apply(lambda x: x.strftime('%d.%m.%Y'))
first_new_match=min(spidf.loc[pd.isna(spidf.HG)].index)
#print(spidf[first_new_match-9:first_new_match+9].drop(columns=["date","team1","team2" ]))
print((spidf.loc[(first_new_match-9):(first_new_match-1), ["HomeTeam", "AwayTeam", "Date", "HG", "AG", "HGFTa", "AGFTa", "Hxsg", "Axsg","Hxnsg", "Axnsg"]]).astype({"HG":int, "AG":int}))
print()
print((spidf.loc[first_new_match:first_new_match+8, ["HomeTeam", "AwayTeam", "Date", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA"]]))

team_mapping_bwin= pd.read_csv(dir_path+"/bwin_team_mapping.csv", sep="\t")
quotes_bwin = pd.read_csv(dir_path+"/all_quotes_bwin.csv", encoding = "utf-8")
quotes_bwin= quotes_bwin.merge(team_mapping_bwin, left_on="HomeTeam", right_on="bwinTeam", how="left").drop(columns="bwinTeam")
quotes_bwin= quotes_bwin.merge(team_mapping_bwin, left_on="AwayTeam", right_on="bwinTeam", how="left").drop(columns="bwinTeam")
quotes_bwin = quotes_bwin.rename(columns=
                   {"HomeTeam":"HomeTeam_bwin", "AwayTeam":"AwayTeam_bwin",
                    "stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam"})
quotes_bwin ["Predict"]=True
quotes_bwin ["Season"]="1920"
quotes_bwin ["Dow"]= pd.to_datetime(quotes_bwin.Date, dayfirst=True).apply(lambda x: x.strftime('%A'))

print(quotes_bwin.drop(columns=["HomeTeam", "AwayTeam"]))

full_data = all_data[['Date', 'Season', 'Predict', 'HomeTeam', 'AwayTeam',
                      'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
                      'HTR', 'HS', 'AS', 'HST', 'AST', 
                      'HF', 'AF', 'HC', 'AC', 
                      'HY', 'AY', 'HR', 'AR',
                      'BWH', 'BWD', 'BWA',
                      ]].copy()
full_data["Dow"]= pd.to_datetime(full_data.Date, dayfirst=True).apply(lambda x: x.strftime('%A'))
full_data["Date"]= pd.to_datetime(full_data.Date, dayfirst=True).apply(lambda x: x.strftime('%d.%m.%Y'))

full_data = full_data.merge(xgdf[["HomeTeam", "AwayTeam", "Date", "Time", "xHG" , "xAG"]], how="left", on=["HomeTeam", "AwayTeam", "Date"])

full_data = full_data.merge(spidf[["HomeTeam", "AwayTeam", "Date", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA", "HGFTa", "AGFTa", "Hxsg", "Axsg", "Hxnsg", "Axnsg"]], how="left", on=["HomeTeam", "AwayTeam", "Date"])

new_data = quotes_bwin[["HomeTeam", "AwayTeam", "Date", "Time", "Dow", "BWH" , "BWD", "BWA", "Season", "Predict"]]
new_data = new_data.merge(spidf[["HomeTeam", "AwayTeam", "Date", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA"]], how="left", on=["HomeTeam", "AwayTeam", "Date"])
full_data = pd.concat([full_data, new_data])

full_data.to_csv(dir_path+"/full_data.csv", index=False)

