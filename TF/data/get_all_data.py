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
import os 
import ssl
from datetime import datetime, date, time

dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

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
  times=[]
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
      timestr = bt.find_previous('div', attrs={'class':'marketboard-event-without-header__market-time'}).get_text()
      
      #print(dow, mdate, team, teamquote)
  
      if hda=='H':
        dates.append(mdate)
        dows.append(dow)
        times.append(timestr)
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



xg_season_list=[2014, 2015, 2016, 2017, 2018, 2019]
xgdf=pd.DataFrame()
for s in xg_season_list:
  xg = load_expected_goals(s)
  print(xg.shape)
  xgdf = pd.concat([xgdf, xg], ignore_index=True)
xgdf.to_csv(dir_path+'/xgoals.csv', index=False)  
print(xgdf)

quotes_bwin = load_bwin_quotes()
quotes_bwin.to_csv(dir_path+"/all_quotes_bwin.csv", encoding = "utf-8")
quotes_bwin.iloc[0:9].to_csv(dir_path+"/quotes_bwin.csv", encoding = "utf-8")
print(quotes_bwin)

all_seasons = ["0910", "1011", "1112", "1213", "1314","1415", "1516", "1617", "1718", "1819", "1920"]
all_data = []
for s in all_seasons:
  sdata = download_data(dir_path, s, skip_download=False) 
  print(sdata.columns.values)
  sdata["Predict"]=False
  all_data.append(sdata)
all_data = pd.concat(all_data, sort=False)
print(all_data.HomeTeam.unique())
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

team_mapping_bwin= pd.read_csv(dir_path+"/bwin_team_mapping.csv", sep="\t")
quotes_bwin = pd.read_csv(dir_path+"/all_quotes_bwin.csv", encoding = "utf-8").drop(columns="Unnamed: 0")
quotes_bwin= quotes_bwin.merge(team_mapping_bwin, left_on="HomeTeam", right_on="bwinTeam", how="left").drop(columns="bwinTeam")
quotes_bwin= quotes_bwin.merge(team_mapping_bwin, left_on="AwayTeam", right_on="bwinTeam", how="left").drop(columns="bwinTeam")
quotes_bwin = quotes_bwin.rename(columns=
                   {"HomeTeam":"HomeTeam_bwin", "AwayTeam":"AwayTeam_bwin",
                    "stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam"})
quotes_bwin ["Predict"]=True
quotes_bwin ["Season"]="1920"
quotes_bwin ["Dow"]= pd.to_datetime(quotes_bwin.Date, dayfirst=True).apply(lambda x: x.strftime('%A'))

print(quotes_bwin)

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

full_data = pd.concat([full_data, quotes_bwin[["HomeTeam", "AwayTeam", "Date", "Time", "Dow", "BWH" , "BWD", "BWA", "Season", "Predict"]]], sort=False)

full_data.to_csv(dir_path+"/full_data.csv", index=False)

