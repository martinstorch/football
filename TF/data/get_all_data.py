# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:44 2018

@author: 811087
"""
from bs4 import BeautifulSoup
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
#import numpy as np
import re
import json
from urllib import request
import os 
import ssl
from datetime import datetime, date, time, timedelta
from requests_html import HTMLSession
import re
import sys
import time

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
  xgoals= soup.find_all('script')[1].string
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
  
  #url='https://sports.bwin.com/de/sports/4/43/wetten/bundesliga#leagueIds=43&sportId=4'
  url='https://sports.bwin.de/de/sports/fu%C3%9Fball-4/wetten/deutschland-17/bundesliga-43'
  #html_content = request.urlopen(url).read()
  
  #text_content1 = html_content.decode('unicode_escape')  # Converts bytes to unicode
  #with open("books_utf8.html", "w+") as f:
  #  f.write(text_content1)
  #"C:\Users\marti\AppData\Local\Google\Chrome SxS\Application\chrome.exe"

  import os
  from selenium import webdriver
  from selenium.webdriver.common.keys import Keys
  from selenium.webdriver.chrome.options import Options

  chrome_options = Options()
  #chrome_options.add_argument("--headless")
  #chrome_options.binary_location = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'
  chrome_options.binary_location = 'C:/Users/marti/AppData/Local/Google/Chrome SxS/Application'
  chrome_options.binary_location = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"

  #driver = webdriver.Chrome(executable_path=os.path.abspath("chromedriver"),   chrome_options=chrome_options)
  #driver = webdriver.Chrome(executable_path='C:/Users/marti/AppData/Local/Google/Chrome SxS/Application/chromedriver.exe',   chrome_options=chrome_options)
  driver = webdriver.Chrome(executable_path="C:/git/football/TF/chromedriver96/chromedriver.exe",   chrome_options=chrome_options)
  driver.set_page_load_timeout(30000)
  driver.set_script_timeout(30000)
  driver.get(url)
  time.sleep(5)
  html = driver.execute_script("return document.documentElement.outerHTML")
  with open("bwin.html", "w+", encoding="utf-8") as f:
      f.write(html)
  if re.search("gladbach", html) is None:
      print("Rendering not successful")
      print(url)
      sys.exit()
  #print(html)

  if False:
      session = HTMLSession()
      r = session.get(url, timeout=10000)
      r.html.render(timeout=10000)
      print(r.html.html)
      script = """
             () => {
                    $(document).ready(function() {  
                         ## $("span.ui-icon.theme-ex").click();
                    })
              }
               """
      #r.html.render(script=script, reload=False)
      #print(r.html.html)
      #div class="content-message-container"
      if re.search("gladbach", r.html.html) is None:
        print("Rendering not successful")
        print(url)
        sys.exit()

      with open("bwin.html", "w+", encoding="utf-8") as f:
        f.write(r.html.html)
  with open("bwin.html", "r", encoding="utf-8") as f:
    html_content = f.read()    
  soup = BeautifulSoup(html_content, 'html.parser')
  quotes = soup.find_all('a', attrs={'class':'grid-event-wrapper'})
  if len(quotes)==0:
      quotes = soup.find_all('div', attrs={'class':'grid-event-wrapper'})
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
    datetm = matchtimer.get_text().strip().split()
    if len(datetm)>2:
      datetm = matchtimer.get_text().strip().split(" / ")
      print(datetm)
    mdate, timestr = datetm

    pdiv = q.find_previous('ms-date-header', attrs={'class':'date-group'})
    span = pdiv.find('div')
    matchdate = span.get_text().strip()
    dowdate = matchdate.split(' - ')
    if len(dowdate)<2:
      dowdate = matchdate.split(' / ')
      print(dowdate)
    if matchdate=="Morgen":
      mdate = datetime.now() + timedelta(days=1)
      print(mdate)
      dow = mdate.strftime('%A')
      print(dow)
      mdate = mdate.strftime('%d.%m.%Y')
    elif matchdate == "Heute":
      mdate = datetime.now()
      print(mdate)
      dow = mdate.strftime('%A')
      print(dow)
      mdate = mdate.strftime('%d.%m.%Y')
    else:
      dow, mdate = dowdate
      mdate = datetime.strptime(mdate, '%d.%m.%y').strftime('%d.%m.%Y') # convert two-digit year to four-digit
      
    print(mdate)
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


def download_betonjamesdata(model_dir, season, skip_download, prefix="DE", league="germany-bundesliga"):
    """Maybe downloads training data and returns train and test file names."""
    file_name = dir_path + "/boj_" + prefix + "_" + season + ".csv"
    print(file_name)
    if (not skip_download):
        url = "https://www.betonjames.com/wp-content/uploads/boj-free-data-downloads/"+ league + "-20"+season[:2]+"-20"+season[2:] + ".csv"
        print("Downloading %s" % url)
        request.urlretrieve(
            url,
            file_name)  # pylint: disable=line-too-long
        print("Data is downloaded to %s" % file_name)
    data = pd.read_csv(
        file_name,
        skipinitialspace=True,
        engine="python",
        encoding = "utf-8",
        skiprows=0)
    data["Season"] = season
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
    xg_season_list=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    xgdf=pd.DataFrame()
    for s in xg_season_list:
      xg = load_expected_goals(s)
      print(xg.shape)
      xgdf = pd.concat([xgdf, xg], ignore_index=True)
    xgdf.to_csv(dir_path+'/xgoals.csv', index=False)  
print(xgdf)

all_seasons = ["0910", "1011", "1112", "1213", "1314","1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]
boj_data = []
for s in all_seasons[1:]:
  sdata = download_betonjamesdata(dir_path, s, skip_download=skip_download)
  #print(sdata.columns.values)
  sdata["Predict"]=False
  boj_data.append(sdata)
boj_data = pd.concat(boj_data, ignore_index=True)
boj_data = boj_data[['Season' ,'Predict',"DATE","HOME_TEAM","AWAY_TEAM","FTHG","FTAG","FTR","HTHG","HTAG","HTR","H_ST","H_SOG","H_SFG","H_PT","H_COR","H_FL","H_YC","H_RC","A_ST","A_SOG","A_SFG","A_PT","A_COR","A_FL","A_YC","A_RC"]]
boj_data.H_PT.replace('SNV', -1, inplace=True)
boj_data.A_PT.replace('SNV', -1, inplace=True)
print(boj_data.HOME_TEAM.unique())
# standardize team names
boj_team_mapping= pd.read_csv(dir_path+"/boj_team_mapping.csv", sep='\t')
boj_data= boj_data.merge(boj_team_mapping, left_on="HOME_TEAM", right_on="bojTeam", how="left")
boj_data = boj_data.merge(boj_team_mapping, left_on="AWAY_TEAM", right_on="bojTeam", how="left")
boj_data = boj_data.rename(columns={"stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam"})
boj_data["Date"]=pd.to_datetime(boj_data.DATE).apply(lambda x: x.strftime('%d.%m.%Y'))
boj_data.drop(columns = ['DATE', 'bojTeam_x', 'bojTeam_y'], inplace=True)
print(boj_data[-9:])

cup_data = []
for s in all_seasons[1:]:
    sdata = download_betonjamesdata(dir_path, s, skip_download=skip_download, prefix="CL", league="europe-champions-league")
    sdata["Predict"]=False
    cup_data.append(sdata)
    sdata = download_betonjamesdata(dir_path, s, skip_download=skip_download, prefix="EL", league="europe-europa-league")
    sdata["Predict"] = False
    cup_data.append(sdata)
cup_data = pd.concat(cup_data, ignore_index=True)
cup_data = cup_data[['Season' ,'Predict',"LEAGUE", "DATE","HOME_TEAM","AWAY_TEAM","FTHG","FTAG","FTR","HTHG","HTAG","HTR","ETR", "ETHG", "ETAG", "PENR", "PENHG", "PENAG", "H_ST","H_SOG","H_SFG","H_PT","H_COR","H_FL","H_YC","H_RC","A_ST","A_SOG","A_SFG","A_PT","A_COR","A_FL","A_YC","A_RC"]]
cup_data.H_PT.replace('SNV', -1, inplace=True)
cup_data.A_PT.replace('SNV', -1, inplace=True)
cup_data.replace('N/A', -1, inplace=True)
cup_data.fillna(-1, inplace=True)
cup_data = cup_data.astype({'HTHG':int, 'HTAG':int, 'ETHG':int, 'ETAG':int, 'PENHG':int, 'PENAG':int})

# standardize team names
cup_data= cup_data.merge(boj_team_mapping, left_on="HOME_TEAM", right_on="bojTeam", how="left")
cup_data = cup_data.merge(boj_team_mapping, left_on="AWAY_TEAM", right_on="bojTeam", how="left")
cup_data = cup_data.rename(columns={"stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam"})
cup_data = cup_data.loc[~pd.isna(cup_data.HomeTeam) | ~pd.isna(cup_data.AwayTeam)]
cup_data.HomeTeam.loc[pd.isna(cup_data.HomeTeam)] = cup_data.LEAGUE.loc[pd.isna(cup_data.HomeTeam)]
cup_data.AwayTeam.loc[pd.isna(cup_data.AwayTeam)] = cup_data.LEAGUE.loc[pd.isna(cup_data.AwayTeam)]

cup_data["Date"]=pd.to_datetime(cup_data.DATE).apply(lambda x: x.strftime('%d.%m.%Y'))
cup_data.drop(columns = ['DATE', 'bojTeam_x', 'bojTeam_y'], inplace=True)
cup_data.rename(columns ={"H_ST":"HS", "A_ST":"AS", "H_SOG":"HST", "A_SOG":"AST", "H_COR":"HC", "A_COR":"AC",
                          "H_FL":"HF", "A_FL":"AF", "H_YC":"HY", "A_YC":"AY", "H_RC":"HR", "A_RC":"AR" }, inplace=True)
cup_data.replace('N/A', -1, inplace=True)
cup_data.replace('SNV', -1, inplace=True)
cup_data.fillna(-1, inplace=True)
#print(cup_data[-36:])
#print(cup_data.HOME_TEAM.unique())
#print(cup_data.HomeTeam.unique())


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
spidf= spidf.loc[spidf.league.isin(["German Bundesliga", "UEFA Champions League", "UEFA Europa League", "UEFA Europa Conference League"])].drop(columns=["league_id"]) #
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
#print((spidf.loc[pd.isna(spidf.HomeTeam) & ~pd.isna(spidf.AwayTeam), ["HomeTeam", "AwayTeam", "Date", "team1", "team2", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA"]]))
#print((spidf.loc[pd.isna(spidf.AwayTeam) & ~pd.isna(spidf.HomeTeam), ["HomeTeam", "AwayTeam", "Date", "team1", "team2", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA"]]))
spidf.HomeTeam.loc[pd.isna(spidf.HomeTeam) & ~pd.isna(spidf.AwayTeam) & spidf.league.eq("UEFA Champions League")] = 'Champions League'
spidf.AwayTeam.loc[~pd.isna(spidf.HomeTeam) & pd.isna(spidf.AwayTeam) & spidf.league.eq("UEFA Champions League")] = 'Champions League'
spidf.HomeTeam.loc[pd.isna(spidf.HomeTeam) & ~pd.isna(spidf.AwayTeam) & spidf.league.eq("UEFA Europa League")] = 'Europa League'
spidf.AwayTeam.loc[~pd.isna(spidf.HomeTeam) & pd.isna(spidf.AwayTeam) & spidf.league.eq("UEFA Europa League")] = 'Europa League'

team_mapping_bwin= pd.read_csv(dir_path+"/bwin_team_mapping.csv", sep="\t")
quotes_bwin = pd.read_csv(dir_path+"/all_quotes_bwin.csv", encoding = "utf-8")
quotes_bwin= quotes_bwin.merge(team_mapping_bwin, left_on="HomeTeam", right_on="bwinTeam", how="left").drop(columns="bwinTeam")
quotes_bwin= quotes_bwin.merge(team_mapping_bwin, left_on="AwayTeam", right_on="bwinTeam", how="left").drop(columns="bwinTeam")
quotes_bwin = quotes_bwin.rename(columns=
                   {"HomeTeam":"HomeTeam_bwin", "AwayTeam":"AwayTeam_bwin",
                    "stdTeam_x":"HomeTeam", "stdTeam_y":"AwayTeam"})
quotes_bwin ["Predict"]=True
quotes_bwin ["Season"]="2122"
quotes_bwin ["Dow"]= pd.to_datetime(quotes_bwin.Date, dayfirst=True).apply(lambda x: x.strftime('%A'))

print(quotes_bwin.drop(columns=["HomeTeam", "AwayTeam"]))

full_data = all_data[['Date', 'Season', 'Predict', 'HomeTeam', 'AwayTeam',
                      'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
                      'HTR', 'HS', 'AS', 'HST', 'AST', 
                      'HF', 'AF', 'HC', 'AC', 
                      'HY', 'AY', 'HR', 'AR',
                      'BWH', 'BWD', 'BWA',
                      ]].copy()
full_data["Date"]= pd.to_datetime(full_data.Date, dayfirst=True).apply(lambda x: x.strftime('%d.%m.%Y'))

full_data = full_data.merge(xgdf[["HomeTeam", "AwayTeam", "Date", "Time", "xHG" , "xAG"]], how="left", on=["HomeTeam", "AwayTeam", "Date"])

full_data = full_data.merge(boj_data[["HomeTeam", "AwayTeam", "Date", "H_PT" , "A_PT"]], how="left", on=["HomeTeam", "AwayTeam", "Date"])
full_data = pd.concat([full_data, cup_data], ignore_index=True) # add cup data to league play data

full_data = full_data.merge(spidf[["HomeTeam", "AwayTeam", "Date", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA", "HGFTa", "AGFTa", "Hxsg", "Axsg", "Hxnsg", "Axnsg"]],
                            how="left", on=["HomeTeam", "AwayTeam", "Date"])

full_data["Dow"]= pd.to_datetime(full_data.Date, dayfirst=True).apply(lambda x: x.strftime('%A'))

new_data = quotes_bwin[["HomeTeam", "AwayTeam", "Date", "Time", "Dow", "BWH" , "BWD", "BWA", "Season", "Predict"]]
new_data = new_data.merge(spidf[["HomeTeam", "AwayTeam", "Date", "Hspi", "Aspi", "Himp", "Aimp", "HGFTe", "AGFTe", "ppH", "ppD", "ppA"]], how="left", on=["HomeTeam", "AwayTeam", "Date"])
full_data = pd.concat([full_data, new_data])
full_data.info()
full_data.fillna(-1, inplace=True)
full_data = full_data.astype({'FTHG':int, 'FTAG':int,
                              'HTHG':int, 'HTAG':int,
                        'H_PT':int, 'A_PT':int,
                       'HS':int, 'AS':int, 'HST':int, 'AST':int,
                       'HF':int, 'AF':int, 'HC':int, 'AC':int,
                       'HY':int, 'AY':int, 'HR':int, 'AR':int,
                              'ETHG':int, 'ETAG':int,
                              'PENHG':int, 'PENAG':int,
                              })
full_data.to_csv(dir_path+"/full_data.csv", index=False)
full_data.info()
