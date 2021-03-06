# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:44 2018

@author: 811087
"""
from bs4 import BeautifulSoup
import pandas as pd
#from urllib import request
#from urllib3 
import requests
import re
baseurl='https://6erpack.sky.de'
import os

os.chdir("D:/gitrepository/Football/football/TF")

def load_page(session, url, params):
  print(url, params)
  page = session.get(url, params=params)
  return page.content
    

def login():
  session = requests.Session()
#  session.auth=('martin.storch@tcs.com', 'Rebekka95')
  response = session.get(baseurl)
  
  html_content = load_page(session, baseurl+'/konto/login', {})
    
  soup = BeautifulSoup(html_content, 'html.parser')
  token = soup.find('input', attrs={'type':'hidden', 'name':'_csrf_token'})
  csrf_token = token["value"]
  print(csrf_token )

  response = session.post(baseurl+"/konto/login", 
      data = {
      'email': 'martin.storch@tcs.com',
      'password': 'Rebekka95',
      '_csrf_token':csrf_token 
      }, 
      allow_redirects=True)
  
  print(response.headers)
  print(response.url)
  print(response.history)
  print(response.cookies)
  #print(response.content)
  print(response)
  return session



def collect_ranking(session, type_, values=[0]):
  ranks=[]
  names=[]
  userids=[]
  points=[]
  fullhit=[]
  tendency=[]
  kind=[]
  rounds=[]
  for v in values:
    url = baseurl+'/rangliste/'+type_
    if type_ != 'saison':
      url += '/'+str(v)
    html_content = load_page(session, url, params={'leaderboard_scroll':'true'})
    
    soup = BeautifulSoup(html_content, 'html.parser')
    start = soup.find('div', attrs={'class':'leagues'})
    
    for tr in start.find_all('tr', attrs={"class":"leaderboard__row leaderboard__row--non-winner"}):
      td = tr.find_all('td')
      rank = td[1].text.strip()
      if rank=='' and td[1].find('span', attrs={'class':'super6-winning-player'})!=None:
        rank='1'
      ranks.append(int(rank))

      fullhit.append(int(td[4].text.strip()))
      tendency.append(int(td[3].text.strip()))
      points.append(int(td[5].text.strip()))
      
      link = td[2].find('a')
      names.append(link.text.strip())
      userids.append(link["href"].split("/")[-1])
      kind.append(type_)
      rounds.append(v)
      
  ranking_data = pd.DataFrame({
    'Type':kind,
    'Round':rounds,
    'Rank':ranks,
    'Name':names,
    'Userid':userids,
    'Points':points,
    'Fullhit':fullhit,
    'Tendency':tendency
  })
  return ranking_data

def collect_user_tipps(session, userid, roundfrom=1, roundto=34, early_stop=False):

  rounds=[]
  users=[]
  names=[]
  hometeams=[]
  awayteams=[]
  FTHGs=[]
  FTAGs=[]
  myFTHGs=[]
  myFTAGs=[]
  myPoints=[]
  missed_counter = 0
  for i in range(roundfrom, roundto+1):
    url = baseurl+'/ergebnisse/spieltag/'+str(i)+'/benutzer/'+str(userid)
    html_content = load_page(session, url, {})
    
    soup = BeautifulSoup(html_content, 'html.parser')
#    spieltag = soup.find('h3', attrs={'class':'ressort'})
    header = soup.find('div', attrs={"id":"round-select"})
    name = header.find('div', attrs={'class':'text--h1'})
    if name is None:
      name = "TCSNet"
    else:
      name = name.text.strip()
    
    table = soup.find('div', attrs={'class':'predictions'})
    
    for tr in table.find_all('div', attrs={'class':'match-details'}):
      td = tr.find_all('div')
      hometeam = td[0].find('p', class_="team-name").text.strip()
      awayteam = td[-1].find('p', class_="team-name").text.strip()
      hometeams.append(hometeam)
      awayteams.append(awayteam)

      scores = tr.find_all('div', class_="score--full-time")
      
      FTHG = scores[-2].text.strip()
      FTAG = scores[-1].text.strip()
      FTHGs.append(int(FTHG))
      FTAGs.append(int(FTAG))
    
      #FTR = td[2].text.strip()
      FTHG = scores[0].text.strip()
      FTAG = scores[1].text.strip()
      if FTHG=="-":
        missed_counter = missed_counter-1
      else:
        missed_counter = missed_counter+1
      if early_stop and missed_counter<0:
        return pd.DataFrame()
      
      myFTHGs.append(int(FTHG) if FTHG!="-" else None)
      myFTAGs.append(int(FTAG) if FTAG!="-" else None)
      
      points = tr.find_next_sibling('div', class_="pill--red")
      if points is None:
        points = "0"
      else:
        points = points.text.strip().split(" ")[0]
      myPoints.append(int(points) if points!="-" else None)
      
      rounds.append(i)
      users.append(userid)
      names.append(name)
      
  sky_data = pd.DataFrame({
    'Round':rounds,
    'Userid':users,
    'Username':names,
    'HomeTeam':hometeams,
    'AwayTeam':awayteams,
    'FTHG':FTHGs,
    'FTAG':FTAGs,
    'uFTHG':myFTHGs,
    'uFTAG':myFTAGs,
    'uPoints':myPoints,
  })
  return   sky_data



session = login()
r1 = collect_ranking(session, 'saison') 
r2 = collect_ranking(session, 'spieltag', range(1, 35))
r3 = collect_ranking(session, 'monat', [8,9,10,11,12,1,2,3,4,5])

all_rankings = pd.concat([r1, r2, r3], axis=0, ignore_index=True)


all_rankings.to_csv("sky_ranking_data_final.csv", encoding = "utf-8", index=True)
print(all_rankings)

all_rankings = pd.read_csv("sky_ranking_data.csv", encoding = "utf-8")

all_user_data = pd.DataFrame()
all_user_data = pd.read_csv("sky_user_tipps.csv", encoding = "utf-8")
known_users = all_user_data.Userid.unique()

for i,userid in enumerate(all_rankings.Userid.unique()):
  if int(userid) not in known_users:
    print(i)
    user_data = collect_user_tipps(session, roundfrom=1, roundto=34, userid=userid)
    all_user_data = pd.concat([all_user_data, user_data], axis=0, ignore_index=True)
    if i%50==0:
      all_user_data.to_csv("sky_user_tipps.csv", encoding = "utf-8", index=False)

all_user_data.to_csv("sky_user_tipps.csv", encoding = "utf-8", index=False)

session = login()

known_users = all_user_data.Userid.unique()
len(known_users)
j=1
for i,userid in enumerate(range(30000,40000)):
  if userid not in known_users:
    user_data = collect_user_tipps(session, roundfrom=1, roundto=34, userid=userid, early_stop=True)
    if user_data.shape[0]>0:
      j=j+1
      all_user_data = pd.concat([all_user_data, user_data], axis=0, ignore_index=True)
      if j%100==0:
        all_user_data.to_csv("sky_user_tipps.csv", encoding = "utf-8", index=False)
    print(i,j)

all_user_data.to_csv("sky_user_tipps.csv", encoding = "utf-8", index=False)

for i,userid in enumerate(known_users):
  print(i)
  user_data = collect_user_tipps(session, roundfrom=34, roundto=34, userid=userid, early_stop=True)
  all_user_data = pd.concat([all_user_data, user_data], axis=0, ignore_index=True)
  if i%500==0:
    all_user_data.to_csv("sky_user_tipps.csv", encoding = "utf-8", index=False)
all_user_data.to_csv("sky_user_tipps.csv", encoding = "utf-8", index=False)

#pistor_data.to_csv("pistor_data.csv", encoding = "utf-8", index=False)


####################################################################################################

#import ssl
#
#context = ssl.create_default_context()
#der_certs = context.get_ca_certs(binary_form=True)
#pem_certs = [ssl.DER_cert_to_PEM_cert(der) for der in der_certs]
#
#with open('wincacerts.pem', 'w') as outfile:
#    for pem in pem_certs:
#        outfile.write(pem + '\n')



