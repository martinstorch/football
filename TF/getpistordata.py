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
baseurl='https://allegegenpistor.wdr2.de'
import os

os.chdir("D:/gitrepository/Football/football/TF")

def login():
  session = requests.Session() 
  response = session.post(baseurl, data = {
      'login' : '1',
      'username': 'TCSNet',
      'password': 'Rebekka95'
      })
  
  print(response.headers)
  return session

def load_page(session, url, params):
  print(url, params)
  page = session.get(url, params=params)
  return page.content
    
def collect_bet_results(session, maxround=34):
  rounds=[]
  round_from=[]
  round_to=[]
  hometeams=[]
  awayteams=[]
  FTHGs=[]
  FTAGs=[]
  myFTHGs=[]
  myFTAGs=[]
  pistorFTHGs=[]
  pistorFTAGs=[]
  myPoints=[]
  pistorPoints=[]
  
  
  for i in range(1,maxround+1):
    html_content = load_page(session, baseurl+'#goToSpieltagsergebnisse', params={'spieltag':i})
    
    soup = BeautifulSoup(html_content, 'html.parser')
    start = soup.find('a', attrs={'class':'hidden', 'name':'goToSpieltagsergebnisse'})
    
    spieltag = start.find_next_sibling('h3')
    print(spieltag.text.strip())
    m = re.search(r'(\d+)\. Spieltag \((\d+\.\d+\.) - (\d+\.\d+\.\d+)\)', spieltag.text.strip())
    sp_id, sp_from, sp_to = m.groups()
    sp_id = int(sp_id)
    
    table = spieltag.find_next_sibling('table')
    
    for tr in table.find('tbody').find_all('tr'):
      td = tr.find_all('td')
      match = td[0].text.strip()
      hometeam, awayteam = match.split(' : ')
      hometeams.append(hometeam)
      awayteams.append(awayteam)
      FTR = td[1].text.strip()
      FTHG, FTAG = FTR.split(':')
      FTHGs.append(int(FTHG))
      FTAGs.append(int(FTAG))
    
      FTR = td[2].text.strip()
      FTHG, FTAG = FTR.split(':')
      pistorFTHGs.append(int(FTHG))
      pistorFTAGs.append(int(FTAG))
    
      FTR = td[4].text.strip()
      FTHG, FTAG = FTR.split(':')
      myFTHGs.append(int(FTHG))
      myFTAGs.append(int(FTAG))
    
      pistorPoints.append(int(td[3].text.strip()))
      myPoints.append(int(td[5].text.strip()))
      
      rounds.append(sp_id)
      round_from.append(sp_from)
      round_to.append(sp_to)
      
    pistorBonus = int(table.find('tfoot').find_all('tr')[1].find_all('td')[1].text)
    myBonus     = int(table.find('tfoot').find_all('tr')[0].find_all('td')[3].text)
    
    hometeams.append("Bonus")
    awayteams.append("Bonus")
    FTHGs.append(-1)
    FTAGs.append(-1)
    
    pistorFTHGs.append(-1)
    pistorFTAGs.append(-1)
    
    myFTHGs.append(-1)
    myFTAGs.append(-1)
    
    pistorPoints.append(pistorBonus)
    myPoints.append(myBonus)
    
    rounds.append(sp_id)
    round_from.append(sp_from)
    round_to.append(sp_to)
    
  
  pistor_data = pd.DataFrame({
    'Round':rounds,
    'DateFrom':round_from,
    'DateTo':round_to,
    'HomeTeam':hometeams,
    'AwayTeam':awayteams,
    'FTHG':FTHGs,
    'FTAG':FTAGs,
    'myFTHG':myFTHGs,
    'myFTAG':myFTAGs,
    'myPoints':myPoints,
    'psFTHG':pistorFTHGs,
    'psFTAG':pistorFTAGs,
    'psPoints':pistorPoints,
  })
  return   pistor_data

def collect_ranking(session, maxpages=1):
  ranks=[]
  names=[]
  userids=[]
  points=[]
  
  for i in range(1,maxpages+1):
    html_content = load_page(session, baseurl+'/spielstand_einzel.php', params={'page':i})
    
    soup = BeautifulSoup(html_content, 'html.parser')
    start = soup.find('h3', attrs={'class':'ressort'})
    
    table = start.find_next_sibling('table')
    prev_rank = 0
    for tr in table.find('tbody').find_all('tr'):
      td = tr.find_all('td')
      rank = td[0].text.strip().replace('.', '')
      if rank=='':
        rank=prev_rank
      else:
        rank=int(rank)
      ranks.append(rank)
      prev_rank = rank

      points.append(int(td[2].text.strip()))
      
      link = td[1].find('a')
      names.append(link.text.strip())
      userids.append(link["href"].split("=")[1])
    
  ranking_data = pd.DataFrame({
    'Rank':ranks,
    'Name':names,
    'Userid':userids,
    'Points':points
  })
  return ranking_data

def collect_user_tipps(session, userid, roundfrom=1, roundto=34):

  rounds=[]
  users=[]
  hometeams=[]
  awayteams=[]
  FTHGs=[]
  FTAGs=[]
  myFTHGs=[]
  myFTAGs=[]
  myPoints=[]
  
  for i in range(roundfrom, roundto+1):
    html_content = load_page(session, baseurl+'/user_tipps.php', params={'spieltag':i, 'id':userid})
    
    soup = BeautifulSoup(html_content, 'html.parser')
    spieltag = soup.find('h3', attrs={'class':'ressort'})
    table = spieltag.find_next_sibling('table')
    
    for tr in table.find('tbody').find_all('tr'):
      td = tr.find_all('td')
      match = td[0].text.strip()
      hometeam, awayteam = match.split(' : ')
      hometeams.append(hometeam)
      awayteams.append(awayteam)
      FTR = td[1].text.strip()
      FTHG, FTAG = FTR.split(':')
      FTHGs.append(int(FTHG))
      FTAGs.append(int(FTAG))
    
      FTR = td[2].text.strip()
      FTHG, FTAG = FTR.split(':')
      myFTHGs.append(int(FTHG) if FTHG!="-" else None)
      myFTAGs.append(int(FTAG) if FTAG!="-" else None)
      points = td[3].text.strip()
      myPoints.append(int(points) if points!="-" else None)
      
      rounds.append(i)
      users.append(userid)
      
    bonus = table.find('tfoot').find_all('tr')[0]
    if bonus.find('td').text.strip() in ["Extra-Punkt für Pistor-Bezwinger:", "Pistor schlägt 2/3:"]:
      myBonus = int(bonus.find_all('td')[1].text)
    else:
      myBonus = 0
    
    hometeams.append("Bonus")
    awayteams.append("Bonus")
    FTHGs.append(-1)
    FTAGs.append(-1)
    myFTHGs.append(-1)
    myFTAGs.append(-1)
    myPoints.append(myBonus)
    rounds.append(i)
    users.append(userid)
    
  
  pistor_data = pd.DataFrame({
    'Round':rounds,
    'Userid':users,
    'HomeTeam':hometeams,
    'AwayTeam':awayteams,
    'FTHG':FTHGs,
    'FTAG':FTAGs,
    'uFTHG':myFTHGs,
    'uFTAG':myFTAGs,
    'uPoints':myPoints,
  })
  return   pistor_data



session = login()
pistor_data = collect_bet_results(session, maxround=3)
print(pistor_data)
ranking_data = collect_ranking(session, 3060)
ranking_data.to_csv("pistor_ranking_data.csv", encoding = "utf-8", index=True)
print(ranking_data)

ranking_data = pd.read_csv("pistor_ranking_data.csv", encoding = "utf-8")

all_user_data = pd.read_csv("user_tipps.csv", encoding = "utf-8")

user_data_sp = collect_user_tipps(session, roundfrom=33, roundto=33, userid=10) # Sven Pistor
user_data_ms = collect_user_tipps(session, roundfrom=33, roundto=33, userid=218206) # ich

all_user_data = pd.concat([user_data_ms, user_data_sp, all_user_data], axis=0, ignore_index=True)

for userid in ranking_data.Userid.iloc[0:1000]:
  user_data = collect_user_tipps(session, roundfrom=33, roundto=33, userid=userid)
  all_user_data = pd.concat([all_user_data, user_data], axis=0, ignore_index=True)
all_user_data.to_csv("user_tipps.csv", encoding = "utf-8", index=False)



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



