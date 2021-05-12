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
import time

#os.chdir("D:/gitrepository/Football/football/TF")
os.chdir("C:/git/football/TF")
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
  #print(url, params)
  page = session.get(url, params=params, allow_redirects=False)
  if page.status_code != requests.codes.ok:
    msg = 'Response code = '+str(page.status_code)
    print(msg)
    raise ValueError(msg)
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
      FTR = td[1].text.strip()
      FTHG, FTAG = FTR.split(':')
      if FTHG!="-" and FTHG!="#":
        hometeams.append(hometeam)
        awayteams.append(awayteam)
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
      
    try:
      pistorBonus = int(table.find('tfoot').find_all('tr')[1].find_all('td')[1].text)
      myBonus     = int(table.find('tfoot').find_all('tr')[0].find_all('td')[3].text)
    except:
      pistorBonus = 0
      myBonus = 0

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

#collect_user_tipps(session, roundfrom=1, roundto=1, userid=10)
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
      FTR = td[1].text.strip()
      FTHG, FTAG = FTR.split(':')
      #print(", ".join([hometeam, awayteam, FTHG, FTAG]))
      if FTHG!="-" and FTHG!="#":
        hometeams.append(hometeam)
        awayteams.append(awayteam)
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
    try:
      bonus = table.find('tfoot').find_all('tr')[0]
      if bonus.find('td').text.strip() in ["Extra-Punkt für Pistor-Bezwinger:", "Pistor schlägt 2/3:"]:
        myBonus = int(bonus.find_all('td')[1].text)
      else:
        myBonus = 0
    except:
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
if False:
  pistor_data = collect_bet_results(session, maxround=31)
  print(pistor_data)
  ranking_data = collect_ranking(session, 3063)
  ranking_data.to_csv("pistor_ranking_data_final.csv", encoding = "utf-8", index=True)
  print(ranking_data)

ranking_data = pd.read_csv("pistor_ranking_data_final.csv", encoding = "utf-8")

all_user_data = pd.read_csv("user_tipps.csv", encoding = "utf-8")

current_data = all_user_data.groupby("Userid").Round.max()

if False:
  for userid,maxround in zip(current_data.index, current_data):
    print(userid, maxround)
    user_data = collect_user_tipps(session, roundfrom=maxround+1, roundto=34, userid=userid)
    all_user_data = pd.concat([all_user_data, user_data], axis=0, ignore_index=True)
    if userid%50==0:
      all_user_data.to_csv("user_tipps.csv", encoding = "utf-8", index=False)

  all_user_data.to_csv("user_tipps.csv", encoding = "utf-8", index=False)

known_users = all_user_data.Userid.unique()
print(len(known_users))
print(type(known_users))


if False:
  user_data_sp = collect_user_tipps(session, roundfrom=1, roundto=31, userid=10) # Sven Pistor
  user_data_ms = collect_user_tipps(session, roundfrom=1, roundto=31, userid=218206) # ich
  user_data_udo = collect_user_tipps(session, roundfrom=1, roundto=31, userid=339368) # udo-h
  user_data_anke = collect_user_tipps(session, roundfrom=1, roundto=31, userid=334333) # hatatitla13

  all_user_data = pd.DataFrame()
  all_user_data = pd.concat([user_data_ms, user_data_sp, user_data_anke, user_data_udo, all_user_data], axis=0, ignore_index=True)

import threading


def thread_function(threadID, start, stop):
  print ("Starting " + str(threadID))
  user_data = pd.DataFrame()
  file_name = "user_tipps"+str(threadID)+".csv"
  if os.path.exists(file_name):
    user_data = pd.read_csv(file_name, encoding="utf-8")
    known_users = pd.concat([all_user_data.Userid, user_data.Userid]).unique()
    print(len(known_users))

    print("loaded " + str(user_data.count()) + "rows from " + file_name)
  else:
    known_users =  [] #all_user_data.Userid.unique()
    print(file_name)

  session = login()
  for i, userid in enumerate(ranking_data.Userid.iloc[start:stop]):
    if userid not in known_users:
      print(str(time.asctime()) + " " + str(threadID) + " " + str(i) + " " + str(start+i) )
      count = 1
      while count != 0:
        try:
          i_user_data = collect_user_tipps(session, roundfrom=32, roundto=32, userid=userid)
          count = 0
        except:
          time.sleep(10*count)
          try:
            session = login()
          except:
            ++count
            print(str(time.asctime()) + " " + str(threadID) + " " + str(i) + " " + str(start+i) + "retry count " + str(count))

      user_data = pd.concat([user_data, i_user_data], axis=0, ignore_index=True)
      if i % 500 == 0:
        user_data.to_csv(file_name, encoding="utf-8", index=False)
      if os.path.exists("c:/git/football/TF/stoppistordata.txt"):
        break
  user_data.to_csv(file_name, encoding="utf-8", index=False)

#thread1 = threading.Thread(target=thread_function, args=(1, 14000, 15000))
#thread2 = threading.Thread(target=thread_function, args=(2, 18600, 20000))
#thread3 = threading.Thread(target=thread_function, args=(3, 8200, 8400))
# thread4 = threading.Thread(target=thread_function, args=(4, 8400, 8600))
# thread5 = threading.Thread(target=thread_function, args=(5, 8600, 8800))
# thread6 = threading.Thread(target=thread_function, args=(6, 8800, 9000))
#thread1 = threading.Thread(target=thread_function, args=(1, 20000, 25000))
#thread2 = threading.Thread(target=thread_function, args=(2, 25000, 30000))
#thread3 = threading.Thread(target=thread_function, args=(3, 30000, 35000))
#thread3 = threading.Thread(target=thread_function, args=(3, 35000, 40000))
# thread1 = threading.Thread(target=thread_function, args=(1, 40000, 50000))
# thread2 = threading.Thread(target=thread_function, args=(2, 50000, 60000))
# thread4 = threading.Thread(target=thread_function, args=(4, 60000, 70000))
# thread3 = threading.Thread(target=thread_function, args=(3, 70000, 78000))
thread1 = threading.Thread(target=thread_function, args=(1, 1, 20000))
thread2 = threading.Thread(target=thread_function, args=(2, 20000, 40000))
thread3 = threading.Thread(target=thread_function, args=(3, 40000, 60000))
thread4 = threading.Thread(target=thread_function, args=(4, 60000, 80000))

# Start new Threads
#print(thread1)
#thread_function(1, 7860, 8000)
thread1.start()
thread2.start()
thread3.start()
thread4.start()
# thread5.start()
# thread6.start()


# session = login()
# for i,userid in enumerate(ranking_data.Userid.iloc[1:10000]):
#   if userid not in known_users:
#     print(i)
#     user_data = collect_user_tipps(session, roundfrom=1, roundto=31, userid=userid)
#     all_user_data = pd.concat([all_user_data, user_data], axis=0, ignore_index=True)
#     if i%50==0:
#       all_user_data.to_csv("user_tipps.csv", encoding = "utf-8", index=False)
#
# all_user_data.to_csv("user_tipps.csv", encoding = "utf-8", index=False)


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



