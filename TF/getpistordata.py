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
#url = 'http://coordinated-tray.surge.sh'
url='https://allegegenpistor.wdr2.de'
#html_content = request.urlopen(url).read()

session = requests.Session() 
#session.auth = ('TCSNet', 'Rebekka95') 
#session.get(url)
#response = session.request('POST', url, auth = ('TCSNet', 'Rebekka95'), headers = {

response = session.post(url, data = {
    'login' : '1',
    'username': 'TCSNet',
    'password': 'Rebekka95'
    })

html_content = response.content
print(response.headers)

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


for i in range(1,33):
  page = session.get(url+'#goToSpieltagsergebnisse', params={'spieltag':i})
  html_content = page.content
  print(i)
  print(page.headers)
  
  #with open("pistor.html", "wb+") as f:
  #  f.write(html_content)
  
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
pistor_data.to_csv("pistor_data.csv", encoding = "utf-8", index=False)
print(pistor_data)


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



