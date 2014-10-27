#! /usr/local/bin/python
import pandas as pd
import pandas.io.sql as psql
import MySQLdb
import numpy as np
from datetime import datetime 
'''
mysql_con= MySQLdb.connect(host='localhost', user='scird', passwd='scirdpass', db='scireader') #host ='171.65.77.20' if NOT connecting from the server
q= "select pubmed_pubmed.pmid,  pubmed_pubmed.DateCreated, pubmed_pubmed.DateCompleted, pubmed_pubmed.DateRevised, pubmed_pubmed.Pubdate, journals_impactfactor.IF , journals_impactfactor.ISSN, pubmed_pubmed.JournalTitle from  pubmed_pubmed inner join journals_impactfactor on (pubmed_pubmed.ISSNLinking = journals_impactfactor.ISSN) "
ifdata1=psql.frame_query(q,mysql_con)
ifdata1.to_csv('Pmid_IF_date_data1.txt',sep='\t')

q= "select pubmed_pubmed.pmid,  pubmed_pubmed.DateCreated, pubmed_pubmed.DateCompleted, pubmed_pubmed.DateRevised, pubmed_pubmed.Pubdate, journals_impactfactor.IF , journals_impactfactor.ISSN, pubmed_pubmed.JournalTitle from  pubmed_pubmed inner join journals_impactfactor on (pubmed_pubmed.ISSN=journals_impactfactor.ISSN) "
ifdata2=psql.frame_query(q,mysql_con)
ifdata2.to_csv('Pmid_IF_date_data2.txt',sep='\t') 

ifdata==pd.merge(ifdata1,ifdata2,how='outer', left_on='pmid', right_on='pmid',sort=False,left_index=False,right_index=False)

'''
mysql_con= MySQLdb.connect(host='localhost', user='scird', passwd='scirdpass', db='scireader') #host ='171.65.77.20' if NOT connecting from the server
q= "select pubmed_pubmed.pmid,  pubmed_pubmed.DateCreated, pubmed_pubmed.DateCompleted, pubmed_pubmed.DateRevised, pubmed_pubmed.Pubdate, pubmed_pubmed.JournalTitle,pubmed_pubmed.ISSN,pubmed_pubmed.ISSNLinking from pubmed_pubmed"
pubmed_data=psql.frame_query(q,mysql_con)
qr= "select pubmed_recent_pubmed.pmid,  pubmed_recent_pubmed.DateCreated, pubmed_recent_pubmed.DateCompleted, pubmed_recent_pubmed.DateRevised, pubmed_recent_pubmed.Pubdate, pubmed_recent_pubmed.JournalTitle,pubmed_recent_pubmed.ISSN,pubmed_recent_pubmed.ISSNLinking from pubmed_recent_pubmed"
pubmed_recent_data=psql.frame_query(qr,mysql_con)
pubmed_data=pd.concat([pubmed_data,pubmed_recent_data])

r="select journals_impactfactor.IF, journals_impactfactor.ISSN from journals_impactfactor" 
if_jourdata=psql.frame_query(r,mysql_con)
compdata=pd.read_csv('all_20140221093827_150_9603compostion.txt',usecols=[0,1,2,3], header=None, skiprows=1, sep='\t', names=['ind','pmid','tpcno1','tpcprob1'])
pubmed_data['pmid']=pubmed_data['pmid'].apply(int)
pubmed_data.drop_duplicates(cols='pmid',inplace=True)
comp_date_data=pd.merge(compdata,pubmed_data,how='left', left_on='pmid', right_on='pmid',sort=False)
mysql_con.close()
def g(x):
    if x[7]==None:
        return x[4]
    else:
        return x[7]     
   # return max(x[4:8].dropna())
    
    
comp_date_data['MostRecentDate']=comp_date_data.apply(g,axis=1)
#chk sum(comp_date_data.MostRecentDate.isnull())
comp_date_data.drop(['DateCreated','DateCompleted','Pubdate','DateRevised'],axis=1,inplace=True)
#Turns out that the if_jourdata has some duplicates. Pick the one with max IF factor. 
#Do this manually

aa=if_jourdata.set_index('ISSN').index.get_duplicates()
for ISSN in aa:
      bb=if_jourdata.IF[if_jourdata.ISSN == ISSN]
      if_jourdata.IF[if_jourdata.ISSN == ISSN]=max(bb)
'''
if_jourdata.IF[if_jourdata.ISSN == '0013-5585']=1.157
if_jourdata.IF[if_jourdata.ISSN == '0379-5136']=0.563
if_jourdata.IF[if_jourdata.ISSN == '0908-4282']=1.155
'''

#imp: now all the rows are duplicated . Need to remove them!!that is why the numbers dont add up.
if_jourdata.drop_duplicates(cols='ISSN',inplace=True)
ifdata1=pd.merge(comp_date_data,if_jourdata,how='left',left_on='ISSNLinking',right_on='ISSN',sort=False) #This does not have the same order as far as pmids are concerened because you link on ISSN nos.
##There are about 37600 pmids for which there is an ISSN # but NO ISSNLInking #.
#check to see if any of those have ISSN numbers.
ifdata1[((ifdata1.ISSNLinking == 'NULL') & (ifdata1.ISSN_x != 'NULL'))]
#yes!37600 of them
#ifdata1[((ifdata1.ISSNLinking == 'NULL') & (ifdata1.ISSN_x != 'NULL')) &(ifdata1.ISSN_y.isnull())] is also 37600 of them.
#replace the ISSNLinking numbers by the ISSN numbers,
#ifdata1.ISSNLinking[((ifdata1.ISSNLinking == 'NULL') & (ifdata1.ISSN_x != 'NULL'))]=ifdata1.ISSN_x[((ifdata1.ISSNLinking == 'NULL') & (ifdata1.ISSN_x != 'NULL'))]

ifdata1.ISSNLinking[ifdata1.ISSNLinking.isnull()]=ifdata1.ISSN_x[ifdata1.ISSNLinking.isnull()]

# Put their IF's as 1 and substitute their (null)ISSLinking #'s with their ISSN numbers.
#then drop  'ISSN_x','ISSN_y'. I do this because for the majority of journals the match is on ISSNLinking.
#chk to see if eLife alraedy has the correct ISSNLining num.
ifdata1.drop(['ISSN_x','ISSN_y'],axis=1,inplace=True)

ifdata1.IF[ifdata1.JournalTitle == 'eLife'] =15
#journals with no if: there are 3783 of them. 
#NoIFJournals=unique(ifdata1.JournalTitle[ifdata1.IF.isnull()])
#convert the IF of these journals into 1.
ifdata1.IF[ifdata1.IF.isnull()]=1.0
#make my own IF list from unique ISSNLinking numbers from ifdata1

def datefunc1(x):
    now=datetime.now()
    delta=now.date()- (x.MostRecentDate)
    if delta.days>90:
       datefctr=0
       return datefctr   
    if 90>= delta.days >60:
       datefctr=(.5)**(delta.days-60.0)  
       return datefctr
    elif 60.0 >= delta.days >=  15:
       datefctr=2.5-((1.5/45)*delta.days)  
       return datefctr
    else:
       datefctr=2
       return datefctr

ifdata1['datefactor']=ifdata1.apply(datefunc1,axis=1)

def impactfactorfctr(x):
    im=x.IF
    print im 
    if im >=30.0:
        im_fctr=8.0
        return im_fctr
    if (30.0>im) & (im>=20.0):
        im_fctr=5.0
        return im_fctr
    if (20.0>im) &  (im>=10.0):
        im_fctr=3.0   
        return im_fctr    
    if (10.0>im) & (im>=4.0):  
        im_fctr=1.0   
        return im_fctr 
    else:     
        im_fctr=0.1        
        return im_fctr      

ifdata1['impact_factor_fctr']=ifdata1.apply(impactfactorfctr,axis=1)

ifdata1['Rec_Score1']=ifdata1['tpcprob1']*8.0+ifdata1['datefactor']+ifdata1['impact_factor_fctr']*0.5
#ifdata1['Rec_Score2']=ifdata1['tpcprob1']*ifdata1['datefactor']*ifdata1['impact_factor_fctr']
ifdata1['Rec_Score1'][ifdata1.datefactor == 0]=0

#finaldf=ifdata1.ix[:,['pmid','tpcno1','tpcprob1','Rec_Score1','Rec_Score2']]

#Making topic lists
for i in xrange(150):
    tmpdf = ifdata1[ifdata1.tpcno1 == i].sort('Rec_Score1',ascending=False)  
    tmpdf=tmpdf.ix[:,['tpcno1','pmid','Rec_Score1']]
    tmpdf=tmpdf.reset_index()
    tmpdf.drop(['index'],axis=1,inplace=True)
    tmpdf.to_csv("/home/priya/For_Yonggan/Topic_lists_sorted/"+str(i)+"_sorted_df.txt", sep='\t',index_label='Index')

#making a single list.

out=open('/home/priya/For_Yonggan/Topic_lists_sorted/All_tpids_sorted.txt','w')
linecnt=1 
out.write("id\tdbname\tdbid\tsubcatid\tprobability\n")
for i in xrange(150):
    tmp=open('/home/priya/For_Yonggan/Topic_lists_sorted/'+str(i)+'_sorted_df.txt').readlines()
    for line in tmp[1:501]:   
         tt=line.split('\t')
         out.write(str(linecnt)+"\tpubmed\t"+tt[2]+'\t'+tt[1]+'\t'+tt[3])
         linecnt=linecnt+1
out.close()      
            
'''       
    tmpdf = ifdata1[ifdata1.tpcno1 == i].sort('Rec_Score1',ascending=False)  
    newdf=tmpdf.ix[0:500,['tpcno1','pmid','Rec_Score1']]
    newdf=newdf.reset_index()
    newdf.drop(['index'],axis=1,inplace=True)
    newdf.to_csv(f, header=False)
'''

#Making Journal lists
ISSN_List= pd.unique(list(ifdata1.ISSNLinking))
#ISSN_List=list(unique(ifdata1.ISSNLinking))

for issn in ISSN_List:
    tmpdf = ifdata1[ifdata1.ISSNLinking == issn].sort('Rec_Score1',ascending=False) 
    tmpdf=tmpdf.ix[:,['tpcno1','pmid','Rec_Score1']]
    tmpdf=tmpdf.reset_index() 
    tmpdf.drop(['index'],axis=1,inplace=True)
    jour_name=ifdata1[ifdata1.ISSNLinking == issn].iloc[0,4]
    tmpdf.to_csv('/home/priya/For_Yonggan/Journals_sorted/'+issn+"_sorted_df.txt", sep='\t',index_label='Index')
    
   
#plot top 3 pmids from every topic.    
    