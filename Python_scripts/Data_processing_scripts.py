#! /usr/local/bin/python
import pandas as pd
import pandas.io.sql as psql
import MySQLdb
import numpy as np
from datetime import datetime,timedelta 
#import h5py
import subprocess
import os
from dateutil import rrule
import glob
import shutil
import numpy as np
import itertools
from bibtexparser.bparser import BibTexParser
import pubRecords_1 as pr
import cPickle as pickle
import csv
import tables as tb
import scipy as sparse
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto
import tables as tb
from numpy import array
from scipy import sparse
'''
Module containing many of my functions to clean/parse Pubmed data, update corpus, calculate word similarities, create and update Impact Factor files
'''


def get_MysqlData(startyear):
    '''
    function that takes a year in YYYY format and pulls out all the relevant data from the mysql database. Note that this function returns a file that may have multiple lines corresponding to the same pmid, but different Abstract Orders, as each AbstractOrder has a separate record. 
    MysqlData_to_OneRecPerLineFmt, used in conjunction with this function will merge them .
    Note- I only pull out ISSNLinking as that is what is used to connect to the IF database
    '''
    q='select p.pmid, p.DateCreated,p.ArticleTitle, p.JournalTitle,p.ISSN,p.ISSNLinking ,a.AbstractText, a.AbstractOrder from pubmed_pubmed as p inner join  pubmed_abstract as a on p.pmid = a.pmid_id where (p.DateCreated between %s and %s)'        
    mysql_con= MySQLdb.connect(host='ASDFGHJ', user='yyyy', passwd=xxxxx', db='88888')
    startyear=int(startyear)
    for month in range(1,13,1):        
        fd=str(startyear)+'-'+str(month)+"-1"
        ld=str(startyear)+'-'+str(month)+"-31"
        data=psql.frame_query(q,params=(fd,ld),con=mysql_con)
        nrow,ncol=data.shape
        if nrow >5:  
            tt=(datetime.strptime(fd,"%Y-%m-%d")).strftime("%b")
            data.to_csv(tt+'_'+str(startyear)+'_data.csv',index=False)
    
def MysqlData_to_OneRecPerLineFmt(filename,newpath):
    '''
    function that takes a csv file containing all relevant data pulled out for selected pmids and merges/appends all the various paragraphs belonging to the same paper, so they are ready to be fed into the LDA machine. The pathname to store the new merged file is te other argument. 
    This function creates two files: one for LDA(For_LDA*) and one contained all the data(Uniq*)
    '''
    data=pd.read_csv(filename)
    nrow,ncol=data.shape
    if nrow >5:  
        uniq_pmids=pd.unique(data.pmid)
        data_df=data.drop_duplicates(cols=['pmid'])
        pmids_list=list(data.pmid)
        full_abs=[]
        exception_pmid=[]
        #print "test1"
        for i in uniq_pmids:
            order_cnt=max(data.AbstractOrder[data.pmid==i])
            if order_cnt>1:
                #print "test1" 
                tmp_abs=''
                for k in range(1,order_cnt+1):
                    try: 
                        tmp_abs= tmp_abs+str(pd.unique(data.AbstractText[(data.pmid==i) & (data.AbstractOrder==k)])[0])                   
                    except:
                        exception_pmid.append(i)
                full_abs.append(tmp_abs)            
            else:
                #print "test2"
                full_abs.append(data_df.AbstractText[data_df.pmid==i].values[0])
        data_df.drop(['AbstractText'],axis=1, inplace=True)  
        data_df.drop(['AbstractOrder'],axis=1,inplace=True)
        data_df['AbsText']=full_abs
        data_df.dropna(axis=0, how='any',inplace=True)            
        data_df.to_csv(newpath+'/Uniq_'+filename,sep='\t',index=False)#write out the file in format to store
        #write out another file with just text for LDA.
        print "writing out " + filename +" to compute lda probabilities"
        data_df.to_csv(newpath+'/ForLDA_'+filename,sep='\t',index=False,cols=['pmid','ArticleTitle','AbsText'],header=False)
        #write out the exception to a file"
        out = csv.writer(open(filename+"exceptions.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
        out.writerow(exception_pmid)
        print "done lda for", filename
        os.remove(filename)
   
def update_new_papers():
    '''
    This function should do the following:
         - update the papers in lda-data with papers downloaded into the Scireader database since the last download/yesterday? 
         - convert into files for LDA processing and Unique data
    '''    
    #download most recent data from Scireader
    curr_mon=datetime.now().month 
    curr_year=datetime.now().year
    path='/home/ubuntu/lda-data/'+str(curr_year)
    print path
    os.chdir(path)
    fd=str(curr_year)+'-'+str(curr_mon)+"-1"
    ld=str(curr_year)+'-'+str(curr_mon)+"-31"
    mysql_con= MySQLdb.connect(host='xxxxxx, user='777777', passwd='8888888', db='tttttt')
    q='select p.pmid, p.DateCreated,p.ArticleTitle, p.JournalTitle,p.ISSN,p.ISSNLinking ,a.AbstractText, a.AbstractOrder from pubmed_pubmed as p inner join  pubmed_abstract as a on p.pmid = a.pmid_id where (p.DateCreated between %s and %s)'        
    data=psql.frame_query(q,params=(fd,ld),con=mysql_con)
    nrow,ncol=data.shape
    if nrow>5:  #dont bother saving if its a Month in the future or a month without much data        
        tt=(datetime.strptime(fd,"%Y-%m-%d")).strftime("%b")
        data.to_csv(tt+'_'+str(curr_year)+'_data.csv',index=False)
        # re-sort the file into format with one record per line as well as make the txt file for LDA    
        MysqlData_to_OneRecPerLineFmt(tt+'_'+str(curr_year)+'_data.csv',path)    
        filelist=glob.glob('ForLDA_'+tt+'*data.csv')
        print "filename is ", filelist 
        for file in filelist:
            if len(open(file).readlines())>5:   
                print "running LDA for ", file 
                subprocess.call(['/home/ubuntu/Scirec_scripts/lda_scripts/run_LDA.sh',file])
                shutil.copyfile('output_file_tpcs_composition_for_database.txt',file.split('csv')[0]+'LDA_composition_for_database.txt')
                shutil.copyfile('output_file_tpcs_composition_with_pmid.txt',file.split('csv')[0]+'LDA_composition_with_pmid.txt')
                os.remove('output_file_tpcs_composition.txt')
                os.remove('output_file_tpcs_composition_for_database.txt') 
                os.remove('output_file_tpcs_composition_with_pmid.txt') 
            else:
                print "file is short. It has length ", len(open(file).readlines())          
    else:
        print "file is short. It has length ", nrow 
    
    
def chk_if_allAbstractOrdersArePresent(filename):    
    data=pd.read_csv(filename)
    uniq_pmids=pd.unique(data.pmid)
    data_df=data.drop_duplicates(cols=['pmid'])
    pmids_list=list(data.pmid)
 #   full_abs=[]
    for i in uniq_pmids:
        order_cnt=max(data.AbstractOrder[data.pmid==i])
        if order_cnt>1:
            #print "test1" 
            tmp_abs=''
            for k in range(1,order_cnt+1):
                try:
                    tt=str(pd.unique(data.AbstractText[(data.pmid==i) & (data.AbstractOrder==k)])[0])   
                except IndexError:
                    failed_pmids.append(i)
                    break 
    print failed_pmids                
    return failed_pmids                                             

def update_corp(curr_basic_dataset_file,user_lib_pmids_list):
    '''
    function to remove all the pmids present in the users library from our sampling corpus so we dont recommend duplicate papers!!  
    Input: the sampling corpus , dataframe of users pmids
    '''
    corpus=pd.read_csv(curr_basic_dataset_file)
    updated_corpus_pmids_df=pd.DataFrame(list(set(list(corpus.pmid)) - set(user_lib_pmids_list)),columns=['pmid'])
    return pd.merge(corpus,updated_corpus_pmids_df,how="inner",on="pmid",sort=False)
    
    

def create_basic_papers_dataset_for_tpcRec(): 
    '''
    This function should find all the papers in the last 6 months (from lda-data) and merge their relevant info into a dataframe. The argument should be todays date.
    Note there are two cases: one when the 6 months straddle two consecutive years, and one where all the 6 months are in the same year.
    This one just takes care of the 2nd case.
    '''
    month_list=['Dummy','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    today=datetime.now().date()
    fromdate=today-timedelta(180)
    basic_data_filelist=[]
    tpc_data_filelist=[]
    if today.year==fromdate.year:   
        curr_year=datetime.now().year
        path='/home/ubuntu/lda-data/'+str(curr_year)
        os.chdir(path) 
        basic_filelist=glob.glob('Uniq_*.csv')              
        basic_data=pd.read_csv(basic_filelist[0],usecols=[0,1,4,5], header=None, skiprows=1, sep='\t', names=['pmid','DateCreated','ISSN','ISSNLinking'])
        for file in basic_filelist[1:]:
            tmp_data=pd.read_csv(file,usecols=[0,1,4,5], header=None, skiprows=1, sep='\t', names=['pmid','DateCreated','ISSN','ISSNLinking'])
            basic_data=pd.concat([basic_data,tmp_data],ignore_index=True)
        #update the basic_data with the IF for the journals.    
        basic_data.ISSNLinking[basic_data.ISSNLinking.isnull()]=basic_data.ISSN[basic_data.ISSNLinking.isnull()]
        update_jour_IF(basic_data)
        if_data=pd.read_csv("/home/ubuntu/lda-data/Journal_IF_data.csv")        
        basic_data_with_IF=pd.merge(basic_data,if_data,how='left',left_on='ISSNLinking',right_on='ISSN',sort=False)           
        basic_data_with_IF.drop(['ISSN_x','ISSN_y'],axis=1,inplace=True)       
        basic_data_with_IF.drop_duplicates(cols='pmid',inplace=True)
        basic_data_with_IF.dropna(axis=0, how='any',inplace=True)  
        basic_data_with_IF.to_csv("/home/ubuntu/lda-data/Current_data/Current_basic_dataset_with_IF.csv",index=False)                      
        lda_filelist=glob.glob("*pmid*") 
        basic_lda_data=pd.read_csv(lda_filelist[0], header=None, sep='\t',usecols=[0,1,2,3,4,5,6], skiprows=1, names=['pmid','tpcno1','tpcprob1','tpcno2','tpcprob2','tpcno3','tpcprob3'])
        for file in lda_filelist[1:]:
            tmp_lda_data=pd.read_csv(file, header=None,sep='\t',usecols=[0,1,2,3,4,5,6], names=['pmid','tpcno1','tpcprob1','tpcno2','tpcprob2','tpcno3','tpcprob3'])
            basic_lda_data=pd.concat([basic_lda_data,tmp_lda_data],ignore_index=True)            
        basic_data_with_IF=basic_data_with_IF[basic_data_with_IF.DateCreated >= str(fromdate)]
        basic_data_with_IF.drop_duplicates(cols='pmid',inplace=True)
        basic_data_with_IF.dropna(axis=0, how='any',inplace=True)  
        basic_data_with_IF.to_csv("/home/ubuntu/lda-data/Current_data/Current_basic_LDA_dataset.csv",index=False)                       
        #The complete basic papers set is:    
        basic_all_data=pd.merge(basic_lda_data,basic_data_with_IF,how='left', left_on='pmid', right_on='pmid',sort=False) 
        #drop papers that are too old:
        basic_all_data=basic_all_data[basic_all_data.DateCreated >= str(fromdate)]
        basic_all_data.drop_duplicates(cols='pmid',inplace=True)
        basic_all_data.dropna(axis=0, how='any',inplace=True)  
        basic_all_data.to_csv("/home/ubuntu/lda-data/Current_data/Current_basic_dataset_with_IF_and_LDA.csv",index=False)
        

def create_basic_papers_dataset_with_AllTpcProb():
    '''
    function to create the  basic dataset made up of pmids, topicc probability of papers from the last 6 months for calculating topic similarity with the users library papers. 
    This basically combining the "LDA_composition_for_database" files 
    '''
    today=datetime.now().date()
    fromdate=today-timedelta(180)
    basic_AllTpcProb_filelist=[]
    if today.year==fromdate.year:   
        curr_year=datetime.now().year
        path='/home/ubuntu/lda-data/'+str(curr_year)
        os.chdir(path) 
        basic_AllTpcProb_filelist=glob.glob('*for_database.txt')              
        basic_AllTpcProb_data=pd.read_csv(basic_AllTpcProb_filelist[0],header=None, sep='\t')
        for file in basic_AllTpcProb_filelist[1:]:
            tmp_data=pd.read_csv(file, header=None, sep='\t')
            basic_AllTpcProb_data=pd.concat([basic_AllTpcProb_data,tmp_data],ignore_index=True)
                    
        basic_AllTpcProb_data.dropna(axis=0, how='any',inplace=True)      
        basic_AllTpcProb_data.to_csv("/home/ubuntu/lda-data/Current_data/Current_AllTpcProb_data.csv",sep ='\t',index=False)

def create_basic_papers_dataset_for_WordSim():
    '''
    function to create the  basic dataset made up of pmids, titles, abstracts of papers from the last 6 months to use for calculating word similarity. Note that this is pretty much the 
    same thing as combining the ForLDA_* files from the last 6 months. I initially absorb all files from the year and then merge with the Current_basic_dataset_with_IF_and_LDA set which 
    is created before this one and used before this one in personal recs to make sure they are consistent. But again remove nan's in abstracts and titles as a they will come and bite you later!
    '''
    today=datetime.now().date()
    fromdate=today-timedelta(180)
    basic_filelist=[]
    if today.year==fromdate.year:   
        curr_year=datetime.now().year
        path='/home/ubuntu/lda-data/'+str(curr_year)
        os.chdir(path) 
        basic_filelist=glob.glob('Uniq_*.csv')              
        basic_WordSim_data=pd.read_csv(basic_filelist[0], usecols=[0,2,6],header=None, skiprows=1, sep='\t', names=['pmid','Title','Abstract'])
        for file in basic_filelist[1:]:
            tmp_data=pd.read_csv(file, header=None, usecols=[0,2,6],sep='\t', skiprows=1,names=['pmid','Title','Abstract'])
            basic_WordSim_data=pd.concat([basic_WordSim_data,tmp_data],ignore_index=True)
        basic_WordSim_data.drop_duplicates(cols='pmid',inplace=True)    
        basic_WordSim_data.dropna(axis=0, how='any',inplace=True)  
        curr=pd.read_csv('/home/ubuntu/lda-data/Current_data/Current_basic_dataset_with_IF_and_LDA.csv',sep=',',usecols=[0],header=None,skiprows=1,names=['pmid'])
        WordSim_df_update=pd.merge(pd.DataFrame(curr),basic_WordSim_data,how="left",on="pmid",sort=False)
        WordSim_df_update.dropna(axis=0, how='any',inplace=True)
        WordSim_df_update.to_csv("/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv",sep ='\t',index=False)

def filter_topically_similar_papers(user_tpcs_file,Current_AllTpcProb_file):   
    '''
    This function would take in two files of topic probabilities: Current_papers_tpc_prob.csv and user_paper_tpc_prob.csv and output a shorter list
    of sample papers that are most topically similar to the user's papers. Using cosine similarity between topic vectors to determine degree of similarity    
    '''
    #reading in the relevant foiles into dataframes
    samplePaper_tpcs_df=pd.read_csv(Current_AllTpcProb_file,sep='\t',header=None)  
    user_tpcs_df=pd.read_csv(user_tpcs_file,sep='\t',header=None)    
    #making sure there are no duplicates in either dataframe:
    samplePaper_tpcs_df.drop_duplicates(cols=[0],inplace=True)    
    samplePaper_tpcs_df.dropna(axis=1,how='any',inplace=True)   
    user_tpcs_df.drop_duplicates(cols=[0],inplace=True)  
    #pulling out pmids
    samplePaper_pmids=samplePaper_tpcs_df[0]
    user_pmids=user_tpcs_df[0]   
    #drop the columns containing pmids to prepare for calculating cosine similarity and converting to np arrays   
    samplePaper_tpcs_df.drop([0],axis=1,inplace=True)
    user_tpcs_df.drop([0],axis=1,inplace=True)    
    samplePaper_tpcs_array=np.array(samplePaper_tpcs_df)
    user_tpcs_array=np.array(user_tpcs_df)
    #importing all required modules
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import cosine
    cos_sim = 1- pairwise_distances(user_tpcs_array,samplePaper_tpcs_array, metric="cosine")    
    out=open('/home/ubuntu/Calculating_Rec_Scores/On_AWS/User_vs_Current_Tpc_sim.txt','w')
    topically_sim_papers_dict={}
    topically_sim_papers_pmids_list=[]
    for i in range(len(user_pmids)):
        topically_sim_papers_dict[user_pmids[i]]=[samplePaper_pmids[k] for k in (-cos_sim[i]).argsort()[0:100]] 
        topically_sim_papers_pmids_list=topically_sim_papers_pmids_list+[samplePaper_pmids[k] for k in (-cos_sim[i]).argsort()[0:100]]
        #print user_pmids[i],':',[samplePaper_pmids[k] for k in (-cos_sim[i]).argsort()[0:100]] 
        out.write(str(user_pmids[i])+': '+str([samplePaper_pmids[i] for k in (-cos_sim[i]).argsort()[0:100]]))
    out.close()
    return set(topically_sim_papers_pmids_list),user_pmids
    
def vocab_list_Similarity_calculator(topically_sim_papers_pmids_list,user_file):
    '''
    This function takes in  files containing the Current pmids/text and creates a word vocabulary list and converts each paper in the Current paper list 
    into a word frequency vector. For each day; we will have ONE word vocabulary vector(??? Is this true?).Am creating the word sim vector using 
    ONLY the topically similar paper list that I get from Natalie? This function will have as input: a list of the pmids of shortlisted papers from the current corpus, 
    user_file (eg. ForLDA_1.txt) which has text info about the users papers (pmid,title, text). Natalie wants the output to be the similarity matrix. So output that.
    **Output is a file with recommendations. Will use nltk to create the TFIDF vector.     
    ''' 
    #Note the Word_Sim_data file is the same for every user: it is '/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv'
    WordSim_data_file='/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv'
    WordSim_df=pd.read_csv(WordSim_data_file, sep='\t',skiprows=1,header=None,names=['pmid','title','abstract'])
    tmp_topically_sim_papers_df= pd.DataFrame(list(topically_sim_papers_pmids_list),columns=['pmid'])     
    topically_sim_papers_df=pd.merge(tmp_topically_sim_papers_df,WordSim_df,how='left',on='pmid',sort=False)  
    #topically_sim_papers_df['pmid'][topically_sim_papers_df['title'].isnull()])
    '''topically_sim_papers_df.dropna(subset=['pmid','title','abstract'],how='any', inplace=True)'''
    #Now to calculate similarities between abstracts of usr lib paper and pmids topically similar to it(from above)
    #First need to combine Titles and Abstracts:    
    topically_sim_papers_df['title_and_abs']= topically_sim_papers_df['title']+ topically_sim_papers_df['abstract']
    topically_sim_papers_pmids_list=list(topically_sim_papers_df['pmid'])
    topically_sim_papers_text=topically_sim_papers_df['title_and_abs']
    text_corpus=list(topically_sim_papers_text)
    #create stoplist:
    stp_list1=open('/home/ubuntu/Calculating_Rec_Scores/en.txt','r').read().split('\n')
    stoplist=[x.strip(' ') for x in (stp_list1)] 
    #using the tfidf Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy as sp
    vectorizer=TfidfVectorizer(min_df=1, decode_error=u'ignore', stop_words=stoplist,strip_accents='ascii')
    X=vectorizer.fit_transform(str(line) for line in text_corpus)
    user_file_df=pd.read_csv(user_file,sep='\t',header=None,names=['pmid','title','abstract'])#
    #user_file_df=pd.read_csv(user_file,usecols=[0,2,6],sep='\t',header=None,skiprows=1,names=['pmid','title','abstract'])# check to make sure of the columns.
    user_file_df['title_and_abs']=user_file_df['title']+user_file_df['abstract']
    user_pmids_list=list(user_file_df['pmid'])
    user_file_text=user_file_df['title_and_abs']
    user_corpus=list(user_file_text)
    user_paper_vec=vectorizer.transform(user_corpus)
    user_paper_vec.toarray()
    #vectorizer.get_feature_names()
    #since its a sparse matrix, cannot really use scipy distance metrics     
    from sklearn.metrics.pairwise import linear_kernel  
    num_samples,num_features=X.shape
    num_user_papers,num_features=user_paper_vec.shape
    Cos_dist_array=[[]for i in range(0,num_user_papers)]
    for j in range(0,num_user_papers):
         Cos_dist_array[j] = linear_kernel(user_paper_vec[j:j+1], X).flatten()
    return(np.asarray(Cos_dist_array))
    
    
def create_vocab_for_corpus():
    '''
    This function creates the original word vector file for all the papers in our six month corpus.
    It reads in the WordSim file and creates the dictionary and the Count vector matrix and the list of pmids in the corpus, in the same order as 
    the papers in the WordSim file == TermFreqVector Matrix. Pickle all three.
    Run only once, then run the update TFV for corpus as a cronjob.
    '''
    WordSim_data_file='/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv'
    WordSim_df=pd.read_csv(WordSim_data_file, sep='\t',skiprows=1,header=None,names=['pmid','title','abstract'])
    WordSim_df['title_and_abs']= WordSim_df['title']+ WordSim_df['abstract']
    text_corpus=list(WordSim_df['title_and_abs'])
    stp_list1=open('/home/ubuntu/Calculating_Rec_Scores/en.txt','r').read().split('\n')
    stoplist=[x.strip(' ') for x in (stp_list1)] 
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy as sp    
    WordSim_df.pmid.to_pickle("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrixPmid.p")
    vectorizer=CountVectorizer(min_df=1,max_df=0.9995,decode_error=u'ignore', encoding='latin-1', stop_words=stoplist,strip_accents='ascii')
    X=vectorizer.fit_transform(str(line) for line in text_corpus)
    # vocab=vectorizer.get_feature_names()
    ##to get vocab dictionary
    words_dict=vectorizer.vocabulary_
    # save dictionary using pickle
    # WordSim_df.pmid
    pickle.dump(words_dict, open( "/home/ubuntu/lda-data/Current_data/CurrentWordDict.p", "wb" ) ) 
    #saving the sparse matrix: X is a csc matrix: compressed sparse column format.
    with open('/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrix', 'wb') as  outfile:
        pickle.dump(X, outfile, pickle.HIGHEST_PROTOCOL)
    
def update_corpus_TermFreqVec():
    '''
    This function reads in the old pickled Term Freq Vector, its associated dictionary, pmid list and
    the new WordSim file. It removes from the TFV, the papers that no longer fit the 180 day criteria. 
    Then finds the new papers added since the last update; converts them into TFV using the SAME vectorizer
    as was used to create the original TFV, and appends it to the TFV. Then it recomputes the new "pmid" list, 
    and makes sure that new TFV is indexed in the same order as the pmid list. It  
     
    '''
    #read in the previously stored TFV and related data
    with open("/home/ubuntu/lda-data/Current_data/CurrentWordDict.p", 'rb') as infile:
        words_dict = pickle.load(infile)
    with open("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrix",'rb') as infile:
        TermFreqMat = pickle.load(infile)
    with open("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrixPmid.p",'rb') as infile:
        TermFreqMatPmid = pickle.load(infile)
    #read in current WordSimfile:
    WordSim_data_file='/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv'
    WordSim_df=pd.read_csv(WordSim_data_file, sep='\t',skiprows=1,header=None,names=['pmid','title','abstract'])
    TermFreqPmid_df=pd.DataFrame(TermFreqMatPmid,columns=['pmid'])
    TermFreqPmid_df['tmp_ind']= TermFreqPmid_df.index
    #List of indices of the TFV that need to be kept are all those that are still in WordSim. All the rest need to be deleted)    
    #do an inner join==intersection and keep those papers.
    tmp_df=pd.merge(pd.DataFrame(WordSim_df.pmid),TermFreqPmid_df,how='inner',on='pmid',sort=False)
    #term freq Vec with old ids removed:
    TFV_tmp=TermFreqMat.tocsr()[(list(tmp_df.tmp_ind)),:]
    #pmids that need to be added to the TermFreqVector: new pmids uploaded since last TFV creation
    new_pmids_toadd=set(list(WordSim_df.pmid))-set(list(tmp_df.pmid))
    #need to get the abstracts for the remaining and append them to TFV in the same order as WordSim.    
    #extract the abst/text from the WordSim file for the new_pmids_toadd
    if len(new_pmids_toadd)>0:
        New_papers_toadd=pd.merge(pd.DataFrame(list(new_pmids_toadd),columns=['pmid']),WordSim_df,how="left",on='pmid',sort=False)
        New_papers_toadd['title_and_abs']=New_papers_toadd['title']+New_papers_toadd['abstract']
        New_papers_toadd_text=list(New_papers_toadd['title_and_abs'])    
        from sklearn.feature_extraction.text import CountVectorizer
        stp_list1=open('/home/ubuntu/Calculating_Rec_Scores/en.txt','r').read().split('\n')
        stoplist=[x.strip(' ') for x in (stp_list1)] 
        cntvectorizer=CountVectorizer(vocabulary=words_dict,decode_error=u'ignore', encoding='latin-1', stop_words=stoplist,strip_accents='ascii')
        New_papers_toadd_TFvec=cntvectorizer.transform(New_papers_toadd_text)
        #convert both the TFV_tmp and New_papers_toadd_TFvec: ie vstack? and save
        import scipy.sparse as sp_sparse
        TFV_tmp=sp_sparse.vstack((TFV_tmp,New_papers_toadd_TFvec),format='csr')        
        with open('/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrix', 'wb') as  outfile:
            pickle.dump(TFV_tmp, outfile, pickle.HIGHEST_PROTOCOL)
        #This is the new TermFreqVector. I need to update the TermFreqMatPmid to reflect this. The "new list should be"
        TFV_tmp_pmids_df=pd.DataFrame((list(tmp_df.pmid))+(list(New_papers_toadd.pmid)),columns=['pmid'])
        TFV_tmp_pmids_df.pmid.to_pickle("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrixPmid.p")
    else:
        with open('/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrix', 'wb') as  outfile:
            pickle.dump(TFV_tmp, outfile, pickle.HIGHEST_PROTOCOL)
        #This is the new TermFreqVector. I need to update the TermFreqMatPmid to reflect this. The "new list should be"
        TFV_tmp_pmids_df=pd.DataFrame((list(tmp_df.pmid)),columns=['pmid'])
        TFV_tmp_pmids_df.pmid.to_pickle("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrixPmid.p")
 

def chk_if_lib_exists(lib_id,outpath): 
    '''
    Function to check if this library file exist, and if it does to download it. now defunct as the date field has been added to the library_library table
    '''
    AWS_ACCESS_KEY_ID = '777777'
    AWS_SECRET_ACCESS_KEY = '%%%%%%%%%%%%%%%'
    s3 = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,host="s3-us-west-1.amazonaws.com")
    bucket=s3.get_bucket('tttttt') 
    bucket_list=bucket.list()
    teststr=str(lib_id)+".h5"
   #teststr="lib"+str(lib_id)+".h5"
    #keystring_list=[str(l.key) for l in bucket_list]
    isthere=False
    for l in bucket_list: 
        #See if the lib exist in S3 
        if teststr== str(l.key):
          #  os.mkdir("tmp_"+str(lib_id),0755)
            os.chdir(outpath)            
            l.get_contents_to_filename(teststr)
            isthere=True
    return isthere
    
def create_pytables_for_all_libraries():
    '''    
    Script to create a pytable representation with pmids,LDA and TFV for all libraries.  Run this only once -the first time to make sure all libraries have a pytable representation. After taht
    After that only run update_pytables_for_libraries   
    '''
    mysql_con= MySQLdb.connect(host='#######', user='scireader', passwd='&&&&&&&&&&', db='scireader')
    q='select * from library_library_papers'
    data=psql.frame_query(q,con=mysql_con)
    lib_list=list(pd.unique(data.libid_id))
    for lib_id in lib_list:
        create_pytable_for_single_library(lib_id) 


def create_pytable_for_single_library(lib_id):
    '''
    take library id as argument and create a h5 pytable with relevant info regarding LDA, pmid list, and Term Frequency Vector. 
    '''
    mysql_con= MySQLdb.connect(host='*********', user='sabsc', passwd='############', db='tttttt')
    q='select * from library_library_papers'
    data=psql.frame_query(q,con=mysql_con)
    lib_publist = list(data.dbid[data.libid_id == lib_id])
    if len(lib_publist)>=1:
        try:
            outpath='/home/ubuntu/usrlibrary-data/libid_'+str(lib_id)
            if not os.path.exists(outpath):
                os.makedirs(outpath)   
            os.chdir(outpath) 
            #remove the h5 file if an earlier incarnation exists for some reason.
            foo=glob.glob("*.h5")
            if foo:
                for i in range(len(foo)):
                    os.remove(foo[i])
            getting_users_papers_for_lda(lib_publist,str(lib_id),outpath)
            #write Uniq to h5 file.
            tmpUniq= pd.read_csv(outpath+"/Uniq_"+str(lib_id)+".txt",sep='\t',header=None,skiprows=1)
            tmpUniq.to_hdf("lib_"+str(lib_id)+".h5","lib"+str(lib_id)+"_uniqAll",mode="a",format="table",compress="blosc") 
            #read in Uniq file
            run_lda_user(outpath+"/ForLDA_"+str(lib_id)+".txt",str(lib_id)) # Run topic inferencer
            tmplda= pd.read_csv(outpath+"/"+str(lib_id)+"LDA_composition_with_pmid.txt",usecols=range(0,21),sep='\t',header=None)
            tmplda.to_hdf("lib_"+str(lib_id)+".h5","lib"+str(lib_id)+"_lda_data",mode="a",format="table",compress="blosc")
            print "got text from db and run lda for library", lib_id 
            #write out the pmids of this library to h5      
            lib_df=pd.DataFrame(lib_publist)
            lib_df.to_hdf("lib_"+str(lib_id)+".h5","lib"+str(lib_id)+"_pmid_list",mode="a",format="table",compress="blosc")
            #create the TFV for this library:
            user_file="ForLDA_"+str(lib_id)+".txt"
            user_paper_TFvec=create_TFV_matrix(user_file)
            #save the user_paper TFV into h5py
            store_sparse_mat2(user_paper_TFvec,"TFM",store="lib_"+str(lib_id)+".h5")
            subprocess.call("aws s3 cp *.h5 s3://usrlibrary-data",shell=True)    
        except:
            pass

 
def create_TFV_matrix(user_file): 
    with open("/home/ubuntu/lda-data/Current_data/CurrentWordDict.p", 'rb') as infile:
        words_dict = pickle.load(infile)
    #import the cnt Vectorizer and create the model
    from sklearn.feature_extraction.text import CountVectorizer
    stp_list1=open('/home/ubuntu/Calculating_Rec_Scores/en.txt','r').read().split('\n')
    stoplist=[x.strip(' ') for x in (stp_list1)] 
    cntvectorizer=CountVectorizer(vocabulary=words_dict,decode_error=u'ignore', encoding='latin-1', stop_words=stoplist,strip_accents='ascii') 
    user_file_df=pd.read_csv(user_file,sep='\t',header=None,names=['pmid','title','abstract'])#
    #user_file_df=pd.read_csv(user_file,usecols=[0,2,6],sep='\t',header=None,skiprows=1,names=['pmid','title','abstract'])# check to make sure of the columns.
    user_file_df['title_and_abs']=user_file_df['title']+user_file_df['abstract']
    user_pmids_list=list(user_file_df['pmid'])
    user_file_text=user_file_df['title_and_abs']
    user_corpus=list(user_file_text)
    user_paper_TFvec=cntvectorizer.transform(user_corpus)
    return(user_paper_TFvec)
          
        
def update_pytables_for_libraries():
    #find out which libraries changed and update their pytables    
    mysql_con= MySQLdb.connect(host='XXXXXXX', user='YYYYYY', passwd='%%%%%', db='*******')
    q='select * from library_libraries'
    data=psql.frame_query(q,con=mysql_con)
    #make a list of new libraries added in the past day
    today=datetime.now().date()
    new_lib=[data.id[i] for i in range(0,len(data.DateCreated)) if ( (data.DateCreated.notnull()[i]) and  (today- data.DateCreated[i].date()<timedelta(2)))]
    for lib in new_lib:
        try:
            create_pytable_for_single_library(lib)
        except:#no pmids in lib)
            pass 
    #make a list of libraries that have  changed..
    lib_changed=[]
    #library has changed if dtedit is NOT null (ie NaT) and today - dtedit.date()<2  
    lib_changed=[data.id[i] for i in range(0,len(data.dtedit)) if ( (data.dtedit.notnull()[i]) and  (today- data.dtedit[i].date()<timedelta(2)))]
    if len(lib_changed)>0:
        #get the updated library_libraries data:
        new_q='select * from library_library_papers'
        library_papers_data=psql.frame_query(new_q,con=mysql_con)     
    for lib in lib_changed:
        get_olddata_from_s3(lib)  
        try:
            outpath='/home/ubuntu/usrlibrary-data/libid_'+str(lib)         
            oldpmid_list=list(pd.read_hdf('lib_'+str(lib)+".h5",'lib'+str(lib)+'_pmid_list')[0])
            newpmid_list=list(library_papers_data.dbid[library_papers_data.libid_id == lib])
        except:
             pass # this library has no papers             
        #see how pmid lists have changed:
        if set(oldpmid_list).issubset(newpmid_list):
            if len(oldpmid_list)<len(newpmid_list): 
            #this is true only iff papers have been added and none removed
                added_papers=list(set(newpmid_list)-set(oldpmid_list))
                getting_users_papers_for_lda(added_papers,'new_papers',outpath)
                #append new Uniq to the h5 file:
                tmpUniq= pd.read_csv(outpath+"/Uniq_"+str(lib_id)+".txt",sep='\t',header=None,skiprows=1)
                tmpUniq.to_hdf("lib_"+str(lib_id)+".h5","lib"+str(lib_id)+"_uniqAll",mode="a",append=True,format="table",compress="blosc")
                #run lda on the new papers
                run_lda_user(outpath+"/ForLDA_new_papers.txt",str(lib)+'new_papers')# Run topic inferencer
                tmplda= pd.read_csv(outpath+"/"+str(lib)+"new_papers"+"LDA_composition_with_pmid.txt",usecols=range(0,21),sep='\t',header=None)
                #write/append lda to h5 ile
                tmplda.to_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_lda_data",mode="a",append=True,format="table",compress="blosc") 
                #append added_papers to the pmid_list in h5 
                added_papers_df=pd.DataFrame(added_papers) 
                added_papers_df.to_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_pmid_list",mode="a",append=True,format="table",compress="blosc")
                #for the TFV sparse matrix, delete the old matrix,remake it,and rewrite it.
                user_file="ForLDA_"+str(lib)+".txt"
                user_paper_TFvec=create_TFV_matrix(user_file) 
                overwrite_old_sparse_mat2(user_paper_TFvec, "TFM", store="lib_"+str(lib)+".h5")
            else:    
                #this means the oldpmid_list=newpmid_list, user added and then removed the paper
                #so do nothing let the file remain as as.
                pass                 
        elif set(newpmid_list).issubset(oldpmid_list) and len(newpmid_list)<len(oldpmid_list):
                #this is true only iff user removed papers 
                removed_papers=list(set(oldpmid_list)-set(newpmid_list))
                #get index of removed papers 
                indx_to_remove=[i for i,j in enumerate(oldpmid_list) if j in removed_papers]
                #read in the old Uniq file and drop removed papers rows
                tmpUniq_df=pd.read_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_uniqAll")
                tmpUniq_df.drop(tmpUniq_df.index[indx_to_remove],inplace=True)
                tmpUniq_df.to_hdf("lib_"+str(lib_id)+".h5","lib"+str(lib_id)+"_uniqAll",mode="a",format="table",compress="blosc") 
                #first read in the lda file:
                lda_df=pd.read_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_lda_data")
                lda_df.drop(lda_df.index[indx_to_remove],inplace=True) 
                lda_df.to_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_lda_data",mode="a",format="table",compress="blosc")
                #read in the pmid part, edit it and rewrite
                pmid_df=pd.read_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_pmid_list")
                pmid_df.drop(pmid_df.index[indx_to_remove],inplace=True) 
                pmid_df.to_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_pmid_list",mode="a",format="table",compress="blosc")
        else:
                #user added and removed papers
                #remake the h5 file -simplest and cleanest..
                create_pytable_for_single_library(lib)
        subprocess.call("aws s3 cp *.h5 s3://usrlibrary-data",shell=True)    
        

         
def get_olddata_from_s3(lib):
    #get the old saved data for library where dtedit is changed,           
    AWS_ACCESS_KEY_ID = '@@@@@@@@@@@'
    AWS_SECRET_ACCESS_KEY = 'SZXDCGVYBHUJNIKM'
    s3 = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,host="ZSXDCFVGHBJN")
    bucket=s3.get_bucket('cfvgbhn') 
    bucket_list=bucket.list()
    #get saved file from s3      
    outpath='/home/ubuntu/usrlibrary-data/libid_'+str(lib)
    if not os.path.exists(outpath):
        os.makedirs(outpath)   
    os.chdir(outpath) 
    teststr='lib_'+str(lib)+".h5" 
    for l in bucket_list: 
        #make sure lib exist in S3 
        if teststr== str(l.key):         
            l.get_contents_to_filename(teststr)
    return
       
    
def store_sparse_mat2(m, name, store='store.h5'):
    '''
    This function writes out the TFV in the TMV group of the h5 file. Call when first creating the matrix/table
    '''
    msg = "This code only works for csr matrices"
    assert(m.__class__ == sparse.csr.csr_matrix), msg
    with tb.open_file(store,'a') as f:
    # f=tb.open_file(store,"a")
        grp=f.create_group(f.root,"TMV_data")
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass
            arr = array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(grp, full_name, atom, arr.shape)
            ds[:] = arr
    f.close()


def overwrite_old_sparse_mat2(m, name, store='store.h5'):
    '''
    This function overwrites the existing TFV in the TMV group of the file. Only call when rewriting the updated matrix
    '''
    msg = "This code only works for csr matrices"
    assert(m.__class__ == sparse.csr.csr_matrix), msg
    with tb.open_file(store,'a') as f:
        f=tb.open_file(store,"a")
        #remove old TFV file
        f.remove_node('/','TMV_data',recursive=True)
        grp=f.create_group(f.root,"TMV_data")
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
               n = getattr(f.root, full_name)
               n._f_remove()
            except AttributeError:
               pass
            arr = array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(grp, full_name, atom, arr.shape)
            ds[:] = arr
    f.close()



def load_sparse_mat2(name, store='store.h5'):
    with tb.open_file(store,"a") as f:
        pars = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            pars.append(getattr(f.root.TMV_data, '%s_%s' % (name, par)).read())
    m = sparse.csr_matrix(tuple(pars[:3]), shape=pars[3])
    return m

    
    
    
def create_updated_termFreqVec_AND_calcSimilarity(corpus_pmid_list,user_file):
    '''
    This function reads in the updated Corpus Term Frequency Vector(which is updated as a cron job), and then find the new papers uploaded int eh past day from
    Current_basic_dataset_with_IF_and_LDA.csv
    '''
    with open("/home/ubuntu/lda-data/Current_data/CurrentWordDict.p", 'rb') as infile:
        words_dict = pickle.load(infile)
    with open("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrix",'rb') as infile:
        TermFreqMat = pickle.load(infile)
    with open("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrixPmid.p",'rb') as infile:
        TermFreqMatPmid = pickle.load(infile)
   # return words_dict, TermFreqMat,TermFreqMatPmid
   # chk if true ...in general; if true then nothing needs to be done.
   # (set(list(TermFreqMatPmid))).issuperset(set(pr.pmid_list(corp)))
   # if len(set(pr.pmid_list(corp))-set(list(TermFreqMatPmid)))>0:
   #     new_pmids_for_term_freq=list(set(corpus_pmid_list)-set(list(TermFreqMatPmid)))
   # OR
    #it is possible that there are some papers(pmids) which made it to LDA and thus to the corpus_pmid_list, 
   #but have either title or abstract missing and best to remove them from the corpus_pmid_df. 
   #This will happen only iff (set(list(corp_pmid_df.pmid)).issubset(set(TermFreqPmid)) is False?
    if len(set(corpus_pmid_list)-set(TermFreqMatPmid))>0:
        remove_list=list(set(corpus_pmid_list)-set(TermFreqMatPmid))
        #this will only keep those pmids in corpus that are also in TermFreqMatPmid
        updated=[pmid for pmid in corpus_pmid_list if pmid not in remove_list]
        corpus_pmid_list=updated
    corp_pmid_df=pd.DataFrame(corpus_pmid_list,columns=['pmid'])
    TermFreqPmid_df=pd.DataFrame(TermFreqMatPmid,columns=['pmid'])
    TermFreqPmid_df['tmp_ind']= TermFreqPmid_df.index
    tmp_df=pd.merge(corp_pmid_df,TermFreqPmid_df,how='left',on='pmid',sort=False)
    #list of indexes for which we need term freq data=list(tmp_df.tmp_ind)
    #its possible that there are some papers which made it to LDA and thus to the corpus_pmid_list, 
    #but have either title or abstract missing
    #convert the TremFreqMat to csr format and  
    TFV_for_corpus=TermFreqMat.tocsr()[(list(tmp_df.tmp_ind)),:]
    from sklearn.feature_extraction.text import TfidfTransformer
    import scipy as sp
    vectorizer=TfidfTransformer()
    X=vectorizer.fit_transform(TFV_for_corpus)  
    from sklearn.feature_extraction.text import CountVectorizer
    stp_list1=open('/home/ubuntu/Calculating_Rec_Scores/en.txt','r').read().split('\n')
    stoplist=[x.strip(' ') for x in (stp_list1)] 
    cntvectorizer=CountVectorizer(vocabulary=words_dict,decode_error=u'ignore', encoding='latin-1', stop_words=stoplist,strip_accents='ascii')
    user_file_df=pd.read_csv(user_file,sep='\t',header=None,names=['pmid','title','abstract'])#
    #user_file_df=pd.read_csv(user_file,usecols=[0,2,6],sep='\t',header=None,skiprows=1,names=['pmid','title','abstract'])# check to make sure of the columns.
    user_file_df['title_and_abs']=user_file_df['title']+user_file_df['abstract']
    user_pmids_list=list(user_file_df['pmid'])
    user_file_text=user_file_df['title_and_abs']
    user_corpus=list(user_file_text)
    user_paper_TFvec=cntvectorizer.transform(user_corpus)
    user_paper_Idfvec=vectorizer.transform(user_paper_TFvec)
    from sklearn.metrics.pairwise import linear_kernel  
    num_samples,num_features=X.shape
    num_user_papers,num_features=user_paper_TFvec.shape
    Cos_dist_array=[[]for i in range(0,num_user_papers)]
    for j in range(0,num_user_papers):
         Cos_dist_array[j] = linear_kernel(user_paper_TFvec[j:j+1], X).flatten()
    return(np.asarray(Cos_dist_array)),corpus_pmid_list

    

def create_updated_termFreqVec_and_calcSimilarity2(corpus_pmid_list,user_h5_file):
    '''
    This updated function reads in the updated Corpus Term Frequency Vector(which is updated as a cron job), and reads in the 
    updated TFV for the users papers, creates TFIDF vectors for them, and calculates the similarity matrix 
    '''
    with open("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrix",'rb') as infile:
        TermFreqMat = pickle.load(infile)
    with open("/home/ubuntu/lda-data/Current_data/CurrentTermFreqMatrixPmid.p",'rb') as infile:
        TermFreqMatPmid = pickle.load(infile)
   # return words_dict, TermFreqMat,TermFreqMatPmid
   # chk if true ...in general; if true then nothing needs to be done.
   # (set(list(TermFreqMatPmid))).issuperset(set(pr.pmid_list(corp)))
   # if len(set(pr.pmid_list(corp))-set(list(TermFreqMatPmid)))>0:
   #     new_pmids_for_term_freq=list(set(corpus_pmid_list)-set(list(TermFreqMatPmid)))
   # OR
    #it is possible that there are some papers(pmids) which made it to LDA and thus to the corpus_pmid_list, 
   #but have either title or abstract missing and best to remove them from the corpus_pmid_df. 
   #This will happen only iff (set(list(corp_pmid_df.pmid)).issubset(set(TermFreqPmid)) is False?
    if len(set(corpus_pmid_list)-set(TermFreqMatPmid))>0:
        remove_list=list(set(corpus_pmid_list)-set(TermFreqMatPmid))
        #this will only keep those pmids in corpus that are also in TermFreqMatPmid
        updated=[pmid for pmid in corpus_pmid_list if pmid not in remove_list]
        corpus_pmid_list=updated
    corp_pmid_df=pd.DataFrame(corpus_pmid_list,columns=['pmid'])
    TermFreqPmid_df=pd.DataFrame(TermFreqMatPmid,columns=['pmid'])
    TermFreqPmid_df['tmp_ind']= TermFreqPmid_df.index
    tmp_df=pd.merge(corp_pmid_df,TermFreqPmid_df,how='left',on='pmid',sort=False)
    #list of indexes for which we need term freq data=list(tmp_df.tmp_ind)
    #its possible that there are some papers which made it to LDA and thus to the corpus_pmid_list, 
    #but have either title or abstract missing
    #convert the TremFreqMat to csr format and  
    TFV_for_corpus=TermFreqMat.tocsr()[(list(tmp_df.tmp_ind)),:]
    from sklearn.feature_extraction.text import TfidfTransformer
    import scipy as sp
    vectorizer=TfidfTransformer()
    X=vectorizer.fit_transform(TFV_for_corpus)  
    user_paper_TFvec=load_sparse_mat2("TFM",store=user_h5_file)
    user_paper_Idfvec=vectorizer.transform(user_paper_TFvec)
    from sklearn.metrics.pairwise import linear_kernel  
    num_samples,num_features=X.shape
    num_user_papers,num_features=user_paper_TFvec.shape
    Cos_dist_array=[[]for i in range(0,num_user_papers)]
    for j in range(0,num_user_papers):
         Cos_dist_array[j] = linear_kernel(user_paper_TFvec[j:j+1], X).flatten()
    return(np.asarray(Cos_dist_array)),corpus_pmid_list

    
  

        
def vocab_list_Similarity_calculator_test(topically_sim_papers_pmids_list,user_file):
    '''
    This function takes in  files containing the Current pmids/text and creates a word vocabulary list and converts each paper in the Current paper list 
    into a word frequency vector. For each day; we will have ONE word vocabulary vector(??? Is this true?).Am creating the word sim vector using 
    ONLY the topically similar paper list that I get from Natalie? This function will have as input: a list of the pmids of shortlisted papers from the current corpus, 
    user_file (eg. ForLDA_1.txt) which has text info about the users papers (pmid,title, text). Natalie wants the output to be the similarity matrix. So output that.
    **Output is a file with recommendations. Will use nltk to create the TFIDF vector.     
    ''' 
    #Note the Word_Sim_data file is the same for every user: it is '/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv'
    WordSim_data_file='/home/ubuntu/lda-data/Current_data/Current_WordSim_data.csv'
    WordSim_df=pd.read_csv(WordSim_data_file, sep='\t',skiprows=1,header=None,names=['pmid','title','abstract'])
    tmp_topically_sim_papers_df= pd.DataFrame(list(topically_sim_papers_pmids_list),columns=['pmid'])     
    topically_sim_papers_df=pd.merge(tmp_topically_sim_papers_df,WordSim_df,how='left',on='pmid',sort=False)  
    #topically_sim_papers_df['pmid'][topically_sim_papers_df['title'].isnull()])
    '''topically_sim_papers_df.dropna(subset=['pmid','title','abstract'],how='any', inplace=True)'''
    #Now to calculate similarities between abstracts of usr lib paper and pmids topically similar to it(from above)
    #First need to combine Titles and Abstracts:    
    topically_sim_papers_df['title_and_abs']= topically_sim_papers_df['title']+ topically_sim_papers_df['abstract']
    topically_sim_papers_pmids_list=list(topically_sim_papers_df['pmid'])
    topically_sim_papers_text=topically_sim_papers_df['title_and_abs']
    text_corpus=list(topically_sim_papers_text)
    #create stoplist:
    stp_list1=open('/home/ubuntu/Calculating_Rec_Scores/en.txt','r').read().split('\n')
    stoplist=[x.strip(' ') for x in (stp_list1)] 
    #using the tfidf Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy as sp
    vectorizer=TfidfVectorizer(min_df=1, decode_error=u'ignore', encoding='ascii', stop_words=stoplist,strip_accents='ascii')
    X=vectorizer.fit_transform(str(line) for line in text_corpus)
    user_file_df=pd.read_csv(user_file,sep='\t',header=None,names=['pmid','title','abstract'])#
    #user_file_df=pd.read_csv(user_file,usecols=[0,2,6],sep='\t',header=None,skiprows=1,names=['pmid','title','abstract'])# check to make sure of the columns.
    user_file_df['title_and_abs']=user_file_df['title']+user_file_df['abstract']
    user_pmids_list=list(user_file_df['pmid'])
    user_file_text=user_file_df['title_and_abs']
    user_corpus=list(user_file_text)
    user_paper_vec=vectorizer.transform(user_corpus)
    user_paper_vec.toarray()
    #vectorizer.get_feature_names()
    #since its a sparse matrix, cannot really use scipy distance metrics     
    from sklearn.metrics.pairwise import linear_kernel  
    num_samples,num_features=X.shape
    num_user_papers,num_features=user_paper_vec.shape
    Cos_dist_array=[[]for i in range(0,num_user_papers)]
    for j in range(0,num_user_papers):
         Cos_dist_array[j] = linear_kernel(user_paper_vec[j:j+1], X).flatten()
    return(np.asarray(Cos_dist_array))
  
def create_basic_jour_IF():
    '''
    This converts all the published Journal IF data into a table.Turns out some of the Journal Data has duplicates: for those cases, I 
    pick the one with max IF factor, and drop the others. IF of journal with IF=NULL or 0 is converted to 1. eLife is given an IF =15. 
    ''' 
    mysql_con= MySQLdb.connect(host='szwsexdcrfvtgbhnjm', user='zsxdcfvgbhnj', passwd='ghbjnkm', db='asdfg')
    #Can I do this in MySQL
    ##?r="select distinct(j1.ISSN), j1.IF  from journals_impactfactor as j1,journals_impactfactor as j2 where ( j1.ISSN=j2.ISSN and j1.IF>j2.IF) " 
    #Do it in pandas -more straightforward
    r="select journals_impactfactor.ISSN,journals_impactfactor.IF  from journals_impactfactor"
    if_jourdata=psql.frame_query(r,mysql_con)
    aa=if_jourdata.set_index('ISSN').index.get_duplicates()
    for ISSN in aa:
        bb=if_jourdata.IF[if_jourdata.ISSN == ISSN]
        if_jourdata.IF[if_jourdata.ISSN == ISSN]=max(bb)
    #Remove the duplicate rows.
    if_jourdata.drop_duplicates(cols='ISSN',inplace=True)  
    #There are journals with IF=0.0,-convert the IF of these journals into 1.
    if_jourdata.IF[if_jourdata.IF==0.0]=1.00
    #This file contains all the journals that are in the published list
    if_jourdata.to_csv("/home/ubuntu/lda-data/Journal_IF_data.csv",index=False)
    
def update_jour_IF(dataframe):      
    '''
    This function reads in the IF file from "/home/ubuntu/lda-data/Journal_IF_data.csv" and then updates it with any new journals that are in the current dataset that 
    are NOT in the IF file, with an IF =1.0. If journal is eLife give if IF=15    
    '''
    if_jourdata= pd.read_csv("/home/ubuntu/lda-data/Journal_IF_data.csv")
    ISSN_uniq=set(pd.unique(dataframe.ISSNLinking))
    new_ISSN=ISSN_uniq-set(if_jourdata.ISSN)
    tmp_df=pd.DataFrame(list(new_ISSN)) 
    tmp_df.columns=['ISSN']
    tmp_df.dropna(inplace=True)
    tmp_df['IF']=1.00
    if len(tmp_df.ISSN)>0:
              tmp_df.IF[tmp_df.ISSN=='2050-084X']=15.00
    update_ifdata=pd.concat([if_jourdata,tmp_df],ignore_index=True) #Note: concat is a better option than merge as they are disjoint sets
    update_ifdata.to_csv("/home/ubuntu/lda-data/Journal_IF_data.csv", index=False)
    


def getting_users_pmids(bibtex):
    bp = BibTexParser(open(bibtex,'r').read())
    dicts = bp.get_entry_dict()
    pmids=[]
    for key in dicts.keys():
        try:
		    pmids.append((dicts[key]["pmid"]).encode('utf-8'))
        except:
	    	pmids.append(0)	    	
    return pmids



def getting_usrpubRecs_andforLDA(pmids,user_name,pathname):
    '''
    this function gets the abstract, title, and relevant data for the pmids in the users library and save the data as a paper class.DEPRECATE SOON...
    see getting users_papers_and_lda_data
    
    '''
    q='select p.pmid, p.DateCreated,p.ArticleTitle, p.JournalTitle,p.ISSN,p.ISSNLinking ,a.AbstractText, a.AbstractOrder from pubmed_pubmed as p inner join pubmed_abstract as a on p.pmid = a.pmid_id where p.pmid in (%s)'        
    mysql_con= MySQLdb.connect(host='asdfghjjhkg', user='asdfghj', passwd='aSDZFXGCHJ', db='ASdzfxgh')
    str_list=','.join(['%s']*len(pmids))
    q=q % str_list
    user_data=psql.frame_query(q,params = pmids,con=mysql_con)
    nrow,ncol=user_data.shape 
    uniq_pmids=pd.unique(user_data.pmid)    
    user_data_df=user_data.drop_duplicates(cols=['pmid'])
    pmids_list=list(user_data.pmid)
    full_abs=[]
    count=0
    for i in uniq_pmids:
        order_cnt=max(user_data.AbstractOrder[user_data.pmid==i])
        if order_cnt>1:
            tmp_abs=''
            for k in range(1,order_cnt+1):
                tmp_abs=tmp_abs+str(pd.unique(user_data.AbstractText[(user_data.pmid==i) & (user_data.AbstractOrder==k)])[0])   
            full_abs.append(tmp_abs) 
        else:
            full_abs.append(user_data_df.AbstractText[user_data_df.pmid==i].values[0])
    user_data_df.drop(['AbstractText'],axis=1, inplace=True)  
    user_data_df.drop(['AbstractOrder'],axis=1,inplace=True)
    user_data_df['AbsText']=full_abs
    pmids_df=pd.DataFrame(pmids,columns=['pmid'])
    new_user_data=pd.merge(pmids_df, user_data_df, how="left",on="pmid",sort=False)                       
    user_data_df.to_csv(pathname+'/Uniq_'+user_name+'.txt',sep='\t',index=False)#write out the file in format to store
    #write out info for each paper into a pubRec.
    lib=[]
    for index, row in  user_data_df.iterrows():   
        paper=pr.pubRecords()
        paper.pmid=row['pmid']    
        paper.pmid=row['pmid']
        paper.date=row['DateCreated']
        paper.title=row['ArticleTitle']
        paper.issn=row['ISSN']
        paper.issnl=row['ISSNLinking']
        paper.abstract=row['AbsText'] 
        lib.append(paper)
   #write out another file with just text for LDA.
    print "writing out " + user_name +" data  to compute lda probabilities"
    user_data_df.to_csv(pathname+'/ForLDA_'+user_name+'.txt',sep='\t',index=False,cols=['pmid','ArticleTitle','AbsText'],header=False)
    print "done lda for ", user_name     
    return lib,list(user_data_df.pmid)    
       



def getting_users_paper_info_lda_pmid_uniq(lib,outpath):
    '''
    This function gets the abstract, title, date and if(relevant data) for the pmids in the users library from the h5 file and read them into the papers class.
    Make sure you take nly the intersection of pmids...as soem have abstract/title etc missing
    variable.
    '''
    os.chdir(outpath)
    lda_df=pd.read_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_lda_data")
    uniq_df=pd.read_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_uniqAll")
    pmid_df=pd.read_hdf("lib_"+str(lib)+".h5","lib"+str(lib)+"_pmid_list")
 #  pmid_df[0]=map(lambda x: int(x),pmid_df[0])
    #get the intersection of the pmids -that is this libraries pmid ..note you have to convert the pmid_list from string to int
   #write out info for each paper into a pubRec.
    lib=[]
    for index, row in  uniq_df.iterrows():   
        paper=pr.pubRecords()
        paper=pr.pubRecords()
        paper.pmid=row[0]
        paper.date=row[1]
        paper.title=row[2]
        paper.issn=row[4]
        paper.issnl=row[5]
        paper.abstract=row[6] 
        lib.append(paper)       
    print "done reading values to the paper class. Now to read in lda values for top 10"
    #read in the LDA 
    for index,row in lda_df.iterrows():
        lib[index].tpcs[lda_df[1][index]]=lda_df[2][index]
        lib[index].tpcs[lda_df[3][index]]=lda_df[4][index]
        lib[index].tpcs[lda_df[5][index]]=lda_df[6][index] 
        lib[index].tpcs[lda_df[7][index]]=lda_df[8][index]
        lib[index].tpcs[lda_df[9][index]]=lda_df[10][index] 
        lib[index].tpcs[lda_df[11][index]]=lda_df[12][index] 
        lib[index].tpcs[lda_df[13][index]]=lda_df[14][index] 
        lib[index].tpcs[lda_df[15][index]]=lda_df[16][index] 
        lib[index].tpcs[lda_df[17][index]]=lda_df[18][index] 
        lib[index].tpcs[lda_df[19][index]]=lda_df[20][index]
    return lib,list(lda_df[0])    
       



def getting_users_papers_for_lda(pmids,user_name,pathname):   
    q='select p.pmid, p.DateCreated,p.ArticleTitle, p.JournalTitle,p.ISSN,p.ISSNLinking ,a.AbstractText, a.AbstractOrder from pubmed_pubmed as p inner join pubmed_abstract as a on p.pmid = a.pmid_id where p.pmid in (%s)'        
    mysql_con= MySQLdb.connect(host='asdfghj', user='asdfgchvj', passwd='sdfcghvbjnk,', db='ASGYHJK')
    str_list=','.join(['%s']*len(pmids))
    q=q % str_list
    user_data=psql.frame_query(q,params = pmids,con=mysql_con)
    nrow,ncol=user_data.shape 
    uniq_pmids=pd.unique(user_data.pmid)
    user_data_df=user_data.drop_duplicates(cols=['pmid'])
    pmids_list=list(user_data.pmid)
    full_abs=[]
    count=0
    for i in uniq_pmids:
        order_cnt=max(user_data.AbstractOrder[user_data.pmid==i])
        if order_cnt>1:
            tmp_abs=''
            for k in range(1,order_cnt+1):
                tmp_abs=tmp_abs+str(pd.unique(user_data.AbstractText[(user_data.pmid==i) & (user_data.AbstractOrder==k)])[0])   
            full_abs.append(tmp_abs) 
        else:
            full_abs.append(user_data_df.AbstractText[user_data_df.pmid==i].values[0])
    user_data_df.drop(['AbstractText'],axis=1, inplace=True)  
    user_data_df.drop(['AbstractOrder'],axis=1,inplace=True)
    user_data_df['AbsText']=full_abs            
    user_data_df.to_csv(pathname+'/Uniq_'+user_name+'.txt',sep='\t',index=False)#write out the file in format to store
    #write out another file with just text for LDA.
    print "writing out " + user_name +" data  to compute lda probabilities"
    user_data_df.to_csv(pathname+'/ForLDA_'+user_name+'.txt',sep='\t',index=False,cols=['pmid','ArticleTitle','AbsText'],header=False)
    print "done lda for ", user_name
    

def calculateRunTime(function, *args):
    import time
    '''run a function and return the run time and the result of the function
     if the function requires arguments, those can be passed in too '''
    startTime = time.time()
    result = function(*args)
    return result,time.time() - startTime,
    
    
'''
#### Possible alternative for time consuming table join in above function?

    q_pubmed='select pmid, DateCreated,ArticleTitle, JournalTitle,ISSN,ISSNLinking from pubmed_pubmed where pmid in (%s)'        
    q_pubmed=q_pubmed % str_list
    user_pubmed_data=psql.frame_query(q_pubmed,params = pmids,con=mysql_con)
    q_abstract='select pmid_id, AbstractText,AbstractOrder from pubmed_abstract where pmid_id in (%s)'
    q_abstract=q_abstract % str_list
    user_abstract_data=psql.frame_query(q_abstract,params = pmids,con=mysql_con)
    user_data=pd.merge(user_pubmed_data,user_abstract_data,left_on="pmid",right_on="pmid_id",sort=False)
'''

def run_lda_user(file,user_name):
    subprocess.call(['/home/ubuntu/Scirec_scripts/lda_scripts/run_LDA.sh',file])
    shutil.move('output_file_tpcs_composition_for_database.txt',user_name+'LDA_composition_for_database.txt')
    shutil.move('output_file_tpcs_composition_with_pmid.txt',user_name+'LDA_composition_with_pmid.txt')                
