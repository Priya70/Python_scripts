#! /usr/bin/local/python
''' 
To read in the input training set file, and compare predicted classification with input data
calling Usage: compare_topics  path_name(where all the main txt files are) 
compare_topics '/Users/prd/Topic_Modelling/pubmed_10tpcsANDkeywds/'
'''
import sys
import re
import numpy as np
import argparse

'''
def init_parser():
    
    Returns an argument parser configured with options for this program
        parser = argparse.ArgumentParser(
            description='A program that get all the txt and result files, parses them and then plots the comparative reuslts'  
            )

    #positional argument
    parser.add_argument('txt_dirname', action='store',
        help='directory containing .txt files (files must be encoded in utf-8)')

    #options
    parser.add_argument('-m', '--mallet_dirname', action='store', dest='malletdir',
            help='directory where the mallet results reside')
    parser.add_argument('-g', '--gensim_dirname', action='store', dest='gensim_dir',
            help='directory where gensim results reside')
    parser.add_argument('-o',' --output_file a', action='store',
            dest='outfile',
            help='output_filename')    
    return parser.parse_args()
'''

def get_ids(path_name,file):
   return [line.split('|Title')[0] for line in open(path_name+file).readlines()]
    
def main(args):
   if len(args) == 0:
        print "Usage: python compare_topics.py path_name  "
   else:
     #   parser = init_parser()
        path_name=args[1]  #argument 1 is the pathname where all the text files reside


        cancer= get_ids(path_name,'cancer.txt')
        microfluidics= get_ids(path_name,'microfluidics.txt')
        gluten=get_ids(path_name,'gluten.txt')
        malaria=get_ids(path_name,'malaria.txt')
        schizophrenia=get_ids(path_name,'schizophrenia.txt')
        bioethics=get_ids(path_name,'bioethics.txt')
        southbeach=get_ids(path_name,'high_protein.txt')
        diabetes=get_ids(path_name,'juv_diabetes.txt')
        typhoid=get_ids(path_name,'typhoid.txt')
        chromatin=get_ids(path_name,'chromatin.txt')


        mallet_data=open('/Users/prd/Topic_Modelling/mallet-2.0.7/pubmed_10tpcs/pubmed10_compostion.txt', 'r').readlines() 
        gensim_data=open('/Users/prd/Topic_Modelling/Gensim/pubmed_10tpcs/pubmed_10tpcs.txt.lda_topics','r').readlines()
        outfile=open("10tpcs.txt",'w')
        gensim_tpids=[]
        gensim_tprob=[]
        orig_tpids=[]
        mallet_tpids=[]
        mallet_tprob=[]
        pubmed_ids=[]
        for gen in gensim_data:
           line=[re.sub('[\[\(\)\]\n]',"",tt) for tt in gen.split('), (')]   
           tpc_list=[]
           prob_list=[]
           for ind in line:
                tpc_list.append(int(ind.split(',')[0]))
                prob_list.append(float(ind.split(',')[1]))
           gensim_tpids.append(tpc_list[prob_list.index(max(prob_list))])      
           gensim_tprob.append(max(prob_list))
                    
        for i in mallet_data[1:]:
           string=i.split()
           pubmed_ids.append(string[1].split('|')[0])
           mallet_tpids.append(string[2])    
           mallet_tprob.append(string[3])
    
        for str in pubmed_ids:    
            if str in cancer:
               orig_tpid='0'   #cancer origid='0'       
               orig_tpids.append(orig_tpid)   
            elif str in schizophrenia:
               orig_tpid='1'   #schizophrenia origid='0'  
               orig_tpids.append(orig_tpid)       
            elif str in gluten:
               orig_tpid='2'   #gluten origid='2'   
               orig_tpids.append(orig_tpid) 
            elif str in bioethics:
               orig_tpid='3'   #bioethics origid='3'  
               orig_tpids.append(orig_tpid)             
            elif str in southbeach:
               orig_tpid='4'   #southbeach origid='4'     
               orig_tpids.append(orig_tpid)          
            elif str in diabetes:
               orig_tpid='5'   #diabetes origid='5'   
               orig_tpids.append(orig_tpid)            
            elif str in malaria:
               orig_tpid='6'   #malaria origid='6' 
               orig_tpids.append(orig_tpid)              
            elif str in microfluidics:
               orig_tpid='7'   #microfluidics origid='7'    
               orig_tpids.append(orig_tpid) 
            elif str in typhoid:
               orig_tpid='8'  
               orig_tpids.append(orig_tpid)
            elif str in chromatin:
               orig_tpid='9'  
               orig_tpids.append(orig_tpid)                 
            else:
               print "pubmed id has no topic?? ",str 
           
            print len(orig_tpids), len(gensim_tpids),len(pubmed_ids),len(mallet_tpids) 
#print orig_tpids        

            gensim_toplot=[]
            mallet_toplot=[]

            for xx in xrange(len(mallet_data)):
               gensim_toplot.append((gensim_tpids[xx])+(gensim_tprob[xx]))
               mallet_toplot.append((mallet_tpids[xx])+(mallet_tprob[xx]))
    
     
            out=open("outputall_topx.txt",'w')             
            out.write("PubmedId   OrigId    MalletId   MalletProb   GensimId    GensimProb \n")    
            for i in range(len(mallet_data)):
  #    new_str=pubmed_ids[i]+'    '+orig_tpids[i]+'    '+ mallet_tpids[i]+'     ' +mallet_tprob+'     '+gensim_tpids[i]     
 #       outfile.write(new_str+'\n')
               out.write('{0}  {1}  {2}  {3}  {4}  {5} \n'.format(pubmed_ids[i],orig_tpids[i],mallet_tpids[i], mallet_tprob[i], gensim_tpids[i], gensim_tprob[i]) )


            out.close()                 

if __name__ == "__main__":
    main(sys.argv)                
