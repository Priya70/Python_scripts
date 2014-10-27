#! /usr/bin/local/python
''' To read in the output for the parsed xml Pubmed output, convert it into 
"bag-of-words"(bow) format and the Blei lda-C format and create the vocabulary dictionary.
 I am reading in the txt file without the words "Abstract" and Title" in them and 
 without the | separator 
Usage is: python Gensim_output_topics.py pubmed__3tpcs.txt num_topics  update_freq  num_passes
   ../Gensim_output_topics.py  pubmed+ads.txt  4   0   300
   Note what we are doing is batch lda and so update freq=0  
  '''

from gensim import corpora, models, similarities
import gensim
import sys 
import csv
import re
import string

def main(args):
   #print "length of args is" , len(args)
    if len(args) < 2:
        return
    if len(args) >1  :
        infile=args[1]
        num_of_topics=int(args[2]) 
        update_freq=int(args[3])
        num_passes=int(args[4])
        stp_list1=open('/Users/prd/Topic_modelling/Stoplists/english_stoplist.txt','r').read().split('\n')
        stp_list2=(open('/Users/prd/Topic_modelling/Stoplists/English_stopwords.txt','r').read()).split(',')
        stoplist=[x.strip(' ') for x in (stp_list1+stp_list2)]
        line=open(infile).readline()
       # print [[re.sub("[\s\*\=\+\.)]","",i) for i in line.lower().split(' ')] for line in open(infile).readlines()]
       # word_set=[[re.sub("[\s\*\=\+\.\)\(-\:$\" ]","",i) for i in line.lower().split(' ')] for line in open(infile).readlines()]
       # print type(word_set)
        word_set=[[re.sub(r'[[.;:$\-\{\}()~`|&)!,\^\'\"\]]',"",i) for i in line.lower().split(' ')] for line in open(infile).readlines()]
        dictionary_out = corpora.Dictionary(word_set)

        stop_ids = [dictionary_out.token2id[stopword] for stopword in stoplist if stopword in dictionary_out.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary_out.dfs.iteritems() if docfreq == 1]
        
        dictionary_out.filter_tokens(stop_ids+once_ids) # filter out or remove stop words 
        dictionary_out.compactify() 
        
        #save dictionary in text readable format :_dictionary has the key-value pairs and _lda_c_vocab had the vocabulary that lda-c needs      
        dict_out=csv.writer(open(infile+'_dictionary','w'),delimiter=':')
        vocab_out=csv.writer(open(infile+'_lda_c_vocabulary','w'))
        for key,value in dictionary_out.token2id.items():
             dict_out.writerow([key,value])
             vocab_out.writerow([key])
  
        #save dictionary as a binary format
        dictionary_out.save(infile+'.dict')
        
        class OutCorpus(object):
            def __iter__(self):
                 for line in open(infile):
                 # assume there's one document per line, tokens separated by whitespace
                     yield dictionary_out.doc2bow(line.lower().split())   
        
        mycorpus=OutCorpus() #mycorpus is bow format
        outfile=open(infile+'_bow.txt','w')
        for doc in mycorpus:
            outfile.write(str(doc)+'\n')
        outfile.close()    
            
        corpora.BleiCorpus.serialize(infile+'_lda_c', mycorpus)  #convert into the format needed for Blei's lda-c routine
        corpora.MmCorpus.serialize(infile+'_corpus.mm', mycorpus)  #format used for  lsi method
        
        ##getting topics via LSI
        tfidf = models.TfidfModel(mycorpus) #initializing model
        mycorpus_tfidf=tfidf[mycorpus]
        lsi = models.LsiModel(mycorpus_tfidf, id2word=dictionary_out, num_topics=num_of_topics)  #initializing lsi  model
        corpus_lsi = lsi[mycorpus_tfidf]
        out=open(infile+'.lsi_topics','w')
        for line in corpus_lsi:  
            out.write(str(line))
            out.write('\n') 
        out.close()         
        
        out=open(infile+'topic_key.txt','w')
        for doc in lsi.show_topics():
            out.write(doc +'\n')   
        out.close()    
        
        #lsi.print_topics(2)
        #for doc in corpus_lsi: print doc
        
        
        ##getting topics via tfidf
        tfidf_model = models.TfidfModel(mycorpus, normalize=True) 
      
        ##lda model: this works best for batch lda  
        mm = gensim.corpora.MmCorpus(infile+'_corpus.mm') 
        lda= gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary_out, num_topics=num_of_topics, update_every=update_freq,  passes=num_passes)   
        doc_lda = lda[mycorpus]
        out=open(infile+'.lda_topics','w')
        for doc in doc_lda: 
            out.write(str(doc))
            out.write('\n')
        out.close() 
        
        out=open(infile+'topic_key.txt','w')
        for doc in lda.show_topics():
          	out.write(doc +'\n')
        out.close()    
if __name__ == "__main__":
    main(sys.argv)       