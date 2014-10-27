'''To parse the Astronomy ads_abstracts.html file and convert each title and abstract into a separate 
txt file
run as : python parse_ads.py >
'''
import re
import itertools
import pickle

with open('ads_abstracts.txt','r') as f:  #with syntax makes sure that the file is closed in the end
     ads=f.read().split('%0') #makes a list of abstract_info as it splits the entire string at %0 which is the new abstract character 
     Titles=[]
     Abstracts=[]
     Pnumners=[]
     for item in ads:
           #finding titles
           tit_beg=item.find('%T')
           tit_end=item.find('%',tit_beg+1) #recall that it starts the search from position tit_beg+1,but returns the index as if counting from 0              
           tit=item[tit_beg+2:tit_end].strip()
           Titles.append(tit)
           #finding abstracts
           abs_beg=item.find('%X')
           abs_end=item.find('%',abs_beg+1)
           abstr=item[abs_beg+2:abs_end].replace('\n', '')
           Abstracts.append(abstr)
           Pnum_beg=item.find('%P')
           Pnum_end=item.find('%',Pnum_beg+1)
           Pnum=item[Pnum_beg+2:Pnum_end].strip()
           print "%s %s %s " %(Pnum,tit,abstr)
           out=open(Pnum+'.txt', 'w')
           out.write(tit)
           out.write(abstr)
           out.close()
           
     all_abs=zip(Titles,Abstracts) # creates a tuple of Titles and Abstracts
     outfile=open('ads_abstract_data','wb')  #open a file to write in binary format
     pickle.dump(all_abs,outfile)  #now the tuples have been saved as pickle
     outfile.close()
     #To open the pickled file:
     # all_abs=pickle.load(open('abstract_data','rb'))
    
               
            
               
         
     

#Build a dictionary of handlers: 

   
