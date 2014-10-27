#! /usr/local/bin/python
#usage: python parse_pubmed.py 'mallet'  'pubmed_2013.xml'  >> fileout ; where pubmed_2013.xml is the xml file. This prints out in utf-8 and then you can pipe it to a file 
# This script prints out title, abstract (if >100 ch) and keywords and only iff all three are present.

import sys
import re
import xml.etree.cElementTree as ET
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)



def iterArticles(f):
    pmid = None
    title = None
    abstract = None
    journal= None
    mesh_list = []
    for event,elem in ET.iterparse(f, events=("start","end")):
        if event == 'start':
            if elem.tag == 'PubmedArticle':
                pmid,title,abstract,journal,mesh_list= None,None,None, None,[]
        elif event == 'end':            
            if elem.tag == 'PubmedArticle':
                yield pmid, title, abstract,journal,mesh_list
            elif elem.tag == 'PMID':
                pmid = elem.text
            elif elem.tag == 'ArticleTitle':
                title = elem.text               
            elif elem.tag == 'AbstractText':
                abstract = elem.text
            elif elem.tag == 'Title':
                journal = elem.text    
            elif elem.tag == 'KeywordList':
                keyword_list=elem.findall("Keyword")
                for aa in keyword_list:
                    mesh_list.append(aa.text)
            elif elem.tag == 'MeshHeadingList':
                mhlist = elem.findall("MeshHeading")  
                for child in mhlist:
                    if child.findtext('DescriptorName'):
                         mesh_list.append(child.findtext('DescriptorName'))
                    if child.findtext('QualifierName'):     
                         mesh_list.append(child.findtext('QualifierName'))
         
def main(args):
    if len(args) == 1:
        return
    
    method,f = args[1:3]
    if method == 'mallet':
        it = iterArticles(f)
        for p,t,a,j,k in it:
            # need to clean up text
            if t:
                t = re.sub('[|:\n\r]','',t)
            else:
                t = ''
            if a:
                a = re.sub('[|:\n\r]','',a)                                  
            if a and t and k:   # the abstract, title and keywords need to be present
          #  if a and t:  
                if len(a)>100:  #Only use if abstract is longer than 100 characters 
                   # print '%s |Title %s |Abstract  %s   |Journal  %s |Keywords %s' %(p,t,a,j,k)
                    print '%s"\t" %s %s %s' %(p,t,a,k)
  
if __name__ == "__main__":
    main(sys.argv)                
            

            
