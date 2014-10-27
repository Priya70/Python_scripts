#! /usr/bin/local/python

'''To plot the results obtained running LDA. 

'''

import numpy
import matplotlib
import scipy
from pylab import *

plt.xlabel("Gensim")
plt.ylabel("Mallet")
plt.legend(loc=2,fontsize=8)
#plt.show() 
pylab.savefig('pubmed_8tpcs.png')