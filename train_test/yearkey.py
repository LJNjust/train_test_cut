import mgkey
import codecs
import os
import time

import matplotlib.pyplot as plt
from nltk.probability import FreqDist
open=codecs.open
year_begin=2003
year_over=2016
root_b='data/year/'
root_r='data/year_Result/'



for year in range(year_begin,year_over+1):
    starttime=time.time()
    root_b=root_b+str(year)
    year_list=[]
    pre = 0
    for path in os.listdir(root_b):
        #f=open(root_b+'/'+path,'r+','utf-8')
        kw=mgkey.keyword(root_b+'/'+path)
        wl=kw.Run()
        for w in wl:
            year_list.append(w)
    Sum=len(year_list)
    fdist=FreqDist(year_list)
    fn = open(root_r + '/' + str(year)+".txt", 'w+', 'utf-8')
    for (s,n)in kw.sort_item(fdist.items()):
        fn.write(s+str(float(n)/Sum)+'\r\n')
        pre=pre+float(n)/Sum
        if pre>0.5:
            fn.write(str(pre))
            fn.close()
            break
    endtime=time.time()
    print str(year)+"needtime:"+str(endtime-starttime)
    root_b = 'data/year/'


