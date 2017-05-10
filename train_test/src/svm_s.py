# -*- coding: utf-8 -*-
import jieba
import numpy as np
import os
import time
import codecs
import re
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

open=codecs.open
path_en='../After/En'
path_peo='../After/new_people'
content_train_src=[]
opinion_train_stc=[]
file_name_src=[]
train_src=[]
test_src=[]



def Chinese_Stopwords():  # 导入停用词库
    stopword = []
    cfp = open('../data/stopWord.txt', 'r+', 'utf-8')
    for line in cfp:
        for word in line.split():
            stopword.append(word)
    cfp.close()
    return stopword
stopwords=Chinese_Stopwords()

def readfile(path):
    for filename in os.listdir(path):
        strattime=time.time()
        filepath=path+"/"+filename
        filestr=open(filepath).read()
        opinion=path[9:]
        train_src.append((filename,filestr,opinion))
        endtime=time.time()
        print '类别:%s >>>>文件:%s >>>>导入用时: %.3f' % (opinion, filename, endtime - strattime)
    return train_src

train_src_all=readfile(path_en)
train_src_all=train_src_all+readfile(path_peo)

def readtrain(train_src_list):
    for (fn,w,s) in train_src_list:
        file_name_src.append(fn)
        content_train_src.append(w)
        opinion_train_stc.append(s)
    train=[content_train_src,opinion_train_stc,file_name_src]
    return train

def Word_pseg(word_str):  # 名词提取函数
    words = pseg.cut(word_str)
    word_list = []
    for wds in words:
        # 筛选自定义词典中的词，和各类名词，自定义词库的词在没设置词性的情况下默认为x词性，即词的flag词性为x
        if wds.flag == 'x' and wds.word != ' ' and wds.word != 'ns' \
                or re.match(r'^n', wds.flag) != None \
                        and re.match(r'^nr', wds.flag) == None:
            word_list.append(wds.word)
    return word_list

def Word_cut_list(word_str):
    word_str = re.sub(r'\s+', ' ', word_str)  # trans 多空格 to空格
    word_str = re.sub(r'\n+', ' ', word_str)  # trans 换行 to空格
    word_str = re.sub(r'\t+', ' ', word_str)  # trans Tab to空格
    #word_str=word_str.encode('utf-8').translate(None,string.punctuation)
    word_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".decode("utf8"), "".decode("utf8"), word_str)

    wordlist = list(jieba.cut(word_str))#jieba.cut  把字符串切割成词并添加至一个列表
    wordlist_N = []
    for word in wordlist:
        if word not in stopwords:#词语的清洗：去停用词
            if word != '\r\n'  and word!=' ' and word != '\u3000'.decode('unicode_escape') and word!='\xa0'.decode('unicode_escape'):#词语的清洗：去全角空格
                wordlist_N.append(word)
    return wordlist_N

def segmentWord(cont):
    listseg=[]
    for i in cont:
        Wordp = Word_pseg(i)
        New_str = ''.join(Wordp)
        Wordlist = Word_cut_list(New_str)
        file_string = ''.join(Wordlist)
        listseg.append(file_string)
    return listseg

train=readtrain(train_src_all)
content=segmentWord(train[0])
filenamel=train[2]
opinion=train[1]


train_content=content[:3000]
test_content=content[3000:]
train_opinion=opinion[:3000]
test_opinion=opinion[3000:]
train_filename=filenamel[:3000]
test_filename=filenamel[3000:]

test_all=[test_content,test_opinion,test_filename]


vectorizer=CountVectorizer()
tfidftransformer=TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  # 先转换成词频矩阵，再计算TFIDF值

print tfidf.shape



# 训练和预测一体
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=1, kernel = 'linear'))])
text_clf = text_clf.fit(train_content, train_opinion)
predicted = text_clf.predict(test_content)
print 'SVC',np.mean(predicted == test_opinion)
print set(predicted)
print metrics.confusion_matrix(test_opinion,predicted) # 混淆矩阵

