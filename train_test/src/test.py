## -*- coding: utf-8 -*-
import jieba
import os
import re
import time
import string
import codecs
from  sklearn.feature_extraction.text import TfidfTransformer
from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn import  preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix,classification_report
import matplotlib.pyplot as plt


from  sklearn.feature_selection import  SelectKBest,chi2
import pandas as pd

import numpy as np



open=codecs.open

def Chinese_Stopwords():  # 导入停用词库
    stopword = []
    cfp = open('../data/stopWord.txt', 'r+', 'utf-8')
    for line in cfp:
        for word in line.split():
            stopword.append(word)
    cfp.close()
    return stopword


rootdir='../After'
os.chdir(rootdir)
#stopword
words_list=[]
filename_list=[]
category_list=[]
all_words={}
stopwords=Chinese_Stopwords()
category=os.listdir(rootdir)
delEStr = string.punctuation + ' ' + string.digits
identify = string.maketrans('', '')

def guiyi(x):
    x[x>1]=1
    return x
def fileWordProcess(contents):
    wordsList = []
    contents = re.sub(r'\s+',' ',contents) # trans 多空格 to 空格
    contents = re.sub(r'\n',' ',contents)  # trans 换行 to 空格
    contents = re.sub(r'\t',' ',contents)  # trans Tab to 空格
    #contents = contents.decode('utf-8').translate(identify, delEStr)
    contents = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".decode("utf8"),
                      "".decode("utf8"), contents)

    for seg in jieba.cut(contents):
        seg = seg.encode('utf8')
        if seg not in stopwords:           # remove 停用词
            if seg!=' ':                   # remove 空格
                wordsList.append(seg)      # create 文件词列表
    file_string = ''.join(wordsList)
    return file_string

for categoryName in category:
    if (categoryName == '.DS_Store'): continue
    categoryPath = os.path.join(rootdir, categoryName)  # 这个类别的路径
    filesList = os.listdir(categoryPath)  # 这个类别内所有文件列表
    # 循环对每个文件分词
    for filename in filesList:
        if (filename == '.DS_Store'): continue
        starttime = time.clock()
        contents = open(os.path.join(categoryPath, filename)).read()
        wordProcessed = fileWordProcess(contents.decode('utf-8'))  # 内容分词成列表
        # 暂时不做#filenameWordProcessed = fileWordProcess(filename) # 文件名分词，单独做特征
        #         words_list.append((wordProcessed,categoryName,filename)) # 训练集格式：[(当前文件内词列表，类别，文件名)]
        words_list.append(wordProcessed)
        filename_list.append(filename)
        category_list.append(categoryName)
        endtime = time.clock();
        print '类别:%s >>>>文件:%s >>>>导入用时: %.3f' % (categoryName, filename, endtime - starttime)

freWord=CountVectorizer(stop_words=None)
transformer=TfidfTransformer()
fre_matrix=freWord.fit_transform(words_list)
tfidf=transformer.fit_transform(fre_matrix)
feature_names=freWord.get_feature_names()               #特征名
freWordVector_df=pd.DataFrame(fre_matrix.toarray())     #全词库，词频，向量矩阵
tfidf_df=pd.DataFrame(tfidf.toarray())                  #tfidf值矩阵
tfidf_df.shape
tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:10000].index
print len(tfidf_sx_featuresindex)
freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_featuresindex] # tfidf法筛选后的词向量矩阵
df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]
print df_columns.shape

tfidf_df_1 = freWord_tfsx_df.apply(guiyi)
tfidf_df_1.columns = df_columns
le = preprocessing.LabelEncoder()
tfidf_df_1['label'] = le.fit_transform(category_list)
tfidf_df_1.index = filename_list

ch2=SelectKBest(chi2,k=7000)
nolabel_feature=[x for x in tfidf_df_1.columns if x not in ['label']]
ch2_sx_np=ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])
label_np=np.array(tfidf_df_1['label'])

X=ch2_sx_np
y=label_np
skf=StratifiedKFold(y,n_folds=10)
y_pre=y.copy()
for train_index,test_index in skf:
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    clf=MultinomialNB().fit(X_train,y_train)
    y_pre[test_index]=clf.predict(X_test)
print '准确率为 %.6f' %(np.mean(y_pre==y))

print "精确率，召回率，F1值："
print classification_report(y,y_pre)
def plot_confusion_matrix(cm,title='matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(category[1:]))
    category_mainname=['En','new_people']
    plt.xticks(tick_marks,category_mainname,rotation=45)
    plt.yticks(tick_marks,category_mainname)
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')
print '矩阵输出'

cm=confusion_matrix(y,y_pre)
plt.figure()
plot_confusion_matrix(cm)

plt.show()
