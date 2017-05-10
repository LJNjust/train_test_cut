import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


freWord=CountVectorizer(stop_words='data/stopWord.txt')
transformer=TfidfTransformer()
fre_matrix=freWord.fit_transform(test.words_list)
tfidf=transformer.fit_transform(fre_matrix)
feature_names=freWord.get_feature_names()
freWordVector_df=pd.DataFrame(fre_matrix.toarray())
tfidf_df=pd.DataFrame(tfidf.toarray())
tfidf_df.shape

tfidf_sx_featuresindex=tfidf_df.sum(axis=0).sort_values(ascending=False)[:10000].index
print  len(tfidf_sx_featuresindex)
freWord_tfsx_df=freWordVector_df.ix[:,tfidf_sx_featuresindex]
df_colums=pd.Series(feature_names)[tfidf_sx_featuresindex]
print df_colums.shape
def guiyi(x):
    x[x>1]=1
    return x

tfidf_df_1=freWord_tfsx_df.apply(guiyi)
tfidf_df_1.columns=df_colums
le=preprocessing.LabelEncoder()
tfidf_df_1['label']=le.fit_transform(test.category_list)
tfidf_df_1.index= test.filename_list


