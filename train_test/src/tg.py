## -*- coding: utf-8 -*-

import os
import random
from tgrocery import Grocery

grocery=Grocery('sample')
path_en='../After/En'
path_peo='../After/new_people'
train_src=[]
test_src=[]
test_src_g=[]


#随机选取列表长度的二分之一个数据量
def list_cut(list):
    list_len=len(list)
    slist=random.sample(list,list_len/2)
    return slist
#两个完全不相同的列表相加，或者包含关系的列表相减
def list_sub(list1,list2):
    slist=[]
    for a in list1:
        if a not in list2:
            slist.append(a)
    return slist



enlist=os.listdir(path_en)
enlist_train=list_cut(enlist)
enlist_test=list_sub(enlist,enlist_train)

peolist=os.listdir(path_peo)
peolist_train=list_cut(peolist)
peolist_test=list_sub(peolist,peolist_train)


# 为训练集添加所属类别，以及标题名称
for path_e in enlist_train:
    train_src.append(("en",path_e))

for path_p in peolist_train:
    train_src.append(("peo",path_p))
#将剩余的文件进行归类整理,我们进行测试的数据分别从政事儿和财新网所以分别归类为经济类和政治类，test_src存有训练机，而test_str_g是为了计算其准确率
for text in peolist_test:
    test_src.append(text)
    test_src_g.append(("peo",text))
for text in enlist_test:
    test_src.append(text)
    test_src_g.append(("en",text))
grocery.train(train_src)
#保存模型
grocery.save()
#加载模型
new_grocery=Grocery('sample')
new_grocery.load()
#逐个预测
for textsrc in test_src:
    print str(new_grocery.predict(textsrc))+"   "+textsrc
#测试准确率
print new_grocery.test(test_src_g)
