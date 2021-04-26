# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random

reader = pd.read_csv('raw_data/March.csv', usecols = ['cdlmkt', 'cdlmid', 'CDLGDID','CDLCATID', 'CDLPPCODE'], chunksize = 100000, iterator = True)
#reader = pd.read_csv('raw_data/March.csv', nrows = 100000, usecols = ['cdlmkt', 'cdlmid', 'CDLGDID','CDLCATID', 'CDLPPCODE'], chunksize = 10000, iterator = True)
d = pd.DataFrame(columns = ['userid', 'itemid', 'classid', 'brandid'])
for chunk in reader:
    chunk.rename(columns = {'cdlmid':'userid', 'CDLGDID':'itemid', 'CDLCATID':'classid', 'CDLPPCODE':'brandid'}, inplace=True)
    chunk = chunk[(chunk['cdlmkt']>6000)&(chunk['brandid']>0)&(chunk['cdlmkt']<6010)]
    chunk = chunk[['userid', 'itemid', 'classid', 'brandid']]
    d = pd.concat([d, chunk], axis = 0, ignore_index = True)
#    print(chunk)
print('***********************')    
#print(d)
d.drop_duplicates(['userid', 'itemid'], inplace=True)
#print(d.columns)
#print(d.sort_values(by='userid'))
dg = d.groupby('userid').count()
#dg = pd.DataFrame(dg)
#print(dg)
c = dg[dg['itemid']>=100]
#print(c)
can_uid = np.array(c.index)
#print(can_uid)
#print(dg['itemid'])
USER_NUM = can_uid.size
a = pd.DataFrame({'userid':can_uid, 'new_userid':np.arange(can_uid.size)})
#print(a)
#print(d.dtypes)
#print(a.dtypes)
d = pd.DataFrame(d, dtype='int')
d = pd.merge(d, a, on='userid')
#print(d)
d = d[['new_userid', 'itemid', 'classid', 'brandid']]
#print(d)

'''renumber the itemid'''

b = d['itemid']
#print(b)
b.drop_duplicates(keep='first', inplace=True)
#print(b)
e = np.array(b.sort_values())
ITEM_NUM = e.size
#print(e)
f = pd.DataFrame({'itemid':e, 'new_itemid':np.arange(e.size)})
#print(f)
#print(d)
#print(d)
d = pd.DataFrame(d)
d = pd.merge(d, f, on='itemid')
#print(d)
d = d[['new_userid', 'new_itemid', 'classid', 'brandid']]
#print(d)

'''renumber the classid'''
g = d['classid']
g.drop_duplicates(keep='first', inplace=True)
g = g.sort_values()
#print(g)
CLASS_NUM = g.size
h = pd.DataFrame({'classid':g, 'new_classid':np.arange(g.size)})
#print(h)
#print(d)
d = pd.DataFrame(d)
d = pd.merge(d, h, on='classid')
#print(d)
d = d[['new_userid', 'new_itemid', 'new_classid', 'brandid']]
#print(d)

'''renumber the brandid'''
i = d['brandid']
i.drop_duplicates(keep='first', inplace=True)
i = i.sort_values()
#print(i)
BRAND_NUM = i.size
j = pd.DataFrame({'brandid':i, 'new_brandid':np.arange(i.size)})
#print(j)
d = pd.DataFrame(d)
d = pd.merge(d, j, on='brandid')
#print(d)
d = d[['new_userid', 'new_itemid', 'new_classid', 'new_brandid']]
#print(d)
d.rename(columns={'new_userid':'uid', 'new_itemid':'iid', 'new_classid':'cid', 'new_brandid':'bid'}, inplace=True)
#print(d)


'''ui-matrix'''
print(USER_NUM, ITEM_NUM)
ui_matrix = np.zeros((USER_NUM, ITEM_NUM), dtype = np.int32)
pair = pd.DataFrame(d, columns = ['uid','iid'])
    
for q in pair.index:
    ui_matrix[pair.loc[q].values[0], pair.loc[q].values[1]] = 1
#print(ui_matrix)
np.save("ui_matrix.npy", ui_matrix)

'''train data and test data'''
test_iid = random.sample(range(ITEM_NUM), int(ITEM_NUM/4))
train_iid = list(set(list(range(ITEM_NUM)))-set(test_iid))
#print(test_iid)
#print(train_iid)
test_iid_df = pd.DataFrame({'iid':test_iid})
train_iid_df = pd.DataFrame({'iid':train_iid})
#print(test_iid_df)
#print(train_iid_df)
test_data = pd.merge(d, test_iid_df, on='iid')
train_data = pd.merge(d, train_iid_df, on='iid')
#print(test_data)
#print(train_data)
test_data.to_csv('data/test_data.csv',index=0)
train_data.to_csv('data/train_data.csv',index=0)
print(USER_NUM, ITEM_NUM, CLASS_NUM, BRAND_NUM)

