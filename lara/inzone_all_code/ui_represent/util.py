# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
import random


def select_field(filename,length,to_filename): 
    reader=pd.read_csv(filename,iterator=True,encoding='ISO-8859-1',chunksize=length,
                       usecols=['cdlmkt','cdlmid','CDLDATE','CDLGDID','CDLCATID','CDLPPCODE','CDLCJJE','CDLSL']) #分块读取 
    i=0
    for chunk in reader:
        chunk.rename(columns={'cdlmid':'userid', 'CDLDATE':'time', 'CDLGDID':'itemid','CDLCATID':'classid', 'CDLPPCODE':'brand','CDLCJJE':'price','CDLSL':'num'}, inplace = True)
        chunk=chunk[(chunk['cdlmkt']>6000)&(chunk['brand']>0)&(chunk['cdlmkt']<6030)]
        print(i)
        i=i+1
        name= to_filename+str(i)+'.csv'
        chunk.to_csv(name,index=0,columns=['userid', 'time', 'itemid','classid', 'brand','num','price'])


def get_rawdata():
    df1 = pd.read_csv('march/marchdata1.csv')
    df2 = pd.read_csv('march/marchdata2.csv')
    df3 = pd.read_csv('march/marchdata3.csv')
    df4 = pd.read_csv('march/marchdata4.csv')
    
    df5 = pd.read_csv('April/Aprildata1.csv')
    df6 = pd.read_csv('April/Aprildata2.csv')
    df7 = pd.read_csv('April/Aprildata3.csv')
    df8 = pd.read_csv('April/Aprildata4.csv')
    
    df9 = pd.read_csv('May/Maydata1.csv')
    df10 = pd.read_csv('May/Maydata2.csv')
    df11 = pd.read_csv('May/Maydata3.csv')
    df12 = pd.read_csv('May/Maydata4.csv')
    data = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12], axis = 0) 
    return data

        
def get_users():
    '''
    4626  users
    '''   
    data = get_rawdata()
    a=data.groupby(['userid']).size().reset_index()
    a=pd.DataFrame(a)
    b=a.sort_values(by=[0], ascending=False)    
    b=b[b[0]>200]
    users=b['userid']
    print('the user num is', len(users))
    return pd.DataFrame(users)

def get_items():
    '''5321 items'''
    data = get_rawdata()
    c = data.groupby(['itemid']).size().reset_index()
    c = pd.DataFrame(c)
    d = c.sort_values(by=[0], ascending = False)
    d = d[d[0]>700]
    items = d['itemid']
    print('the item num is', len(items))
    return pd.DataFrame(items)


    

def renumber_users_items():
    
    '''140323 lines that means 140323 pairs
    '''
    users = get_users()
    items = get_items()
    data = get_rawdata()
    data = pd.DataFrame(data)
#    remove repeat
    data = pd.merge(data, users, how='inner', on=['userid'])
    data = pd.merge(data, items, how='inner', on=['itemid'])  
    data = data.drop_duplicates(['userid', 'itemid'])
    
    
    
#    renumber  users
    users=users.reset_index(drop=True)  
    users=users.reset_index()
    data = pd.merge(data, users, how='inner', on=['userid'])   
    data.rename(columns={'index':'re_userid'},inplace = True)
    
    
#   renumber items
    items=items.reset_index(drop=True)  
    items=items.reset_index()
    print(items)
    data = pd.merge(data, items, how='inner', on=['itemid'])    
    data.rename(columns={'index':'re_itemid'},inplace = True)    
    data.to_csv( 'data/data.csv',columns=[ 're_userid', 'time', 're_itemid','classid', 'brand','num','price'],index=0,)    


def get_brands():
    '''669 brands'''
    data = pd.read_csv('data/data.csv')
    e = data.groupby(['brand']).size().reset_index()
    e = pd.DataFrame(e)
    brands = e['brand']
    print('the num of brands is',len(brands))
    return pd.DataFrame(brands)
    
def get_classids():
    '''347 classes'''
    data = pd.read_csv('data/data.csv')
    e = data.groupby(['classid']).size().reset_index()
    e = pd.DataFrame(e)
    classids = e['classid']
    print('the num of classids is',len(classids))
    return pd.DataFrame(classids)

def renumber_brands_classids():

    print('2')
    brands = get_brands()
    classids = get_classids()
    data = pd.read_csv('data/data.csv') 
    print('1')
    #  renumber brand   
    brands = brands.reset_index(drop=True)  
    brands = brands.reset_index()
    data = pd.merge(data, brands, how='inner', on=['brand'])   
    data.rename(columns={'index':'re_brand'},inplace = True)
    
     #  renumber classid   
    classids = classids.reset_index(drop=True)  
    classids = classids.reset_index()
    data = pd.merge(data, classids, how='inner', on=['classid'])   
    data.rename(columns={'index':'re_classid'},inplace = True)
    
    data.to_csv('data/data.csv', columns = ['re_userid', 'time', 're_itemid','re_classid', 're_brand','num','price'],index=0)



def get_ui_matrix():
    data = pd.read_csv('data/data.csv')
    u = len(get_users())
    i = len(get_items())
    ui_matrix = np.zeros((u, i), dtype = np.int32)

    
    pair = pd.DataFrame(data, columns = ['re_userid','re_itemid'])
    
    for j in pair.index:
#        print(pair.loc[j].values[0])
#        print(pair.loc[j].values[1])
        ui_matrix[pair.loc[j].values[0], pair.loc[j].values[1]] = 1
    print( ui_matrix[0])
    np.save("ui_matrix.npy", ui_matrix)
#    np.savetxt("ui_matrix.txt", ui_matrix)    
    
    
def spilt_train_test():
    data=pd.read_csv('data/data.csv')
    data.rename(columns={'re_itemid':'itemid','re_userid':'userid'},inplace = True)
    
    test_id = random.sample(range(2062),400)
    train_id = set(list(range(2062)))-set(test_id)
    
    test_id = pd.DataFrame(test_id)    
    test_id.rename(columns={0:'itemid'},inplace = True)   
    test_data = pd.merge(data,test_id, how='inner', on=['itemid'])      
    
    train_id = pd.DataFrame(list(train_id))    
    train_id.rename(columns={0:'itemid'},inplace = True) 
    train_data = pd.merge(data,train_id, how='inner', on=['itemid'])
    print('the num of train samples is', len(train_data))      
    
    test_data.to_csv('data/test_data.csv',index=0)
    train_data.to_csv('data/train_data.csv',index=0)
    
    
#select_field('raw_data/May.csv', 5000000, 'May/Maydata')
renumber_users_items()
renumber_brands_classids()
get_ui_matrix()
spilt_train_test()
        








