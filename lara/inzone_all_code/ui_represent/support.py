# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#train_data = np.array(pd.read_csv('data/train_data.csv', usecols = ['userid','itemid','re_classid','re_brand']))
## load the test data
#test_data_csv = pd.read_csv('data/test_data.csv', usecols = ['itemid', 're_classid', 're_brand'])
## drop duplicates
#test_data = np.array(test_data_csv.drop_duplicates(subset='itemid', keep='first', inplace=False))

train_data = np.load("np_train_data.npy")
np.random.shuffle(train_data)
#print(train_data.shape)
counter_data = np.load("np_train_counter_examples.npy")
counter_size = counter_data.shape[0]
np.random.shuffle(counter_data)

test_data = np.load("np_test_data.npy")

test_data = test_data[:, 1:4]

test_data = np.unique(test_data, axis=0)

ui_matrix = np.load("ui_matrix.npy")
ucb_matrix = np.load("ucb_matrix.npy")
#user_emb_matrix = np.load("cf_user_emb.npy")



def get_popular_item(top_k):
     sale_volume = np.sum(ui_matrix, axis = 0)
     popular_item_list = np.argsort(-sale_volume)[0: top_k]
     sub_matrix = ui_matrix[:, popular_item_list]
     return sub_matrix
 
#user_emb_matrix = get_popular_item(1047)
user_emb_matrix = ui_matrix


#sum_user_emb = np.sum(user_emb_matrix, axis=0)
#user_num = np.size(user_emb_matrix, 0)
#mean_user_emb = np.true_divide(sum_user_emb, user_num)
#print(mean_user_emb)


def get_batchdata(start_index, end_index): 
    '''get train samples'''
    batch_data = train_data[start_index: end_index]
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    class_batch = [x[2]for x in batch_data]
    brand_batch = [x[3] for x in batch_data]
    user_emb_batch = user_emb_matrix[user_batch]
    return item_batch, brand_batch, class_batch, user_emb_batch

def get_counter_batch(start_index, end_index):
    '''get counter examples'''
    start_index = start_index % counter_size
    end_index = end_index % counter_size
    counter_batch_data = counter_data[start_index: end_index]
    counter_user_batch = counter_batch_data[:, 0]
    counter_class_batch = counter_batch_data[:, 1]
    counter_brand_batch = counter_batch_data[:, 2]
    counter_user_emb_batch = user_emb_matrix[counter_user_batch]
    return counter_brand_batch, counter_class_batch, counter_user_emb_batch
    
def get_testdata():
    '''get test samples'''
    test_item_batch = test_data[:, 0]
    test_classid_batch = test_data[:, 1]
    test_brand_batch = test_data[:, 2]
    return test_item_batch, test_brand_batch, test_classid_batch


user_sqrt = np.sqrt(np.sum(np.multiply(user_emb_matrix, user_emb_matrix), axis=1))
def get_intersection_similar_user(G_user, k):
    user_emb_matrixT = np.transpose(user_emb_matrix)
    A = np.matmul(G_user, user_emb_matrixT)
    
#    A = np.divide(A, user_sqrt)
    
    intersection_rank_matrix = np.argsort(-A)
#    print( intersection_rank_matrix[0,:k])
#    print( intersection_rank_matrix[50,:k])
    
    return intersection_rank_matrix[:, 0:k]


def test(test_item_batch, test_G_user):
    k_value = 10
    test_BATCH_SIZE = np.size(test_item_batch)
    test_intersection_similar_user = get_intersection_similar_user(test_G_user, k_value)
#    print(test_intersection_similar_user[0:5])

    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        for test_u in test_userlist:
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1
            
    p_at_k = round(count/(test_BATCH_SIZE * k_value), 4)
    

    '''ndcg'''
    para_10 = 0.15
    para_20 = 0.0985
    NDCG = 0
    for test_i, userlist in zip(test_item_batch, test_intersection_similar_user): 
        ndcg = 0.0
        for j in range(k_value):
            test_u = userlist[j]
            if ui_matrix[test_u, test_i] == 1:
                dcg = 1/np.log(j+2)
                ndcg += dcg
        ndcg = ndcg*para_10  
        NDCG += ndcg
    NDCG_at_k = round(NDCG/test_BATCH_SIZE, 4)
#    
#    '''map'''
    kk = k_value 
    ap = 0
    for test_i, userlist in zip(test_item_batch, test_intersection_similar_user): 
        count = 0
        rel_user_num = sum(ui_matrix[:, test_i])
        for j in range(kk):
            test_u = userlist[j]
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1
                ap = ap + count/((j+1)*rel_user_num)
    mAP = round(ap/test_BATCH_SIZE, 4)
#    
    return p_at_k, NDCG_at_k, mAP

    
#    test_proportion_similar_user = get_proportion_similar_user(test_G_user, k_value)
#    
#    count = 0
#    for test_i, test_userlist in zip(test_item_batch, test_proportion_similar_user):
#        for test_u in test_userlist:
#            if ui_matrix[test_u, test_i] == 1:
#                count = count + 1
#    
#    p_at_k = round(count/(test_BATCH_SIZE * k_value), 4)
#    print('p@k prop is', p_at_k)
#test_item_batch, _, _ = get_testdata()
#for test_i in test_item_batch:
#    print('there are', np.sum(ui_matrix[:, test_i]), 'users buying item', test_i)