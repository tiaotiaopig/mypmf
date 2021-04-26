# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def remove_test_samples():
    train_data = np.array(pd.read_csv('data/train_data.csv', usecols = ['uid','iid','cid','bid']))
    test_data = np.array(pd.read_csv('data/test_data.csv', usecols = ['uid','iid','cid','bid']))
    
    A = train_data[:, 1:4]
    AA = np.unique(A, axis=0)
    print(AA.shape)

    train_brands = train_data[:, 3]
    test_brands = test_data[:,3]
    brand_only_in_test = np.setdiff1d(test_brands, train_brands)
    train_classes = train_data[:, 2]
    test_classes = test_data[:, 2]
    class_only_in_test = np.setdiff1d(test_classes, train_classes)
    train_users = train_data[:, 0]
    test_users = test_data[:, 0]
    user_only_in_test = np.setdiff1d(test_users, train_users)
    
    print(brand_only_in_test)
    print(class_only_in_test)
    print(user_only_in_test)
    print(user_only_in_test.shape)

    '''remove the brands'''
    index = np.in1d(test_brands, brand_only_in_test)
    not_index = np.logical_not(index)
    move_brand_samples = test_data[index]
    remain_test_data = test_data[not_index]

    print('the num of original train samples', np.size(train_brands, 0))
    print('the num of original test samples', np.size(test_brands, 0))
    print('the num of the moved test samples', np.size(move_brand_samples, 0))
    #print(move_brand_samples)

    train_data = np.concatenate((train_data, move_brand_samples), axis=0)
    print('the number of train samples after moving brands', np.size(train_data, 0))
    print('the number of test samples after moving brands', np.size(remain_test_data, 0))
    
    A = train_data[:, 1:4]
    AA = np.unique(A, axis=0)
    print('the item in train data is', AA.shape)
    B = remain_test_data[:, 1:4]
    BB = np.unique(B, axis=0)
    print('the item in test data is', BB.shape)

    '''remove the classes'''
    remain_test_classes = remain_test_data[:, 2]

    index = np.in1d(remain_test_classes, class_only_in_test)
    not_index = np.logical_not(index)
    move_class_samples = remain_test_data[index]
    remain_test_data = remain_test_data[not_index]

    print('the num of the moved test samples', np.size(move_class_samples, 0))

    train_data = np.concatenate((train_data, move_class_samples), axis=0)
    print('the number of train samples after moving classes', np.size(train_data, 0))
    print('the number of test samples after moving classes', np.size(remain_test_data, 0))
    A = train_data[:, 1:4]
    AA = np.unique(A, axis=0)
    print('the item in train data is', AA.shape)
    B = remain_test_data[:, 1:4]
    BB = np.unique(B, axis=0)
    print('the item in test data is', BB.shape)
    
#    '''remove the users'''
#    remain_test_users = remain_test_data[:, 0]
#    
#    index = np.in1d(remain_test_users, user_only_in_test)
#    not_index = np.logical_not(index)
##    E = remain_test_data[:, 0:2][index]
##    E = E[E[:,1].argsort()]
##    print(E)
##    H, h = np.unique(E[:,0], return_index=True)
##    print(H)
##    print(np.unique(E[h][:,1]).shape)
#   
#    move_user_samples = remain_test_data[index]
#    remain_test_data = remain_test_data[not_index]
#    
#    print('the num of the moved user samples', np.size(move_user_samples, 0))
#    
#    train_data = np.concatenate((train_data, move_user_samples), axis=0)
#    print('the number of train samples after moving users', np.size(train_data, 0))
#    print('the number of test samples after moving users', np.size(remain_test_data, 0))
#    A = train_data[:, 1:4]
#    AA = np.unique(A, axis=0)
#    print('the item in train data is', AA.shape)
#    B = remain_test_data[:, 1:4]
#    BB = np.unique(B, axis=0)
#    print('the item in test data is', BB.shape)
    
    USER_NUM = 54765
    CLASS_NUM = 178
    BRAND_NUM = 254
    
    uc_matrix = np.zeros((USER_NUM, CLASS_NUM), dtype = np.int)    
    pair = train_data[:,[0,2]]
    M = np.unique(pair, axis=0)
    print(M)

    for m in M:
        uc_matrix[m[0], m[1]] = 1
    print(uc_matrix[1])
    
    ub_matrix = np.zeros((USER_NUM, BRAND_NUM), dtype = np.int)    
    pair = train_data[:,[0,3]]
    M = np.unique(pair, axis=0)
    print(M)

    for m in M:
        ub_matrix[m[0], m[1]] = 1
    print(ub_matrix[1])

    ucb_matrix = np.concatenate((uc_matrix, ub_matrix), axis=1)
    np.save('ucb_matrix.npy', ucb_matrix)
    
    
    np.save("np_train_data.npy", train_data)
    np.save("np_test_data.npy", remain_test_data)

 


def generate_train_counter_examples():
    train_data = np.load("np_train_data.npy")
    train_users = train_data[:, 0]
    users = np.unique(train_users)
    
    train_brands = train_data[:, 3]
    brands = np.unique(train_brands)
    
    train_classes = train_data[:, 2]
    classes = np.unique(train_classes)
    
    brand_class = np.unique(train_data[:, 2:4], axis=0)
    
    train_counter_examples = np.empty([0, 3], dtype=int)
    
    for b_c in brand_class:
        b = b_c[1]
        b_index = train_brands == b
        user_like_brand = np.unique(train_users[b_index])
        user_dislike_brand = np.setdiff1d(users, user_like_brand)    
        c = b_c[0]
        c_index = train_classes == c
        user_like_class = np.unique(train_users[c_index])
        user_dislike_class = np.setdiff1d(users, user_like_class)
        user_dislike_brand_class = np.intersect1d(user_dislike_brand, user_dislike_class)
#        print(np.size(user_dislike_brand_class))
        DISLIKE_NUM = 1000
        sub_user_dislike_brand_class = np.random.choice(user_dislike_brand_class, DISLIKE_NUM)
        A = np.transpose(sub_user_dislike_brand_class.reshape(1, -1))
        B = np.repeat(b_c.reshape(1, -1), DISLIKE_NUM, axis=0)
#        print(A.ndim)
#        print(B.ndim)
        samples = np.concatenate((A, B), axis=1)
#        print(samples)
        train_counter_examples = np.concatenate((train_counter_examples, samples), axis=0)
#        print(train_counter_examples.size)
    print(train_counter_examples.size)
    np.save("np_train_counter_examples.npy", train_counter_examples)   


remove_test_samples() 
generate_train_counter_examples()
D = np.load("np_train_counter_examples.npy")   
print(D.shape)

#def test_item_sale():
#    ui_matrix = np.load("ui_matrix.npy")
#    
#    test_data = np.load("np_test_data.npy")
#    test_data = test_data[:, 1]
#    test_data = np.unique(test_data, axis=0)
#    for item_id in test_data:
#        sale = np.sum(ui_matrix[:, item_id])
#        print(sale)
#    
#test_item_sale()
    
        
        
