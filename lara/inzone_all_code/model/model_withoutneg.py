# -*- coding: utf-8 -*-
import tensorflow as tf
import support
import numpy as np
import pandas as pd
 
brand_num = 254 
class_num =  178
user_emb_dim = brand_num + class_num

D_brand_emb_dim = 128
D_class_emb_dim = 128

G_brand_emb_dim = 128
G_class_emb_dim = 128

hidden_dim = 128
alpha = 0


'''D variables'''
D_brand_embs = tf.get_variable('D_brand_embs', [brand_num, D_brand_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
D_class_embs = tf.get_variable('D_class_embs', [class_num, D_class_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
# D layer_1
D_l1_input_size = user_emb_dim + D_brand_emb_dim + D_class_emb_dim
D_W1 = tf.get_variable('D_W1', [D_l1_input_size, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.get_variable('D_b1', [1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
D_W2 = tf.get_variable('D_W2', [hidden_dim, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.get_variable('D_b2', [1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())

D_W3 = tf.get_variable('D_W3', [hidden_dim, 1],initializer=tf.contrib.layers.xavier_initializer())
D_b3 = tf.get_variable('D_b3', [1, 1],initializer=tf.contrib.layers.xavier_initializer())

D_params = [D_brand_embs, D_class_embs, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

'''G variables'''
G_brand_embs = tf.get_variable('G_brand_embs', [brand_num, G_brand_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
G_class_embs = tf.get_variable('G_class_embs', [class_num, G_class_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
# D layer_1
G_l1_input_size =  G_brand_emb_dim + G_class_emb_dim
G_W1 = tf.get_variable('G_W1', [G_l1_input_size, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.get_variable('G_b1', [1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())


G_W2 = tf.get_variable('G_W2', [hidden_dim, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.get_variable('G_b2', [1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())

G_W3 = tf.get_variable('G_W3', [hidden_dim, user_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
G_b3 = tf.get_variable('G_b3', [1, user_emb_dim],initializer=tf.contrib.layers.xavier_initializer())

G_params = [G_brand_embs, G_class_embs, G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

'''placeholder'''
brand_id = tf.placeholder(tf.int32)
class_id = tf.placeholder(tf.int32)
real_user_emb = tf.placeholder(shape = [None, user_emb_dim], dtype = tf.float32)
counter_brand_id = tf.placeholder(tf.int32)
counter_class_id = tf.placeholder(tf.int32)
counter_user_emb = tf.placeholder(shape = [None, user_emb_dim], dtype = tf.float32)

'''G'''
def generator(brand_id, class_id):
    brand_emb = tf.nn.embedding_lookup(G_brand_embs, brand_id)
    class_emb = tf.nn.embedding_lookup(G_class_embs, class_id)
    brand_class_emb = tf.concat([class_emb, brand_emb], 1)
    l1_outputs = tf.nn.sigmoid(tf.matmul(brand_class_emb, G_W1) + G_b1)
    l2_outputs = tf.nn.sigmoid(tf.matmul(l1_outputs, G_W2) + G_b2)
    l3_outputs = tf.nn.sigmoid(tf.matmul(l2_outputs, G_W3) + G_b3)
   
    return l3_outputs

'''D'''
def discriminator(brand_id, class_id, user_emb):
    brand_emb = tf.nn.embedding_lookup(D_brand_embs, brand_id)
    class_emb = tf.nn.embedding_lookup(D_class_embs, class_id)
    emb = tf.concat([class_emb, brand_emb, user_emb], 1)
    l1_outputs = tf.nn.sigmoid(tf.matmul(emb, D_W1) + D_b1)
    l2_outputs = tf.nn.sigmoid(tf.matmul(l1_outputs, D_W2) + D_b2)
    D_logit = tf.matmul(l2_outputs, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

'''loss'''
fake_user_emb = generator(brand_id, class_id)
D_real, D_logit_real = discriminator(brand_id, class_id, real_user_emb)
D_fake, D_logit_fake = discriminator(brand_id, class_id, fake_user_emb)
D_counter, D_logit_counter = discriminator(counter_brand_id, counter_class_id, counter_user_emb)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss_counter = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_counter, labels=tf.zeros_like(D_logit_counter))) 

D_regular = alpha * (tf.nn.l2_loss(D_brand_embs) + tf.nn.l2_loss(D_class_embs) + tf.nn.l2_loss(D_W1) + tf.nn.l2_loss(D_b1) + tf.nn.l2_loss(D_W2) + tf.nn.l2_loss(D_b2)) 
G_regular = alpha * (tf.nn.l2_loss(G_brand_embs) + tf.nn.l2_loss(G_class_embs) + tf.nn.l2_loss(G_W1) + 
                     tf.nn.l2_loss(G_b1) + tf.nn.l2_loss(G_W2) + tf.nn.l2_loss(G_b2) + tf.nn.l2_loss(G_W2) + tf.nn.l2_loss(G_b2))

D_loss = D_loss_real + D_loss_fake + D_regular 
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))) + G_regular


'''optimizer'''
D_solver = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(D_loss, var_list=D_params)
G_solver = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(G_loss, var_list=G_params)
#D_counter_solver = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(D_loss_counter, var_list=D_params)

tf_config = tf.ConfigProto()  
tf_config.gpu_options.allow_growth = True  
sess = tf.Session(config=tf_config) 
#sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size = 1024


max_p_at_10 = 0
max_p_at_20 = 0
max_M_at_10 = 0
max_M_at_20 = 0 
max_G_at_10 = 0
max_G_at_20 = 0
p_10_to_save = []
p_20_to_save = []
M_10_to_save = []
M_20_to_save = []
G_10_to_save = []
G_20_to_save = []

'''train'''
for it in range(505):
    D_range = 4
    G_range = 1
           
    for D_it in range(D_range):
        index = 0
        index_2 = 0
        while index < 198488:
             if index + batch_size <= 198488:
                 train_item_batch, train_brand_batch, train_class_batch, train_user_emb_batch = support.get_batchdata(index, index + batch_size)
                 index = index + batch_size
             else:
                 train_item_batch, train_brand_batch, train_class_batch, train_user_emb_batch = support.get_batchdata(index, 198488)
                 index = 198488
            
             _, D_loss_now = sess.run([D_solver, D_loss], 
                                      feed_dict={brand_id:train_brand_batch, class_id:train_class_batch, real_user_emb:train_user_emb_batch,
                                                })  
  
    for G_it in range(G_range):
        index = 0
        while index < 198488:
             if index + batch_size <= 198488:
                 train_item_batch, train_brand_batch, train_class_batch, train_user_emb_batch = support.get_batchdata(index, index + batch_size)
                 index = index + batch_size 
             else:
                 train_item_batch, train_brand_batch, train_class_batch, train_user_emb_batch = support.get_batchdata(index, 198488)
                 index = 198488
             _, G_loss_now = sess.run([G_solver, G_loss], feed_dict={brand_id:train_brand_batch, class_id:train_class_batch})
        
    if it % 1 == 0:
        test_item_batch, test_brand_batch, test_classid_batch = support.get_testdata()
        test_G_user = sess.run(fake_user_emb, feed_dict={brand_id:test_brand_batch, class_id:test_classid_batch})
        
        p_at_10,p_at_20,M_at_10,M_at_20,G_at_10,G_at_20 = support.test(test_item_batch, test_G_user)
        if p_at_10 > max_p_at_10:
            max_p_at_10 = p_at_10
        p_10_to_save.append(p_at_10)
        if p_at_20 > max_p_at_20:
            max_p_at_20 = p_at_20
        p_20_to_save.append(p_at_20)
        if M_at_10 > max_M_at_10:
            max_M_at_10 = M_at_10
        M_10_to_save.append(M_at_10)
        if M_at_20 > max_M_at_20:
            max_M_at_20 = M_at_20
        M_20_to_save.append(M_at_20)
        if G_at_10 > max_G_at_10:
            max_G_at_10 = G_at_10
        G_10_to_save.append(G_at_10)
        if G_at_20 > max_G_at_20:
            max_G_at_20 = G_at_20
        G_20_to_save.append(G_at_20)
      
        print('it', it, 'p_at_10 ', p_at_10, 'p_at_20', p_at_20,'M_at_10',M_at_10,'M_at_20',M_at_20,'G_at_10',G_at_10,'G_at_20',G_at_20)
        print('max p_at_10 ', max_p_at_10, 'p_at_20', max_p_at_20,'M_at_10',max_M_at_10,'M_at_20',max_M_at_20,'G_at_10',max_G_at_10,'G_at_20',max_G_at_20)
    if it % 100 == 0:
        pd.DataFrame(p_10_to_save).to_csv('p10.csv')
        pd.DataFrame(p_20_to_save).to_csv('p20.csv')
        pd.DataFrame(M_10_to_save).to_csv('m10.csv')
        pd.DataFrame(M_20_to_save).to_csv('m20.csv')
        pd.DataFrame(G_10_to_save).to_csv('g10.csv')
        pd.DataFrame(G_20_to_save).to_csv('g20.csv')
    
        
