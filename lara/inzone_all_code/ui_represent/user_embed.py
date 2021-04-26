# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import  numpy as np 

def pca ():
    ui_matrix = np.loadtxt("ui_matrix.txt",dtype=np.int32)  
    pca = PCA(n_components=2000)
    pca.fit(ui_matrix)
    X_new = pca.transform(ui_matrix)
    np.savetxt("PCA_ui_2000.txt", X_new)


def get_popular_item(top_k):
     ui_matrix = np.load("ui_matrix.npy")  
     sale_volume = np.sum(ui_matrix, axis = 0)
     popular_item_list = np.argsort(-sale_volume)[0: top_k]
     sub_matrix = ui_matrix[:, popular_item_list]
     return popular_item_list, sub_matrix
 
    
def get_hamming_similar_user(G_user_emb, user_emb_matrix, user_num, k):
    hamming_d_array = np.zeros(user_num)
    for user_id, user_emb in zip(range(user_num), user_emb_matrix):
        G_user_emb = np.round(G_user_emb)
        hamming_d = np.size(np.nonzero(user_emb - G_user_emb))
        hamming_d_array[user_id] = hamming_d
    hamming_d_rank = np.argsort(hamming_d_array)
    return hamming_d_rank[0: k]

def get_euclidean_similar_user(G_user_emb, user_emb_matrix, k):
    A = user_emb_matrix - G_user_emb
    #AT = np.transpose(A)
    euclidean = np.sqrt(np.sum(A * A, axis = 1))
    euclidean_d_rank = np.argsort(euclidean)
    return euclidean_d_rank[0: k]

# G_user is the generated users
def get_intersection_similar_user(G_user, user_emb_matrix, k):
    #G_user = np.round(G_user)
    user_emb_matrixT = np.transpose(user_emb_matrix)
    A = np.matmul(G_user, user_emb_matrixT)
    intersection_rank_matrix = np.argsort(-A)
    return intersection_rank_matrix[:, 0:k]
    
    
    


    
    
#G_user_emb = np.array([0.4, 0.1, 0.6])
#user_emb_matrix = np.array([[1,1,0],
#                            [0,0,0],
#                            [0,1,0],
#                            [0,0,1]])
#print(user_emb_matrix - G_user_emb)
#user_num = 4
#print(
#      get_hamming_similar_user(G_user_emb, user_emb_matrix, user_num, 3)
#      )