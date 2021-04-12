# -*- coding: utf-8 -*-
import numpy as np


class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.9, max_epoch=20, num_batches=10,
                 batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.max_epoch = max_epoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    def fit(self, train_vec, test_vec):
        # mean subtraction
        mean_inv = np.mean(train_vec[:, 2])  # 评分平均值
        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数
        # 确定 R 的大小
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数
        # 初始化
        epoch = 0
        self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
        self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵
        # 使用带有动量的SGD,用于更新 w_Item 和 w_User
        w_item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
        w_user_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while epoch < self.max_epoch:  # 检查迭代次数
            epoch += 1
            # 创建 train 中的随机索引
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)
            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标
                # 0-1000 -> shuffle -> userid
                batch_user_id = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_item_id = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')
                # 计算预测结果, axis=1 就是D那一维
                pred_out = np.sum(
                    np.multiply(self.w_User[batch_user_id, :], self.w_Item[batch_item_id, :]),
                    axis=1)
                raw_err = pred_out + mean_inv - train_vec[shuffled_order[batch_idx], 2]
                # 计算梯度
                ix_user = 2 * np.multiply(raw_err[:, np.newaxis], self.w_Item[batch_item_id, :]) \
                          + self._lambda * self.w_User[batch_user_id, :]
                ix_item = 2 * np.multiply(raw_err[:, np.newaxis], self.w_User[batch_user_id, :]) \
                          + self._lambda * (self.w_Item[batch_item_id, :])  # np.newaxis :increase the dimension

                dw_item = np.zeros((num_item, self.num_feat))
                dw_user = np.zeros((num_user, self.num_feat))
                # 这个batch(1000)中,有重复的用户和商品,对于相同元素的梯度要累加
                # 根据前面pred_out 的产生, 第 i 个梯度,属于batch_ItemID[i]
                for i in range(self.batch_size):
                    dw_item[batch_item_id[i], :] += ix_item[i, :]
                    dw_user[batch_user_id[i], :] += ix_user[i, :]
                # 使用动量更新梯度
                w_item_inc = self.momentum * w_item_inc + self.epsilon * dw_item / self.batch_size
                w_user_inc = self.momentum * w_user_inc + self.epsilon * dw_user / self.batch_size
                #
                self.w_Item = self.w_Item - w_item_inc
                self.w_User = self.w_User - w_user_inc
            # 计算训练集上的均方根误差
            raw_err = self.get_raw_err(train_vec, mean_inv)
            # 求2范数,再平方
            obj = np.linalg.norm(raw_err) ** 2 \
                  + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)
            self.rmse_train.append(np.sqrt(obj / pairs_train))
            # 计算测试集上的均方根误差
            raw_err = self.get_raw_err(test_vec, mean_inv)
            self.rmse_test.append(np.linalg.norm(raw_err) / np.sqrt(pairs_test))
            print('Train RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))

    def get_raw_err(self, data_vec, mean_inv):
        pred_out = np.sum(
            np.multiply(self.w_User[np.array(data_vec[:, 0], dtype='int32'), :],
                        self.w_Item[np.array(data_vec[:, 1], dtype='int32'), :]),
            axis=1)
        return pred_out - data_vec[:, 2] + mean_inv
