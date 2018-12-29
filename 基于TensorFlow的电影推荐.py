# coding:utf-8

import pandas as pd
import numpy as np
import tensorflow as tf

def get_data(path):
    data_df = pd.read_table(filepath_or_buffer=path, header=None, sep="::")
    data_df.columns = ["user_id", "movie_id", "score", "timestamp"]
    shuffed_index = np.random.permutation(data_df.shape[0])
    data_df = data_df.iloc[shuffed_index]
    return data_df

def create_model(user_batch, movie_batch, user_num, movie_num, dim):
    with tf.name_scope(name="LSI"):
        # 定义变量
        bias_global = tf.Variable(0.0, name="global_bias")
        bias_user = tf.Variable(tf.random_normal([user_num]), name="user_bias")
        bias_movie = tf.Variable(tf.random_normal([movie_num]), name="movie_bias")
        user_hv_matrix = tf.Variable(tf.random_normal([user_num, dim], stddev=0.1), name="uhv")
        hv_movie_matrix = tf.Variable(tf.random_normal([movie_num, dim], stddev=0.1), name="hvm")
        # embedding_lookup(matrix, list)：根据list中的数字在matrix中取出对应的行
        bias_user_sub = tf.nn.embedding_lookup(bias_user, user_batch, name="user_bias_sub")
        bias_movie_sub = tf.nn.embedding_lookup(bias_movie, movie_batch, name="movie_bias_sub")
        uhvm_sub = tf.nn.embedding_lookup(user_hv_matrix, user_batch, name="uhvm_sub")
        hvmm_sub = tf.nn.embedding_lookup(hv_movie_matrix, movie_batch, name="hvmm_sub")

        # 预测结果
        # uhvm_sub矩阵相当于是当前user batch中用户的参数，它的第一行代表第一个用户的参数，
        # hvmm_sub矩阵则是当前movie batch中用户的参数，其第一行代表第一部电影的参数，
        # 而这两个矩阵是存在一一对应关系的，第一个用户评价的是第一部电影，第二个用户评价的
        # 是第二部电影，因此应该用第一个矩阵的第一行乘上第二个矩阵的第一行，然后求和得到
        # 第一个用户对第一部电影的评分，用uhvm_sub的第二行乘上hvmm_sub的第二行求和得到第
        # 二个用户对第二部电影的评分
        pred = tf.reduce_sum(tf.multiply(uhvm_sub, hvmm_sub), axis=1)
        pred = tf.add(pred, bias_global)
        pred = tf.add(pred, bias_user_sub)
        pred = tf.add(pred, bias_movie_sub, name="pred")
        regularizer = tf.add(tf.nn.l2_loss(uhvm_sub), tf.nn.l2_loss(hvmm_sub), name="L2")
    return regularizer, pred

def loss_and_train(score_batch, pred, regularizer, learning_rate, alpha):
    cost = tf.reduce_mean(tf.square(score_batch-pred))
    penalty = tf.constant(alpha, name="alpha")
    loss = tf.add(regularizer, tf.multiply(penalty, cost))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return loss, train_op

if __name__ == "__main__":
    np.random.seed(0)
    batch_size = 1024
    max_epoch = 50
    learning_rate = 0.001
    alpha = 0.2
    dim = 5

    data_path = r"E:\python\PythonSpace\Data\recommendationsystem\movie\ratings_1.dat"
    data_df = get_data(data_path)
    user_num = len(data_df.user_id.unique().tolist()) + 1
    movie_num = len(data_df.movie_id.unique().tolist()) + 1

    user_batch = tf.placeholder(dtype=tf.int32, shape=[None])
    movie_batch = tf.placeholder(dtype=tf.int32, shape=[None])
    score_batch = tf.placeholder(dtype=tf.float32, shape=[None])
    pred, regularizer = create_model(user_batch, movie_batch, user_num, movie_num, dim)
    loss, train_op = loss_and_train(score_batch, pred, regularizer, learning_rate, alpha)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # user_id_list = data_df.user_id.tolist()
        # movie_id_list = data_df.movie_id.tolist()
        # score_list = data_df.score.tolist()
        all_data_num = data_df.shape[0]
        batch_nums = int(all_data_num / batch_size)
        print("总数据量：%d，batch size：%d, batch nums：%d" % (all_data_num, batch_size, batch_nums))
        print("总用户数：%d，总音乐数：%d" % (user_num, movie_num))
        print("开始训练...")
        for epoch in range(max_epoch):
            print("第%d轮训练" % (epoch+1))
            for num in range(batch_nums):
                start_index = batch_size * num
                end_index = min(batch_size*(num+1), all_data_num)
                user = np.array(data_df.user_id.tolist()[start_index: end_index])
                movie = np.array(data_df.movie_id.tolist()[start_index: end_index])
                score = np.array(data_df.score.tolist()[start_index: end_index])
                sess.run([pred, regularizer, loss, train_op], feed_dict={user_batch: user, movie_batch: movie, score_batch: score})
                if num % 10 == 0:
                    cost = sess.run([pred, regularizer, loss, train_op], feed_dict={user_batch: user, movie_batch: movie, score_batch: score})
                    print("  Loss：", cost)





