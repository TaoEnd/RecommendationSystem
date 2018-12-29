# coding:utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

# 数据位置：http://files.grouplens.org/datasets/movielens

def get_data(path):
    data_df = pd.read_table(filepath_or_buffer=path, header=None, sep="::")
    data_df.columns = ["user", "movie", "score", "time"]
    rows_num = data_df.shape[0]
    shuffed_index = np.random.permutation(rows_num)
    data_df = data_df.iloc[shuffed_index]
    return data_df

# 1、
    # user_batch表示每个batch中存在多少个不同的user，
    # movie_batch表示每个batch中存在多少部不同的电影，
    # user_num表示用户的总数，movie_num表示电影的总数，
    # dim表示对矩阵进行隐语义分解时，隐变量的个数
# 2、
    # 原始数据中一条表示一个用户对一部电影的打分，
    # 因此每个batch中只包含了部分用户和部分电影，并没有包含
    # 全部的电影
def model(user_batch, movie_batch, user_num, movie_num, dim):
    with tf.device("/cpu:0"):
        # 使用get_variable时：a = tf.get_variable("weight", shape=[2])，
        # a表示在python中的变量名，weight表示在tensorflow中的变量名，为了
        # 使得变量能重用，需要设置reuse=True，
        # 否则再定义b = tf.get_variable("weight", shape=[2])时，会报错
        with tf.variable_scope("LSI", reuse=tf.AUTO_REUSE):
            # global_bias表示整体的评分矩阵所具有的大趋势
            # shape=[]表示是一个0维的张量，它是一个纯量
            global_bias = tf.get_variable("global_bias", shape=[])
            # user_bias、movie_bias分别表示用户和电影的自身特性：
            # 用户的平均打分，电影的平均得分
            user_bias = tf.get_variable("user_bias", shape=[user_num])
            movie_bias = tf.get_variable("movie_bias", shape=[movie_num])
            # user_bias中包含了所有用户的平均打分，但是每个batch中只包含部分用户，
            # 并且在每个batch的模型训练中也只会用这部分用户的信息，其它用户的信息不会
            # 使用，因此可以使用embedding_lookup找出这部分用户的平均打分
            user_bias_sub = tf.nn.embedding_lookup(user_bias, user_batch, name="user_bias_sub")
            movie_bias_sub = tf.nn.embedding_lookup(movie_bias, movie_batch, name="movie_bias_sub")
            # 权重
            user_w = tf.get_variable("user_weight", shape=[user_num, dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.3))
            movie_w = tf.get_variable("movie_weight", shape=[movie_num, dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.2))
            user_w_sub = tf.nn.embedding_lookup(user_w, user_batch, name="user_w_sub")
            movie_w_sub = tf.nn.embedding_lookup(movie_w, movie_batch, name="movie_w_sub")
    with tf.device("/cpu:0"):
        # multiply是两个矩阵对应位置相乘，两个矩阵的维度必须相等，
        # 1是按列进行加和
        infer = tf.reduce_sum(tf.multiply(user_w_sub, movie_w_sub), 1)
        infer = tf.add(infer, global_bias)
        infer = tf.add(infer, user_bias_sub)
        infer = tf.add(infer, movie_bias_sub, name="svd_inference")
        # 正则化项
        regularizer = tf.add(tf.nn.l2_loss(user_w_sub), tf.nn.l2_loss(movie_w_sub), name="svd_regularizer")
    return infer, regularizer

# 定义损失函数
def loss(infer, regularizer, score_batch, learning_rate, alpha):
    with tf.device("/cpu:0"):
        # l2_loss：对于向量X中的任意元素xi，
        # l2_loss等于所有xi的平方和再除上2
        # loss_l2 = tf.nn.l2_loss(tf.subtract(infer, score_batch))
        mean_loss = tf.reduce_mean(tf.reduce_sum(tf.square(infer - score_batch)))
        penalty = tf.constant(alpha, dtype=tf.float32, shape=[], name="alpha")
        loss = tf.add(mean_loss, tf.multiply(regularizer, penalty))
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return loss, train_op

if __name__ == "__main__":
    np.random.seed(0)
    learning_rate = 0.001
    alpha = 0.1
    dim = 5

    path = r"E:\python\PythonSpace\Data\recommendationsystem\movie\ratings_1.dat"
    data_df = get_data(path)

    user_num = len(data_df.user.unique().tolist()) + 1
    movie_num = len(data_df.movie.unique().tolist()) + 1
    user_batch = tf.placeholder(tf.int32, shape=[None], name="user_id")
    movie_batch = tf.placeholder(tf.int32, shape=[None], name="movie_id")
    score_batch = tf.placeholder(tf.float32, shape=[None], name="score")
    infer, regularizer = model(user_batch, movie_batch, user_num, movie_num, dim)
    loss, train_op = loss(infer, regularizer, score_batch, learning_rate, alpha)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_size = 1024
        max_epoch = 50
        all_data_num = data_df.shape[0]
        batch_nums = int(all_data_num / batch_size)
        print("总数据量：%d，batch size：%d, batch nums：%d" % (all_data_num, batch_size, batch_nums))
        print("总用户数：%d，总音乐数：%d" % (user_num, movie_num))
        print("开始训练...")
        for epoch in range(max_epoch):
            print("第%d轮训练" % (epoch))
            for num in range(batch_nums):
                if (num+1) % 100 == 0:
                    print("  batch：", (num+1))
                start_index = batch_size * num
                end_index = min(batch_size*(num+1), all_data_num)
                user = np.array(data_df.user.tolist()[start_index: end_index])
                movie = np.array(data_df.movie.tolist()[start_index: end_index])
                score = np.array(data_df.score.tolist()[start_index: end_index])
                # print(user.tolist())
                # print(movie.tolist())
                # print(score.tolist())
                _, cost = sess.run([train_op, loss], feed_dict={user_batch: user,
                                                           movie_batch: movie,
                                                           score_batch: score})
                print(cost)





