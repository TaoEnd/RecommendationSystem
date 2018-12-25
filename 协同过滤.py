# coding:utf-8

import numpy as np
import pandas as pd
from pandas import Series

# 使用协同过滤推荐音乐
# 首先找到需要推荐音乐的用户听过的歌曲，
# 然后分别计算其余歌曲与这些歌曲中的每一首的相似度，将相似度平均值最高的top-k
# 的歌曲作为推荐，比如当前用户一共听过20首歌曲，还有100首歌曲他没有听过，首先
# 计算这100首歌曲与这20首歌曲的相似度，从而得到一个20*100的矩阵，然后计算每一
# 列的平均值，表示当前这首歌与用户听过的20首歌的平均相似度，将值最高的top-k的
# 歌曲作为推荐

# 相似度计算方式：对于用户听过的第i首歌曲，与未听过的第j首歌曲，首先找到所有
# 听过第i首歌曲的人，听过第j首歌的人，然后统计这两类人的交集，再统计这两类人
# 的并集，将它们交集与并集的比例作为相似度

class Recommender():
    def __init__(self, train_data, user_id):
        self.train_data = train_data
        self.user_id = user_id

    # 获得需要推荐的用户听过的歌曲
    def get_has_listened_music(self):
        all_listened_musics = self.train_data[self.train_data.user==self.user_id]
        # 如果不加tolist，则返回的是类型是ndarray的
        has_listened_musics = list(all_listened_musics.title.unique().tolist())
        return has_listened_musics

    # 获得推荐用户所有未听过的歌曲
    def get_not_listened_music(self, has_listened_musics):
        has_listened_musics = set(has_listened_musics)
        not_listened_musics = set()
        all_musics = self.train_data.title.tolist()
        for music_name in all_musics:
            if music_name not in has_listened_musics:
                not_listened_musics.add(music_name)
        not_listened_musics = list(not_listened_musics)
        return not_listened_musics

    # 根据歌曲名字，获得听过当前歌曲的用户
    def get_user_by_music_name(self, music_name):
        all_user = self.train_data[self.train_data.title==music_name]
        unique_user = set(all_user.user.unique())
        return unique_user

    # 得到相似矩阵
    def get_similarity_matrix(self, has_listened_musics, not_listened_musics):
        # 先得到没有听过的音乐中每首音乐都被哪些人听过
        not_listened_vs_user = []
        for i in range(len(not_listened_musics)):
            unique_user = self.get_user_by_music_name(not_listened_musics[i])
            not_listened_vs_user.append(unique_user)

        has_listened_num = len(has_listened_musics)
        not_listened_num = len(not_listened_musics)
        similarity_matrix = np.zeros(shape=(has_listened_num, not_listened_num), dtype=float)
        for i in range(has_listened_num):
            music_i = self.get_user_by_music_name(has_listened_musics[i])
            for j in range(not_listened_num):
                music_j = not_listened_vs_user[j]
                user_intersection = music_i.intersection(music_j)
                user_union = music_i.union(music_j)
                # if len(user_intersection) != 0:
                #     print(len(user_intersection), len(user_union))
                similarity_i_j = len(user_intersection)/len(user_union)
                similarity_matrix[i][j] = similarity_i_j
        return similarity_matrix

    # 得到推荐歌曲的排序
    def get_sorted_music(self):
        has_listened_musics = self.get_has_listened_music()
        not_listened_musics = self.get_not_listened_music(has_listened_musics)
        similarity_matrix = self.get_similarity_matrix(has_listened_musics, not_listened_musics)
        similarity_matrix = np.array(similarity_matrix)
        avg_similarity_scores = similarity_matrix.sum(axis=0)/similarity_matrix.shape[0]
        # 对相似度进行逆序排序，返回的一个列表，表中的元素是一个元组，
        # 元组中的第一个元素是相似度，第二个元素是当前相似度在原来的列表中的index
        similarity_scores_tuple = ((s, i) for i, s in enumerate(avg_similarity_scores))
        sorted_tuple = sorted(similarity_scores_tuple, reverse=True)
        columns = ["user_id", "recommendation_music", "similarity_score", "rank"]
        sorted_music_df = pd.DataFrame(columns=columns)
        rank = 1
        for tuple in sorted_tuple:
            recommendation_music = not_listened_musics[tuple[1]]
            similarity_score = tuple[0]
            item = [self.user_id, recommendation_music, similarity_score, rank]
            rank += 1
            s = Series(item, index=columns)
            sorted_music_df = sorted_music_df.append(s, ignore_index=True)
        return sorted_music_df

if __name__ == "__main__":
    data_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_data.csv"
    dataset = pd.read_csv(filepath_or_buffer=data_path, encoding="ISO-8859-1")
    train_data = dataset.head(5000)
    rank = train_data.groupby("title").sum().reset_index
    pd.set_option("display.width", 300)
    print(train_data.head(n=1))
    print("训练集中用户总数：%d，歌曲总数：%d" % (len(train_data.user.unique()),
                                                    len(train_data.title.unique())))
    user_id = train_data.loc[0].user
    print("需要推荐的用户：", user_id)
    recommender = Recommender(train_data, user_id)
    print("开始推荐...")
    sorted_music_df = recommender.get_sorted_music()
    print(sorted_music_df.shape)
    print("top10歌曲：")
    print(sorted_music_df.head(n=10))