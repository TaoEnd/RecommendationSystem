# coding:utf-8

import sqlite3
import pandas as pd

# music_playcount_path = r"E:\python\PythonSpace\Data\recommendationsystem\music_playcount.csv"
# music_playcount_df = pd.read_csv(filepath_or_buffer=music_playcount_path)
# music_subset = list(music_playcount_df.head(50000).music)
# music_dict = {}
# for music in music_subset:
#     music_dict.update({music: 1})
#
# # 获得每首歌的信息，并将其写入本地csv文件中
# db_path = r"E:\python\PythonSpace\Data\recommendationsystem\track_metadata.db"
# conn = sqlite3.connect(db_path)
# cur = conn.cursor()
# # cur.execute("select name from sqlite_master where type = 'table'")
# # print(cur.fetchall())
# metadata_df = pd.read_sql(con=conn, sql="select * from songs")
# cur.close()
#
# metadata_df_sub = metadata_df[metadata_df.song_id.isin(music_dict)]
# metadata_df_sub = metadata_df_sub.drop_duplicates(["song_id"])
# del(metadata_df_sub["track_id"])
# del(metadata_df_sub["artist_mbid"])
# del(metadata_df_sub["artist_id"])
# del(metadata_df_sub["duration"])
# del(metadata_df_sub["artist_familiarity"])
# del(metadata_df_sub["artist_hotttnesss"])
# del(metadata_df_sub["track_7digitalid"])
# del(metadata_df_sub["shs_perf"])
# del(metadata_df_sub["shs_work"])
# metadata_sub_path = r"E:\python\PythonSpace\Data\recommendationsystem\metadata_sub.csv"
# metadata_df_sub.to_csv(path_or_buf=metadata_sub_path, index=False)
# pandas获取列名：metadata_df_sub.columns.values.tolist()
# print(metadata_df_sub.shape)
# print(metadata_df_sub.columns.values.tolist())

# 将音乐信息和训练样本拼接在一起，得到最终的样本
metadata_sub_path = r"E:\python\PythonSpace\Data\recommendationsystem\metadata_sub.csv"
metadata_df_sub = pd.read_csv(filepath_or_buffer=metadata_sub_path, encoding="ISO-8859-1")
dataset_sub_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_triplets_sub.csv"
dataset_sub_df = pd.read_csv(filepath_or_buffer=dataset_sub_path, header=None,
                          names=["user", "song_id", "listen_count"], encoding="ISO-8859-1")
train_data_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_data.csv"
train_data_df = pd.merge(dataset_sub_df, metadata_df_sub, how="left",
                         left_on="song_id", right_on="song_id")
del(train_data_df["song_id"])
train_data_df.to_csv(path_or_buf=train_data_path, index=False)
print(train_data_df.shape)
print(train_data_df.columns.values.tolist())