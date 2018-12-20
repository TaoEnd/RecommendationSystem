# coding:utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train_data_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_data.csv"
train_data_df = pd.read_csv(filepath_or_buffer=train_data_path, encoding="ISO-8859-1")
# reset_index()：为每一行增加一个新的整型索引（0、1、2....）
popular_songs = train_data_df[["title", "listen_count"]].groupby("title").sum().reset_index()
popular_songs_top20 = popular_songs.sort_values("listen_count", ascending=False).head(n=20)

# 前20首最受欢迎的歌
objects = list(popular_songs_top20["title"])
x_pos = np.arange(len(objects))
y = list(popular_songs_top20["listen_count"])
plt.bar(x_pos, y, align="center", alpha=0.8)
plt.xticks(x_pos, objects, rotation="vertical")
plt.ylabel("Listen Count")
plt.title("Most Popular Songs")
plt.show()

# 统计每个用户的听了多少不同的歌
user_song_count_distribution = train_data_df[["user", "title"]].groupby("user").count().reset_index()
user_song_count_distribution = user_song_count_distribution.sort_values("title", ascending=False)
print(user_song_count_distribution.describe())

# 画出每个用户听歌数量的分布图
x = user_song_count_distribution.title
# bins=50，表示分成50个箱子
n, bins, patches = plt.hist(x, 50, facecolor="green", alpha=0.5)
plt.xlabel("Play Counts")
plt.ylabel("Num of Users")
plt.title("Histogram of User Play Count")
plt.grid(True)
plt.show()