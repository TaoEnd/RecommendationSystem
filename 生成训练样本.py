# coding:utf-8

import pandas as pd
import csv
import time

data_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_triplets.txt"
user_playcount_path = r"E:\python\PythonSpace\Data\recommendationsystem\user_playcount.csv"
music_playcount_path = r"E:\python\PythonSpace\Data\recommendationsystem\music_playcount.csv"
dataset_sub_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_triplets_sub.csv"
user_playcount_df = pd.read_csv(filepath_or_buffer=user_playcount_path)
music_playcount_df = pd.read_csv(filepath_or_buffer=music_playcount_path)
print("用户量：%d，歌曲量：%d" % (user_playcount_df.shape[0], music_playcount_df.shape[0]))
user_count_subset = user_playcount_df.head(300000)
music_count_subset = music_playcount_df.head(50000)
user_subset = list(user_count_subset.user)
music_subset = list(music_count_subset.music)
user_dict = {}
music_dict = {}
for user in user_subset:
    user_dict.update({user: 1})
for music in music_subset:
    music_dict.update({music: 1})

print("用户使用量：%d，歌曲使用量：%d" % (len(user_dict), len(music_dict)))

print("开始...")
with open(data_path, "r") as fr:
    lines = fr.readlines()
    li = []
    num = len(lines)
    sub_num = 0
    for index, line in enumerate(lines):
        temp = line.strip().split("\t")
        if temp[0] in user_dict and temp[1] in music_dict:
            li.append(temp)
        if len(li) % 50000 == 0 or index == num-1:
            sub_num += len(li)
            with open(dataset_sub_path, "a+", newline="") as fw:
                writer = csv.writer(fw)
                writer.writerows(li)
            time.sleep(0.5)
            li = []
        if (index+1) % 50000 == 0 or index == num-1:
            print("第%d条数据" % (index+1))
    print("总样本量：%d，使用样本量：%d" % (num, sub_num))
