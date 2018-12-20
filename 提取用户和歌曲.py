# coding:utf-8

import numpy as np
import pandas as pd

data_path = r"E:\python\PythonSpace\Data\recommendationsystem\train_triplets.txt"
# dataset = pd.read_csv(filepath_or_buffer=data_path, sep="\t", header=None,
#                       names=["user", "music", "play_count"])

user_playcount_dict = {}  # 记录每个用户的总音乐播放量
music_playcount_dict = {}  # 记录每首音乐的总播放量
with open(data_path, "r") as fr:
    lines = fr.readlines()
    for index, line in enumerate(lines):
        if (index+1) % 100000 == 0:
            print("第%d行" % (index+1))
        temp = line.strip().split("\t")
        user = temp[0]
        music = temp[1]
        play_count = int(temp[2])
        if user in user_playcount_dict:
            user_playcount_dict.update({user: play_count+user_playcount_dict[user]})
        else:
            user_playcount_dict.update({user: play_count})

        if music in music_playcount_dict:
            user_playcount_dict.update({music: play_count+music_playcount_dict[music]})
        else:
            music_playcount_dict.update({music: play_count})

    user_playcount_list = [{"user": k, "play_count": v} for k, v in user_playcount_dict.items()]
    user_playcount_df = pd.DataFrame(user_playcount_list)
    user_playcount_df = user_playcount_df.sort_values(by="play_count", ascending=False)

    music_playcount_list = [{"music": k, "play_count": v} for k, v in music_playcount_dict.items()]
    music_playcount_df = pd.DataFrame(music_playcount_list)
    music_playcount_df = music_playcount_df.sort_values(by="play_count", ascending=False)

    user_playcount_path = r"E:\python\PythonSpace\Data\recommendationsystem\user_playcount.csv"
    music_playcount_path = r"E:\python\PythonSpace\Data\recommendationsystem\music_playcount.csv"
    user_playcount_df.to_csv(path_or_buf=user_playcount_path, index=False)
    music_playcount_df.to_csv(path_or_buf=music_playcount_path, index=False)