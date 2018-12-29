# coding:utf-8

import pandas as pd

path = r"E:\python\PythonSpace\Data\recommendationsystem\movie\ratings.dat"
rewrite_path = r"E:\python\PythonSpace\Data\recommendationsystem\movie\ratings_1.dat"
data = pd.read_table(filepath_or_buffer=path, header=None, sep="::")
data.columns = ["user", "movie", "score", "time"]
movie_replace_dict = {}
movies = data.movie.unique().tolist()
movies.sort()
print(movies)
index = 1
for movie in movies:
    movie_replace_dict[movie] = index
    index += 1

line = ""
with open(rewrite_path, "w") as fw:
    for i in range(data.shape[0]):
        if i % 10000 == 0 or i == data.shape[0]-1:
            fw.write(line)
            print(i)
            line = ""
        movie_id = data.movie[i]
        data.movie[i] = movie_replace_dict[movie_id]
        line =  line + str(data.user[i]) + "::" + str(data.movie[i]) + "::" \
                + str(data.score[i]) + "::" + str(data.time[i]) + "\n"
# (3408, 3178)
print(data.head())
print(type(data.loc[0][2]), data.loc[0][2])
