# coding:utf-8

from scipy.sparse import coo_matrix, linalg
import numpy as np

# 首先构造一个稀疏矩阵
def create_sparse_matrix():
    data = np.array([5, 8, 9, 10, 4, 7, 8, 6, 9])
    row = [1, 1, 3, 3, 4, 5, 5, 6, 8]
    col = [2, 5, 1, 5, 3, 0, 4, 9, 7]
    matrix = coo_matrix((data, (row, col)), shape=(10, 10), dtype=float)
    return matrix

# k代表存在多少个隐变量
def get_svd(matrix, k):
    U, s, V = linalg.svds(matrix, k)
    shape = (len(s), len(s))
    S = np.zeros(shape=shape, dtype=np.float32)
    for i in range(len(s)):
        S[i][i] = s[i]
    temp = S.dot(V)
    new_matrix = np.round(U.dot(temp), 1)
    print(new_matrix)

# 重构矩阵
def recreate_matrix(u, v):
    pass

if __name__ == "__main__":
    matrix = create_sparse_matrix()
    print(matrix.toarray())
    print("------------")
    get_svd(matrix, 2)

