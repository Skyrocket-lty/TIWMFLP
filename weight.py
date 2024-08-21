import numpy as np


def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)
    return N


def calculate_weight_matrix(file_path, k):
    """
    计算加权矩阵并将其保存到文件
    参数：
    file_path (str): 输入矩阵的文件路径
    k (int): 邻居数
    output_path (str): 输出加权矩阵的文件路径
    """
    # 加载矩阵
    S_L = np.loadtxt(file_path)

    # 计算邻居集合 N_j 和 N_i
    N_i = calculate_neighbors(S_L, k=k)  # 这是求的行的邻居集合
    N_j = calculate_neighbors(S_L.T, k=k)  # 这是求的列的邻居集合

    # 生成 w 矩阵
    w = np.zeros((len(S_L), len(S_L)))

    for i in range(len(S_L)):
        for j in range(len(S_L)):
            if i in N_j[j] and j in N_i[i]:
                w[i][j] = 1
            elif i not in N_j[j] and j not in N_i[i]:
                w[i][j] = 0
            else:
                w[i][j] = 0.5

    # # 保存加权矩阵到文件
    # np.savetxt(output_path, w, fmt='%f', delimiter='\t')
    return w
