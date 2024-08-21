#************************双向均匀归一化预处理****************************
import numpy as np

K1 = 20
K2 = 20


# 从txt文件中读取数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            row = [float(x) for x in line.split()]
            data.append(row)
    return np.array(data)

# 列归一化
def column_normalize(matrix):
    normalized_matrix = np.zeros(matrix.shape)
    for j in range(matrix.shape[1]):
        column_sum = np.sum(matrix[:, j])
        normalized_matrix[:, j] = matrix[:, j] / column_sum
    return normalized_matrix



# 计算邻居集合N(包含自身)
def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)
    return N

# 行归一化(邻居归一化）
def row_normalization(S, N):
    t = 0.99
    k = len(N[0])
    weight = [t ** i for i in range(1, k + 1)]  # 生成权重矩阵
    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        num = 0
        denominator=np.sum(S[i, N[i]]) # 分母计算的是邻居的和
       # denominator= np.sum(S[i])   #分母计算的是一整行
        for j in range(S.shape[1]):
            if j in N[i]:
                if denominator != 0:
                    num = num + 1

                    result[i, j] = weight[num-1] * S[i, j] / denominator
                else:
                    result[i, j] = 0  # Set to 0 or handle differently based on your requirements
            else:
                result[i, j] = 0
    return result




GKS_drug = np.loadtxt('GKS_drug.txt')
GKS_disease = np.loadtxt('GKS_disease.txt')
LKS_drug = np.loadtxt('LKS_drug.txt')
LKS_disease = np.loadtxt('LKS_disease.txt')
lty = np.loadtxt('association.txt')
# 计算邻居集合N
N1 = calculate_neighbors(GKS_drug, K1)
N2 = calculate_neighbors(GKS_disease, K2)
N3 = calculate_neighbors(LKS_drug, K1)
N4 = calculate_neighbors(LKS_disease, K2)
# 执行列归一化
GKS_drug_col = column_normalize(GKS_drug)
GKS_disease_col = column_normalize(GKS_disease)
LKS_drug_col = column_normalize(LKS_drug)
LKS_disease_col = column_normalize(LKS_disease)
# 执行行归一化
GKS_drug_row = row_normalization(GKS_drug, N1)
GKS_disease_row = row_normalization(GKS_disease, N2)
LKS_drug_row = row_normalization(LKS_drug, N3)
LKS_disease_row = row_normalization(LKS_disease, N4)
#第二步
drug_P1=GKS_drug_col
drug_P2=LKS_drug_col
drug_S1=GKS_drug_row
drug_S2=LKS_drug_row
alpha_1 =0.5
disease_P1=GKS_disease_col
disease_P2=LKS_disease_col
disease_S1=GKS_disease_row
disease_S2=LKS_disease_row
drug_P2_t=drug_P2
drug_P1_t=drug_P1
for i in range(1000):
    drug_p1=alpha_1*(drug_S1@(drug_P2_t/2)@drug_S1.T)+(1-alpha_1)*(drug_P2/2)
    drug_p2=alpha_1*(drug_S2@(drug_P1_t/2)@drug_S2.T)+(1-alpha_1)*(drug_P1/2)
    err1 = np.sum(np.square(drug_p1-drug_P1_t))
    err2= np.sum(np.square(drug_p2-drug_P2_t))
    if (err1 < 1e-6) and (err2 < 1e-6):
        print("drug迭代的次数：",i)
        break
    drug_P2_t=drug_p2
    drug_P1_t=drug_p1
# #简单平均
drug_sl=0.5*drug_p1+0.5*drug_p2


#*******************************************************************************************
disease_P2_t=disease_P2
disease_P1_t=disease_P1
for j in range(1000):
    disease_p1=alpha_1*(disease_S1@(disease_P2_t/2)@disease_S1.T)+(1-alpha_1)*(disease_P2/2)
    disease_p2=alpha_1*(disease_S2@(disease_P1_t/2)@disease_S2.T)+(1-alpha_1)*(disease_P1/2)
    err1 = np.sum(np.square(disease_p1-disease_P1_t))
    err2= np.sum(np.square(disease_p2-disease_P2_t))
    if (err1 < 1e-6) and (err2 < 1e-6):
        print("disease迭代的次数：", i)
        break
    disease_P2_t=disease_p2
    disease_P1_t=disease_p1
disease_sl=0.5*disease_p1+0.5*disease_p2
#利用先验信息在各自的网络中进行迭代循环

alpha1 = 0.2
alpha2 = 0.9
for t in range(1):
    drug_sl = alpha1 * ( lty @  np.transpose(lty)  @  drug_sl) + (1-alpha1) * drug_sl
for t in range(1):
    disease_sl = alpha2 * ( np.transpose(lty) @ lty @ disease_sl ) + (1 - alpha2) * disease_sl
drug_sl = column_normalize(drug_sl)
disease_sl = column_normalize(disease_sl)

#--------------------------------------------------------------------
def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)
    return N
def compute_weighted_matrix(S1, k1):
    # 计算邻居集合 N_j 和 N_i
    N_i = calculate_neighbors(S1, k1)  # 行的邻居集合
    N_j = calculate_neighbors(S1.T, k1)  # 列的邻居集合
    # 生成 w 矩阵
    w = np.zeros((len(S1), len(S1)))

    for i in range(len(S1)):
        for j in range(len(S1)):
            if i in N_j[j] and j in N_i[i]:
                w[i][j] = 1
            elif i not in N_j[j] and j not in N_i[i]:
                w[i][j] = 0
            else:
                w[i][j] = 0.5
    return w
# 示例使用
w1 = compute_weighted_matrix(drug_sl, 20)
w2 = compute_weighted_matrix(disease_sl, 20)

average_drug = w1 @ drug_sl
average_disease = w2 @ disease_sl
np.savetxt('drug_SMF.txt',average_drug,fmt='%6f',delimiter='\t')
np.savetxt('disease_SMF.txt',average_disease,fmt='%6f',delimiter='\t')


