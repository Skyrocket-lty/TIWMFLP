# -*- codeing = utf-8 -*-
# @Time : 2023/3/24 21:17
# @Author : 刘体耀
# @File : TSPN.py
# @Software: PyCharm


import numpy as np

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy
import weight


drug = np.loadtxt(r"drug_SMF.txt", dtype=float)
disease = np.loadtxt(r"disease_SMF.txt", dtype=float)

Y0= np.loadtxt(r"association.txt",dtype=float)

drug_disease_k = np.loadtxt(r"known.txt",dtype=int)
drug_disease_uk = np.loadtxt(r"unknown.txt",dtype=int)

drug_disease_uk_18416 = np.loadtxt(r"drug_disease_uk_18416.txt",dtype=int)

drugW = weight.calculate_weight_matrix(r"drug_SMF.txt", 10)
diseaseW = weight.calculate_weight_matrix(r"disease_SMF.txt", 10)

# #第一层矩阵分解进行处理
def TCMF3(alpha, Y, maxiter,A,B,C):
    iter0=1
    while True:
        a = np.dot(Y,B)
        b = np.dot(np.transpose(B),B)+alpha*C
        A = np.dot(a, np.linalg.inv(b))
        c = np.dot(np.transpose(Y),A)
        d = np.dot(np.transpose(A), A) + alpha * C
        B = np.dot(c, np.linalg.inv(d))
        #先给他俩搞一个加权
        if iter0 >= maxiter:
            #print('reach maximum iteration!')
            break
        iter0 = iter0 + 1
    Y= np.dot(A,np.transpose(B))
    Y_recover = Y
    return Y_recover


def run_MC_3(Y):
    maxiter = 300
    alpha = 0.1
    #SVD
    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r =  20
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C=Wt
    Y = TCMF3(alpha, Y, maxiter,A,B,C)
    dda = Y
    return dda

drug = run_MC_3(drug)
disease = run_MC_3(disease)
Y1 = run_MC_3(Y0)


#对关联矩阵进行处理
def TCMF1(alpha, beta,gamma, Y, maxiter,A,B,C,drug,disease):
    iter0=1
    while True:

        a = np.dot(Y,B)+beta*np.dot(np.multiply(drugW,drug),A)
        b = np.dot(np.transpose(B),B)+alpha*C+beta*np.dot(np.transpose(A),A)
        A = np.dot(a, np.linalg.inv(b))
        c = np.dot(np.transpose(Y),A)+gamma*np.dot(np.multiply(diseaseW,disease),B)
        d = np.dot(np.transpose(A), A) + alpha * C + gamma * np.dot(np.transpose(B), B)
        B = np.dot(c, np.linalg.inv(d))
        #先给他俩搞一个加权
        drug = np.dot(A, np.transpose(A))  # 这个是确定的  ，  这就已经是交互式啦
        disease = np.dot(B, np.transpose(B)) # 这个是确定的   ， 这就已经是交互式啦
        if iter0 >= maxiter:
            #print('reach maximum iteration!')
            break
        iter0 = iter0 + 1


    Y= np.dot(A,np.transpose(B))
    Y_recover = Y
    return Y_recover



#矩阵分解算法
def run_MC_1(Y):
    maxiter = 500
    alpha = 0.1
    beta = 0.01
    gamma = 0.01
    #SVD
    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 22
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C=Wt
    Y = TCMF1(alpha, beta,gamma,Y, maxiter,A,B,C,drug,disease)
    DDA = Y
    return DDA

Y2 = run_MC_1(Y1)
#标签传播算法
def run_MC_2(A):
    gama= 0.2
    beta =0.4
    PT = A.T
    PT0 = A
    P0 = A.T
    P1 = A
    delta = 1
    # 列归一化
    M = np.sum(drug, axis=1)

    for i in range(269):
        for j in range(269):
            drug[i, j] = drug[i, j] / np.sqrt(M[i] * M[j])

    D = np.sum(disease, axis=1)
    for i in range(598):
        for j in range(598):
            disease[i, j] = disease[i, j] / np.sqrt(D[i] * D[j])


    while delta > 1e-6:
        PT1 = gama * np.dot(disease, PT) + (1 - gama) * (P0+np.transpose(Y2))/2
        # PT1 = gama * np.dot(disease, PT) + (1 - gama) * P0
        delta = np.abs(np.sum(np.abs(PT1) - np.abs(PT)))
        PT = PT1

    # prediction from Disease space
    delta = 1
    while delta > 1e-6:
        DD = gama * np.dot(drug, PT0) + (1 - gama) * (P1+Y2)/2
        # DD = gama * np.dot(drug, PT0) + (1 - gama) * P1
        delta = np.abs(np.sum(np.abs(DD) - np.abs(PT0)))
        PT0 = DD

    F = (beta * PT1 + (1 - beta) * DD.T).T

    return F

def main():

   run_MC_2(Y0)



if __name__ == "__main__":
      main()



