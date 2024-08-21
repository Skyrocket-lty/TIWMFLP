import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


association = np.loadtxt(r"association.txt", dtype=int)




#计算药物高斯轮廓核相似性
def Gaussian():
    row = association.shape[0]
    sum=0
    drug1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(association[i,])*np.linalg.norm(association[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            drug1[i,j]=math.exp(-ps*np.linalg.norm(association[i,]-association[j,])*np.linalg.norm(association[i,]-association[j,]))


    GSM = drug1
    return GSM
#计算疾病高斯轮廓核相似性
def Gaussian1():
    column = association.shape[1]
    sum=0
    disease1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(association[:,i])*np.linalg.norm(association[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            disease1[i,j]=math.exp(-ps*np.linalg.norm(association[:,i]-association[:,j])*np.linalg.norm(association[:,i]-association[:,j]))


    GKS_disease = disease1
    return GKS_disease


def main():
    Gaussian()
    Gaussian1()



if __name__ == "__main__":

        main()