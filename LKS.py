import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


association = np.loadtxt(r"association.txt", dtype=int)



#计算药物拉普拉斯核相似性
def Laplace():
    row = association.shape[0]
    a = 2
    drug1 = np.matlib.zeros((row,row))

    for i in range(0,row):
        for j in range(0,row):
            drug1[i,j]=math.exp(-(1/a)*np.linalg.norm((association[i,]-association[j,])))


    LKS_drug = drug1
    return LKS_drug



#计算疾病拉普拉斯核相似性
def Laplace1():
    column = association.shape[1]
    a = 2
    disease1 = np.matlib.zeros((column,column))

    for i in range(0,column):
        for j in range(0,column):
            disease1[i,j]=math.exp(-(1/a)*np.linalg.norm((association[:,i]-association[:,j])))
    LKS_disease = disease1
    return LKS_disease



def main():
    Laplace()
    Laplace1()


if __name__ == "__main__":

        main()