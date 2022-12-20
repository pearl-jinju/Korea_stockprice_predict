import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import params
import torch.nn as f


# Dataset 불러오기
data = params.DATA_SET


# load
with open(data, 'rb') as f:
    vector_dataset = pickle.load(f)


# 기초 전처리==========
# 현재 종목정보는 필요없음 
vector_dataset = vector_dataset.iloc[:,:-1]
# Outlier 제거 (5연상한가, 5연하한가 범위)
vector_dataset = vector_dataset[(vector_dataset[50]<271)&(vector_dataset[50]>-85)]

def label(x):
    if x <-15 :
        return "대폭락 예상(-15%~)"
    elif x<-8:
        return "폭락 예상(-8% ~ -15%)"
    elif x<-3:
        return "하락 예상(-3% ~ -8%)"
    elif x<-1:
        return "소폭 하락 예상(-1% ~ -3%)"
    elif x<0:
        return "보합 예상(-1 ~ 1%)"
    elif x < 1 :
        return "보합 예상(-1 ~ 1%)"
    elif x < 3 :
        return "소폭 상승 예상(1% ~ 3%)"
    elif x < 8 :
        return "상승 예상(3% ~ 8%)"
    elif x<15:
        return "폭등 예상(8% ~ 15%)"
    else:
        return "대폭등 예상(+15%~)"

# target 
vector_dataset[50] = vector_dataset[50].apply(label) 
# vector_dataset = vector_dataset.iloc[:,:-1]

# train_x, test_x, train_y, test_y = train_test_split(vector_dataset.iloc[:,:-1], vector_dataset.iloc[:,-1:], test_size=0.1, random_state=1)