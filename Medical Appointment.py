#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:55:31 2017

@author: LuisClaudio
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
#%matplotlib inline

#Pessoas mais velhas faltam mais?
#Pessoas com doencas faltam menos?
#Qual eh o dia da semana que as pessoas mais faltam?
#O horario esta relacionado com as faltam
#Mulheres ou homens faltam memnos
#-------------
#Cluster
#Previsao de vai faltar
df = pd.read_csv('~/Documents/No-show-Issue-Comma-300k.csv').head(50000)

def select_columns(df, name_columns):
    select_columns = df[name_columns].copy
    return select_columns

def Week_day_to_int(df):
    week = ['Sunday', 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
    qtd_rows = len(df.index)
    for i in range(qtd_rows):
        j = 1
        for day in week:
            if df.loc[i, 'DayOfTheWeek'] == day:
                df.loc[i, 'DayOfTheWeek'] = j
            j = j + 1
    return df
df = Week_day_to_int(df)

def Gender_to_01(df):
    qtd_rows = len(df.index)
    for i in range(qtd_rows):
        if df.loc[i, 'Gender'] == 'M':
            df.loc[i, 'Gender'] = 1
        else:
            df.loc[i, 'Gender'] = 0
    return df
df = Gender_to_01(df)
#print df.head()

def age_range(df):
    qtd_rows = len(df.index)
    df['Age_range'] = 0
    for i in range(qtd_rows):
        age = df.iloc[i, 0]
        df.loc[i, 'Age_range'] = int(age/10)
    return df.Age_range
df['Age_range'] = age_range(df)
#print df.head()


def status_to_01(df):
    ix = 0
    for i in df['Status']:
        if i == 'Show-Up' or i == 1:
            df.loc[ix, 'Status'] = int(1)
        else:
            df.loc[ix, 'Status'] = int(0)
        ix = ix + 1
    df.Status.astype(int)
    return df
df = status_to_01(df)

def make_total_diseases(df):
    qtd_columns = len(df.columns)
    qtd_rows = len(df.index)
    df['total_diseases'] = 0
    for i in range(qtd_rows -1):
        df.iloc[i,qtd_columns] = df.iloc[i, :].sum()
    return df['total_diseases']

df['total_diseases'] = make_total_diseases(df.iloc[:, 6:11])

sequence = ['Age',
 'Age_range',
 'Gender',
 'AppointmentRegistration',
 'ApointmentData',
 'DayOfTheWeek',
 'Diabetes',
 'Alcoolism',
 'HiperTension',
 'Handcap',
 'Smokes',
 'total_diseases',
 'Scholarship',
 'Tuberculosis',
 'Sms_Reminder',
 'AwaitingTime',
 'Status',]
df = df.reindex(columns=sequence)

#Respondendo --------> Pessoas mais velhas faltam mais?

'''
faixa_idade = [0]*12
for i in df.Age:
    faixa_idade[int(i/10)] = faixa_idade[int(i/10)] + 1
faixa_idade_faltou = [0]*12
for i, j in zip(df.Age, df.Status):
    if j == 0:
        faixa_idade_faltou[int(i/10)] = faixa_idade_faltou[int(i/10)] + 1
percent_faltas = np.zeros(10)
for i in range(len(percent_faltas)):
    percent_faltas[i] = 100*faixa_idade_faltou[i]/faixa_idade[i]
    
#print faixa_idade
#print faixa_idade_faltou
#print percent_faltas
idade = []
for i in range(1, 11):
    idade.append(i*10)
#plt.hist(percent_faltas, bins = 10, color = ['Red'])
plt.figure(1)
plt.bar(idade, percent_faltas, 2, color="red")
plt.xlabel('Idade', fontsize=18)
plt.ylabel('% de Faltas', fontsize=16)


#Pessoas com doencas faltam menos?
faixa_doenca = [0]*10
for i in df.total_diseases:
    faixa_doenca[i] = faixa_doenca[i] + 1
    
faixa_doenca_faltou = [0]*10
for i, j in zip(df.total_diseases, df.Status):
    if j == 0:
        faixa_doenca_faltou[i] = faixa_doenca_faltou[i] + 1
        
percent_faltas = np.zeros(10)
for i in range(len(percent_faltas)):
    if faixa_doenca[i] != 0:
        percent_faltas[i] = 100*faixa_doenca_faltou[i]/faixa_doenca[i]
    
#print faixa_doenca
#print faixa_doenca_faltou
#print percent_faltas
qtd_doencas = range(1,11)

plt.figure(2)
plt.bar(qtd_doencas, percent_faltas, 0.5, color="blue")
plt.xlabel('qtd_doencas', fontsize=18)
plt.ylabel('% de Faltas', fontsize=16)


#Qual eh o dia da semana que as pessoas mais faltam?
dia_semana_faltas = np.zeros(7)
dia_semana = np.zeros(7)
for i, j in zip(df.DayOfTheWeek, df.Status):
    dia_semana[i - 1] = dia_semana[i -1] + 1
    if j == 0:
        dia_semana_faltas[i - 1] = dia_semana_faltas[i - 1] + 1
#print dia_semana
#print dia_semana_faltas
percent_semana = np.zeros(7)
for i in range(7):
    if dia_semana[i] != 0:
        percent_semana[i] = dia_semana_faltas[i]/dia_semana[i]
#print percent_semana

plt.figure(3)
plt.bar(range(1,8), percent_semana, 0.5, color="blue")
plt.xlabel('dia_da_semana', fontsize=18)
plt.ylabel('% de Faltas', fontsize=16)
'''

#Naive_Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

gnb = GaussianNB()
name_columns = ['Age', 'Gender', 'DayOfTheWeek', 'Diabetes', 'Alcoolism', 'HiperTension', 'Handcap', 'Smokes', 'Scholarship']
attributes_matrix = df.as_matrix(name_columns)
status_array = np.array(df['Status'], dtype = int)
X_train, X_test, Y_train, Y_test = train_test_split(attributes_matrix, status_array)
gnb.fit(X_train,Y_train)
predicted = gnb.predict(X_test)
expected = Y_test
#print gnb.predict([30, 1, 6, 1, 1, 1, 0, 1, 0])
print metrics.accuracy_score(expected, predicted)
print '\n\n'
del name_columns, X_train, X_test, Y_train, Y_test

from sklearn.svm import SVC
svm_model = SVC()
name_columns = ['Age',
 'Gender',
 'DayOfTheWeek',
 'Diabetes',
 'Alcoolism',
 'HiperTension',
 'Handcap',
 'Smokes',
 'Scholarship',
 'Tuberculosis',
 'Sms_Reminder',
 'AwaitingTime',
 'Status']
attributes_matrix_svm = df.as_matrix(name_columns)
X_train, X_test, Y_train, Y_test = train_test_split(attributes_matrix_svm, status_array)
svm_model.fit(X_train,Y_train)
predicted = svm_model.predict(X_test)
expected = Y_test
print metrics.accuracy_score(expected, predicted)