#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:40:43 2019

@author: shourya
"""

import numpy as np
import csv
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


label_encoder = preprocessing.LabelEncoder()
#classes = ['false','true', 'mostly-true','half-true','barely-true','pants-fire']
labels = ['sl.no', 'json', 'label', 'sentence', 'topic','person', 'post', 'state', 'party','no.1','no.2','no.3', 'no.4','no.5','where','justification']

train_raw = pd.read_csv('train.tsv',sep='\t').fillna(' ')
train_raw.columns = labels
train_raw.to_csv('train_2.csv')
train = pd.read_csv('train_2.csv').fillna(' ')
val = input("Enter 2 if you want binary and 6 if 6 class: ")
if(val == '2'):
	train = train.replace({'label': {'mostly-false': 'false', 'pants-fire': 'false', 'half-true':'true', 'mostly-true': 'true', 'barely-true':'true'}})
else:
	train = train
classes = train.label.unique().tolist()
label_encoder.fit(classes)

test_raw = pd.read_csv('test.tsv',sep='\t').fillna(' ')
test_raw.columns = labels
test_raw.to_csv('test_2.csv')
test = pd.read_csv('test_2.csv').fillna(' ')
if(val == '2'):
	test = test.replace({'label': {'mostly-false': 'false', 'pants-fire': 'false', 'half-true':'true', 'mostly-true': 'true', 'barely-true':'true'}})
else:
	test = train
train_text = train['sentence']
test_text = test['sentence']

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1))
    

word_vectorizer.fit(train_text.squeeze())
train_features = word_vectorizer.transform(train_text.squeeze())
test_features = word_vectorizer.transform(test_text.squeeze())

train_target = train['label'].values.ravel()
train_target = label_encoder.transform(train_target)
onehot_encoder = OneHotEncoder(sparse=False)
train_target = train_target.reshape(len(train_target), 1)
onehot_encoded_train = onehot_encoder.fit_transform(train_target)

test_target = test['label'].values.ravel()
test_target = label_encoder.transform(test_target)
test_target = test_target.reshape(len(test_target), 1)
onehot_encoded_test = onehot_encoder.fit_transform(test_target)
    
accuracy_measure = []
count = np.unique(train_target)
for i in count:
    train_target1 = onehot_encoded_train[:,i]
    test_target1 = onehot_encoded_test[:,i]

    classifier = LogisticRegression(C=0.1, solver='sag')

    classifier.fit(train_features, train_target1)
    c = classifier.predict(test_features)
    acc = accuracy_score(test_target1, c)
    print('Accuracy for class {} is {}'.format(i,acc))
    accuracy_measure.append(acc)
print('Total Accuracy is {}'.format(np.mean(accuracy_measure)))
