#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:40:44 2017

@author: juan
"""

import pandas as pd
pd.set_option('max_colwidth',1000)

# Si alguno de los archivos está en .xml, lo importamos y convertimos a un dataframe

from lxml import objectify

xml = objectify.parse(open('archivo.xml'))
root = xml.getroot()
tweets_corpus = pd.DataFrame(columns=('tweetid', 'content'))
tweets = root.getchildren()
for i in range(0,len(tweets)):
    tweet = tweets[i]
    row = dict(zip(['tweetid', 'content'], [tweet.tweetid.text, tweet.content.text]))
    row_s = pd.Series(row)
    row_s.name = i
    tweets_corpus = tweets_corpus.append(row_s)

tweets_corpus = tweets_corpus.sort_values('tweetid')
tweets_corpus[:15]
tweets_corpus = tweets_corpus.astype(str)

notas = pd.read_csv('general-sentiment-3l.csv', sep='\t', header=0, encoding='utf-8')
notas = notas.sort_values('tweetid')
notas[:15]
notas = notas.astype(str)

train_set = pd.merge(tweets_corpus, notas, on='tweetid', sort=False)
train_set = train_set.drop('tweetid', axis=1)

# Creamos dos nuevas columnas: positivo_bin y negativo_bin. Ponemos un 1 
# cuando el tuit sea positivo o negativo. Si no, ponemos un 0.
# Con esto, el problema cambio a ser dos problemas de clasificacion binaria y usamos
# ROC_AUC (area under the curve).
# Utilizamos como modelo Linear Support Vector Classifier from sklearn

train_set['positivo_bin']=0
train_set['negativo_bin']=0
train_set.positivo_bin[train_set.sentiment.isin(['P', 'P+'])] = 1
train_set.negativo_bin[train_set.sentiment.isin(['N', 'N+'])] = 1

# Agregamos stopwords y signos de puntuacion

from nltk.corpus import stopwords

spanish_stopwords = stopwords.words('spanish')

from string import punctuation

non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))

# Creamos una funcion que haga stem de las palabras.
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

stemmer = SnowballStemmer('spanish')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = ''.join([c for c in text if c not in non_words])
    # tokenize
    tokens =  word_tokenize(text)
    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

# Buscamos los mejores parametros para un LinearSVC que clasifique positivos
    
from sklearn.feature_extraction.text import CountVectorizer       
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])

parameters = {
    'vect__max_df': (0.5, 1.9),
    'vect__min_df': (10, 20, 50),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search.fit(train_set.content, train_set.positivo_bin)

grid_search.best_params_

# Guardamos el LinearSVC como pickle

from sklearn.externals import joblib
joblib.dump(grid_search, 'svc_pos.pkl')

# Hacemos cross-validation con 10 folds. En model ponemos los parámetros que obtuvimos
# con best_params_

from sklearn.model_selection import cross_val_score

model = LinearSVC(C=0.2, loss='hinge',max_iter=500,multi_class='ovr',
              random_state=None,
              penalty='l2',
              tol=0.0001
)

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = spanish_stopwords,
    min_df = 10,
    max_df = 0.5,
    ngram_range=(1, 2),
    max_features=1000
)

corpus_data_features = vectorizer.fit_transform(train_set.content)
corpus_data_features_nd = corpus_data_features.toarray()

scores = cross_val_score(
    model,
    corpus_data_features_nd[0:len(train_set)],
    y=train_set.positivo_bin,
    scoring='roc_auc',
    cv=5
    )

scores.mean()

# Buscamos los mejores parametros para un LinearSVC que clasifique negativos

vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])

parameters = {
    'vect__max_df': (0.5, 1.9),
    'vect__min_df': (10, 20,50),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000),
}

grid_search2 = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search2.fit(train_set.content, train_set.negativo_bin)

grid_search2.best_params_

# Guardamos el LinearSVC como pickle

joblib.dump(grid_search2, 'svc_neg.pkl')

# Hacemos cross-validation con 10 folds. En model ponemos los parámetros que obtuvimos
# con best_params_

model = LinearSVC(C=0.2, loss='hinge',max_iter=500,multi_class='ovr',
              random_state=None,
              penalty='l2',
              tol=0.0001
)

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = spanish_stopwords,
    min_df = 10,
    max_df = 0.5,
    ngram_range=(1, 2),
    max_features=1000
)

corpus_data_features = vectorizer.fit_transform(train_set.content)
corpus_data_features_nd = corpus_data_features.toarray()

scores = cross_val_score(
    model,
    corpus_data_features_nd[0:len(train_set)],
    y=train_set.negativo_bin,
    scoring='roc_auc',
    cv=5
    )

scores.mean()

# Entrenamiento de un modelo con parametros basados en la literatura

vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])

parameters = {
    'vect__max_df': [0.5],
    'vect__min_df': [10],
    'vect__max_features': [1000],
    'vect__ngram_range': [(1, 2)],  # unigrams or bigrams
    'cls__C': [0.2],
    'cls__loss': ['hinge'],
    'cls__max_iter': [500],
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search.fit(train_set.content, train_set.positivo_bin)
joblib.dump(grid_search, 'svc_pos.pkl')

grid_search2 = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search2.fit(train_set.content, train_set.negativo_bin)
joblib.dump(grid_search2, 'svc_neg.pkl')