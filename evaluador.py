#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:05:41 2017

@author: juan
"""
import pandas as pd
from sklearn.externals import joblib

pos = joblib.load('svc_pos.pkl')
neg = joblib.load('svc_neg.pkl')

import numpy as np

def dar_sentimiento(lista_tweets):
    df = pd.DataFrame({'texto': lista_tweets})
    df.texto = df.texto.astype(str)
    df['positivo'] = pos.predict(df.texto)
    df['negativo'] = neg.predict(df.texto)
    df['sentimiento'] = df['positivo'] - df['negativo']
    df.drop(['positivo','negativo'], axis=1, inplace=True)
    dictionary = {df.texto[i]: np.asscalar(df.sentimiento[i]) for i in range(len(df))}
    return dictionary

# Ejemplo 1
ejemplo = pd.read_csv('tuits_enrique.csv', header=0)
lista_tweets = ejemplo.tweet.values.tolist()
salida = dar_sentimiento(lista_tweets)
salida.to_csv('tuits_enrique_marcados.csv', sep='\t', encoding='utf-8')