import pandas as pd
import numpy as np

############################################################

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

spanish_stopwords = stopwords.words('spanish')

non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))

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

############################################################

from sklearn.externals import joblib

pos = joblib.load('linear_svc_pos.pkl') # Import the positive model
neg = joblib.load('linear_svc_neg.pkl') # Import the negative model


def dar_sentimiento(lista_tweets):
    df = pd.DataFrame({'texto': lista_tweets})
    df.texto = df.texto.astype(str)
    df['positivo'] = pos.predict(df.texto)
    df['negativo'] = neg.predict(df.texto)
    df['sentimiento'] = df['positivo'] - df['negativo']
    df.drop(['positivo','negativo'], axis=1, inplace=True)
    dictionary = {df.texto[i]: np.asscalar(df.sentimiento[i]) for i in range(len(df))}
    return dictionary
