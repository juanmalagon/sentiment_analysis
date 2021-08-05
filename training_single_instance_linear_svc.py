import pandas as pd
from lxml import objectify
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
#import statsmodels as sm
#import matplotlib.pyplot as plt

# Import the TASS 2017 General Corpus (tweetid and content)

corpus = pd.read_csv('corpus_TASS_2017.csv', sep='\t', header=0, encoding='utf-8')

# Read XML infro from TASS 2017

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

# Open the sentiment for each tweet and merge it with the previous dataframe

notas = pd.read_csv('general-sentiment-3l.csv', sep='\t', header=0, encoding='utf-8')
notas = notas.sort_values('tweetid')
notas[:15]
notas = notas.astype(str)

corpus = pd.merge(tweets_corpus, notas, on='tweetid', sort=False)
del corpus['tweetid']

# The documents with sentiment NONE or NEU are clustered together as NEU 

corpus = pd.get_dummies(corpus, columns=['sentiment'])
corpus['sentiment_NEU'] = corpus['sentiment_NEU'] + corpus['sentiment_NONE']
del corpus['sentiment_NONE']

# Save the corpus as CSV

corpus.to_csv('corpus2017.csv', sep='\t', encoding='utf-8')

# Define variables X (features) and y (target). The target variable is 'sentiment_P' (use the commented lines to set 'sentiment_N' as target variable)

X = corpus['content'].as_matrix()
y = corpus['sentiment_P'].as_matrix()
y = y = corpus['sentiment_N'].as_matrix()

# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Import Spanish stopwords and punctuation

spanish_stopwords = stopwords.words('spanish')
non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))
stemmer = SnowballStemmer('spanish')

# Define a function for tokenizing using stemming and removing stopwords

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

# Create a model using CountVectorizer and Linear SVC

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
}

# Grid search for the best parameters

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(X_train, y_train)

grid_search.best_params_

# Save the model as pickle

joblib.dump(grid_search, 'linear_svc_pos.pkl')
joblib.dump(grid_search, 'linear_svc_neg.pkl')

# Calculate MSE for training and test sets

mse = mean_absolute_error(y_train, grid_search.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

mse = mean_absolute_error(y_test, grid_search.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

# Load the model and calculate the score of the model with the test set

pos = joblib.load('linear_svc_pos.pkl')
neg = joblib.load('linear_svc_neg.pkl')
pos.score(X_test, y_test)
neg.score(X_test, y_test)

# Cross-validation for the model

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

corpus_data_features = vectorizer.fit_transform(X_train)
corpus_data_features_nd = corpus_data_features.toarray()

scores = cross_val_score(
    model,
    corpus_data_features_nd[0:len(X_train)],
    y=y_train,
    cv=5
    )

scores.mean()
