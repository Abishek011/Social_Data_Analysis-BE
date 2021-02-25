import nltk
import numpy as np
import re
import pandas as pd 
import pylab as pl
import matplotlib.pyplot as plt

from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn import metrics
from sklearn.cluster import KMeans 
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot
""" 
#plt.style.use('fivethirtyeight')
%matplotlib inline
%config InlineBackend.figure_format = 'retina' """

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

import datashader as ds
import datashader.transfer_functions as tf

df = pd.read_csv("./pyFiles/dataset.csv",encoding='UTF-8')
print(len(df))

unique_text = df.full_text.unique()
print(len(unique_text))


# Number of unique users
unique_user = df.user.unique()
len(unique_user)

df.head(2)

# Number of Unique Locations
unique_location = df.location.unique()
len(unique_location)
df.location.value_counts().head(10)

df.full_text.count()

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

df['Clean_text'] = np.vectorize(remove_pattern)(df['full_text'], "@[\w]*")

# remove special characters, numbers, punctuations
df['Clean_text'] = df['Clean_text'].str.replace("[^a-zA-Z#]", " ")

df['Clean_text'] = df['Clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_tweet = df['Clean_text'].apply(lambda x: x.split())

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['Clean_text'] = tokenized_tweet

df.loc[:,('full_text','Clean_text')]

# Number of unique tweets
unique_clean_text = df.Clean_text.unique()
unique_full_text = df.full_text.unique()
print(len(unique_clean_text))
print(len(unique_full_text))
print(len(df))

df.drop_duplicates(subset=['Clean_text'], keep = 'first',inplace= True)

df.reset_index(drop=True,inplace=True)

df['Clean_text_length'] = df['Clean_text'].apply(len)
df.head()


df[df['Clean_text_length']==0]['Clean_text'] ## Looks like these are tweets with different languages or just hastags.
# We can simply drop these tweets
list = df[df['Clean_text_length']==0]['Clean_text'].index
list

df.drop(index = list,inplace=True)

df.info()

df.reset_index(drop=True,inplace=True)
df.info()

def calculate_sentiment(Clean_text):
    return TextBlob(Clean_text).sentiment

def calculate_sentiment_analyser(Clean_text):    
    return analyser.polarity_scores(Clean_text)

df['sentiment']=df.Clean_text.apply(calculate_sentiment)
df['sentiment_analyser']=df.Clean_text.apply(calculate_sentiment_analyser)


s = pd.DataFrame(index = range(0,len(df)),columns= ['compound_score','compound_score_sentiment'])

for i in range(0,len(df)): 
  s['compound_score'][i] = df['sentiment_analyser'][i]['compound']
  
  if (df['sentiment_analyser'][i]['compound'] <= -0.05):
    s['compound_score_sentiment'][i] = 'Negative'    
  if (df['sentiment_analyser'][i]['compound'] >= 0.05):
    s['compound_score_sentiment'][i] = 'Positive'
  if ((df['sentiment_analyser'][i]['compound'] >= -0.05) & (df['sentiment_analyser'][i]['compound'] <= 0.05)):
    s['compound_score_sentiment'][i] = 'Neutral'
    
df['compound_score'] = s['compound_score']
df['compound_score_sentiment'] = s['compound_score_sentiment']
df.head(4)

df.to_csv('dataset2.csv')

df.compound_score_sentiment.value_counts()

df['Clean_text'].head()

#tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Considering 3 grams and mimnimum frq as 0
tf_idf_vect = CountVectorizer(analyzer='word',ngram_range=(1,1),stop_words='english', min_df = 0.0001)
tf_idf_vect.fit(df['Clean_text'])
desc_matrix = tf_idf_vect.transform(df["Clean_text"])

# implement kmeans
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()

# create DataFrame films from all of the input files.
tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
frame = pd.DataFrame(tweets, index = [clusters])
frame


frame['Cluster'].value_counts()

frame[frame['Cluster'] == 1]

frame[frame['Cluster'] == 2]