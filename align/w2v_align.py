import scipy.linalg
import pandas as pd
import numpy as np
from gensim.models import FastText
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import random

dic = pd.read_csv('Book1.csv').iloc[:,:].values

#corpus = pd.read_csv('new_corpus.csv', header=None).iloc[1:,1].values
#corpus = np.load("Banglish_total.npy")[:,0]
#corpus = pd.read_excel('IMDB_Dataset.xlsx').iloc[0:2000,0].values
#corpus = np.concatenate((corpus_1, corpus_2))
#corpus = pd.read_csv('sentiment_corpus.csv').iloc[:,0].values
#corpus = pd.read_csv('new_corpus.csv', header=None).iloc[1:,1].values

#corpus = pd.DataFrame(pd.read_csv('new_corpus.csv')['content']).iloc[:,:].values
"""df1 = pd.read_excel('banglish total.xlsx').iloc[:,:].values

sentiment = []
labels = []

for i in range(df1.shape[0]):
    try:
        if not np.isnan(df1[i,2]):
            if df1[i,2] != 0:
                sentiment.append(df1[i,0])
                labels.append(df1[i,2])
        else:
            sentiment.append(df1[i,0])
            labels.append(df1[i,1])
    except Exception:
        print('Problem encountered in index ', i)

sentiment = np.array(sentiment).reshape((-1,1))

labels = np.array(labels).reshape((-1,1))
df = np.concatenate((sentiment, labels), axis=1)"""
#df2 = pd.read_csv('sentiment_corpus.csv').iloc[0:5000,:].values
#df = np.concatenate((df1,df2),axis=0)

df = pd.read_csv('sentiment_corpus.csv').iloc[0:5000,:].values
corpus = df[:,0]

def tokenize(data, sampling_rate):
    data_list = []
    for i in range(len(data)):
        line = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        #lower_words = line.lower()
        tokens = nltk.word_tokenize(line)
        words = [w.lower() for w in tokens]
        data_list.append(words)
        for rate in sampling_rate:
            idxA = 0
            idxB = 0
            new_tokens = []
            for word in words:
                if word in dic and idxA == 0:
                    context_index = np.where(dic == word)
                    alpha = random.randint(0,len(context_index[0])-1)
                    row = context_index[0][alpha]
                    if context_index[1][0] == 0:
                        col = 1
                    else:
                        col = 0
                    context_word = str(dic[row,col])
                    new_tokens.append(context_word)
                    idxA = 1
                else:
                    new_tokens.append(word)
                    idxB += 1
                    #idxA = 0
                if idxB >= rate:
                    idxA = 0
                    idxB = 0
            if len(new_tokens) > 1:
                data_list.append(new_tokens)
            else:
                continue
    return data_list

tokens = tokenize(corpus, sampling_rate = [1,2,3])

"""new_train = []
del_index = []
#new_y = np.zeros((449,1))
for i in range(len(tokens)):
    if len(tokens[i]) > 1:
        new_train.append(tokens[i])
    else:    
        del_index.append(i) 
tokens = new_train"""

from gensim.models import FastText
import gensim
Embedding_Dim = 100
#model_ted = FastText(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)
model_ted = gensim.models.Word2Vec(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)

model_ted.wv.most_similar('good',topn=15)

model_ted.wv.most_similar('bad',topn=15)

model_ted.wv.most_similar('bhalo',topn=15)

#model_ted2 = FastText.load('saved_model_fastex')

model_ted.wv.most_similar('best',topn=15)

#model_ted2.wv.most_similar('bad',topn=15)

#model_ted2.wv.most_similar('bhalo',topn=15)


model_ted.save('saved_model_w2v_mask_banglish_zero_shot')
