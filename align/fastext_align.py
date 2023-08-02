import scipy.linalg
import pandas as pd
import numpy as np
from gensim.models import FastText
import re
import nltk
#nltk.download('stopwords')
#nltk.download('words')
#nltk.download('wordnet')
from nltk.corpus import stopwords
#import enchant 
#from banglish_detector import Banglish_English_Detector
from collections import Counter
import random

dic = pd.read_csv('Book1.csv').iloc[:,:].values


"""modify_dic_bn = []
modify_dic_en = []
for i in range(len(dic)):
    en_line = re.sub('[^a-zA-Z]', ' ', str(dic[i,1]))
    bn_line = re.sub('[^a-zA-Z]', ' ', str(dic[i,0]))
    modify_dic_en.append(en_line)
    modify_dic_bn.append(bn_line)
    en_tokens = nltk.word_tokenize(en_line)
    bn_tokens = nltk.word_tokenize(bn_line)
    if len(en_tokens)<1:
        continue
    else:
        en_longest_string = max(en_tokens, key=len)
        modify_dic_en.append(en_longest_string)
        bn_longest_string = max(bn_tokens, key=len)
        modify_dic_bn.append(bn_longest_string)

modify_dic_en = np.array(modify_dic_en).reshape((-1,1))    
modify_dic_bn = np.array(modify_dic_bn).reshape((-1,1))    
dic = np.concatenate((modify_dic_bn,modify_dic_en),axis=1)"""


"""data2 = pd.read_csv('total_205619_reviews_bkash.csv')
label = data2.iloc[:, 4].values
data2 = data2['content']
data2 = pd.DataFrame(data2).iloc[:,:].values

bn_en_detector = Banglish_English_Detector()
(banglish_count, banglish_index, english_count, english_index) = bn_en_detector.detector('total_205619_reviews_bkash.csv')

banglish_sentences = data2[banglish_index]
banglish_label = label[banglish_index]

english_sentences = data2[english_index]
english_label = label[english_index]

label2num = {1:0, 2:0, 3:0, 4:0, 5:0}
for i in range(len(banglish_sentences)):
    label_ = banglish_label[i]
    label2num[label_] = label2num[label_] + 1
 

def english_sentiment(english_sentences, english_label, label, dictonary):
    dataset = []
    data_label = []
    
    for i in range(len(english_sentences)):
        tokens = nltk.word_tokenize(english_sentences[i,0])
        if len(tokens) > 5 and english_label[i] == label and (label == 2 or label == 3 or label == 4) and len(data_label) < dictonary[label]+4000:
            dataset.append(english_sentences[i,0])
            data_label.append(english_label[i])
        if len(tokens) > 5 and english_label[i] == label and (label == 1 or label == 5) and len(data_label) < 1000:
            dataset.append(english_sentences[i,0])
            data_label.append(english_label[i])
    return np.array(dataset), np.array(data_label)
    
english_1, label_1 = english_sentiment(english_sentences,english_label,1,label2num)
english_2, label_2 = english_sentiment(english_sentences,english_label,2,label2num)
english_3, label_3 = english_sentiment(english_sentences,english_label,3,label2num)
english_4, label_4 = english_sentiment(english_sentences,english_label,4,label2num)
english_5, label_5 = english_sentiment(english_sentences,english_label,5,label2num)

total_english = np.concatenate((english_1,english_2,english_3,english_4,english_5),axis=0).reshape((-1,1))
total_english_label = np.concatenate((label_1,label_2,label_3,label_4,label_5),axis=0).reshape((-1,1))"""
"""total_label = np.concatenate((total_english_label,banglish_label.reshape((-1,1))),axis=0) 

corpus = np.concatenate((total_english,banglish_sentences),axis=0)

sentimene_dataset = np.concatenate((corpus,total_label),axis=1) 

dataframe = pd.DataFrame(corpus)
dataframe.to_csv('new_corpus_v1.csv')
sentiment_dataframe = pd.DataFrame(sentimene_dataset)
sentiment_dataframe.to_csv('sentiment_corpus_v2.csv',index=False)"""

"""corpus = total_english
sentimene_dataset = np.concatenate((corpus,total_english_label),axis=1) 

dataframe = pd.DataFrame(corpus)
dataframe.to_csv('new_corpus_v1.csv')
sentiment_dataframe = pd.DataFrame(sentimene_dataset)
sentiment_dataframe.to_csv('sentiment_corpus_v3.csv',index=False)

temp = sentimene_dataset
label_num = {'1.0':0, '2.0':0, '3.0':0, '4.0':0, '5.0':0}
for i in range(temp.shape[0]):
    label = temp[i,1]
    label_num[label] += 1 """

#corpus = pd.read_csv('new_corpus.csv', header=None).iloc[1:,1].values
#corpus = np.load("Banglish_total.npy")[:,0]
#corpus = pd.read_excel('IMDB_Dataset.xlsx').iloc[0:2000,0].values
#corpus = np.concatenate((corpus_1, corpus_2))
#corpus = pd.read_csv('sentiment_corpus.csv').iloc[:,0].values
#corpus = pd.read_csv('new_corpus.csv', header=None).iloc[1:,1].values

#corpus = pd.DataFrame(pd.read_csv('new_corpus.csv')['content']).iloc[:,:].values
df1 = pd.read_excel('banglish total.xlsx').iloc[:,:].values

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
df = np.concatenate((sentiment, labels), axis=1)
#df2 = pd.read_csv('sentiment_corpus.csv').iloc[0:5000,:].values
#df = np.concatenate((df1,df2),axis=0)

#df = pd.read_csv('sentiment_corpus.csv').iloc[0:5000,:].values
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
model_ted = FastText(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)
#model_ted = gensim.models.Word2Vec(tokens, size=Embedding_Dim, window=5, min_count=1, workers=4,sg=1)

model_ted.wv.most_similar('good',topn=15)

model_ted.wv.most_similar('bad',topn=15)

model_ted.wv.most_similar('bhalo',topn=15)

#model_ted2 = FastText.load('saved_model_fastex')


model_ted.wv.most_similar('best',topn=15)

#model_ted2.wv.most_similar('bad',topn=15)

#model_ted2.wv.most_similar('bhalo',topn=15)


model_ted.save('saved_model_fastex_mask_banglish_supervised')
