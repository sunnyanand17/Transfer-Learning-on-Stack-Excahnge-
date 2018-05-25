import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import os
import re
import gensim.models.word2vec as word2vec
from random import randint

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('punkt')
nltk.download('stopword')

model_name = "wordVec100Features"
model = word2vec.Word2Vec.load(model_name)


#remove html tags
#to get tokens from the content and title
def getwordlist( text, remove_stopwords=False ):
    quesText = BeautifulSoup(text, "lxml").get_text()
    quesText = re.sub("[^a-zA-Z]"," ", quesText)
    words = quesText.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

#to split the content and title into sentences
def getsentences( content,remove_stopwords=False ):
    raw_content = nltk.sent_tokenize(content.strip().decode('UTF-8'))
    sentences = []
    for raw_sentence in raw_content:
        if len(raw_sentence) > 0:
            sentences += getwordlist( raw_sentence, \
              remove_stopwords )
    return sentences

	#loop through all the input files and get the word tokens to build word vector
def getWordVecs(fileName):
    d = pd.read_csv(fileName)

    tags = []   #tags in the document
    for index,value in d['tags'].iteritems():
        for val in value.split(" "):
            tags.append(val)

    inputWordVec = []
    output = []
    c =0

    for index,row in d.iterrows():
        entry = []
        entry = getsentences(row['title']) +getsentences(row['content'])
        wordvecArray = []
        for word in entry:
            wordvecArray.append(model.wv[word])
        localtags = []
        for tag in row['tags'].split(" "):
            localtags.append(tag)
        if len(wordvecArray) < 200:
            diff = 200 - len(wordvecArray)
            for i in range(0,diff):
                wordvecArray.append(np.zeros(100))
        elif len(wordvecArray) > 200:
            wordvecArray = wordvecArray[0:200]

        for tag in localtags:
            inputVec = []
            inputVec += wordvecArray
	    for i in range(0, 50):
            	inputVec.append(model.wv[tag])
            finalvec = np.array(inputVec)
            inputWordVec.append(np.array(finalvec))
            output.append(1)

        iter = 0
        ratio = 1
        while iter < len(localtags)*ratio:
            tag = tags[randint(0,len(tags)-1)]
            if tag in localtags:
                continue
            iter += 1
            inputVec = []
            inputVec += wordvecArray
            for i in range(0, 50):
            	inputVec.append(model.wv[tag])
            finalvec = np.array(inputVec)
            inputWordVec.append(np.array(finalvec))
            output.append(0)
        c =c +1
        if c == 1000:
            break
        print(c)

    y = np.array(output)
    x = np.array(inputWordVec)
    return  x,y

def round_array(num_array):
    for i in range(0, len(num_array)):
	if(num_array[i] >= 0.5):
		num_array[i] = 1
	else:
		num_array[i] = 0
    num_array[i] = (round(num_array[i]))
    return num_array.astype(int)

x,y = getWordVecs("cooking.csv")
x_train = x[0:len(x) - 1000]
x_test = x[len(x) - 1000:]
y_train = y[0:len(y) - 1000]
y_test = y[len(y) - 1000:]

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


#RNN model
numpy.random.seed(7)
model = Sequential()
#model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu', input_shape=(250,100)))
model.add(LSTM(100, input_shape=(250,100), return_sequences='true'))
model.add(LSTM(50))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, nb_epoch=20, batch_size=64)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
from sklearn.metrics import f1_score
predictions = model.predict(x)
model.save("CookingModel")
print f1_score(y, round_array(predictions))

