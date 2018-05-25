import pandas as pd
import numpy as np
import nltk
import os
import re
import gensim.models.word2vec as word2vec
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopword')

#remove html tags
#to get tokens from the content and title
def gentokens(text):
    quesText = BeautifulSoup(text, "lxml").get_text()    #remove html tags
    quesText = re.sub("[^a-zA-Z]"," ", quesText)
    words = quesText.lower().split()
    return(words)

#to split the content and title into sentences
def getsentences(content):
    raw_content = nltk.sent_tokenize(content.strip())
    list_of_tokens = []
    for sentence in raw_content:
        if len(sentence) > 0:
            list_of_tokens.append(gentokens(sentence))
    return list_of_tokens


#loop through all the input files and get the word tokens to build word vector
sentences = []
for root, dirs, files in os.walk("C:/Users/sachi/PycharmProjects/ML"):
    for file in files:
        if file.endswith(".csv"):
            print(file)
            d = pd.read_csv(file)
            for content in d['content']:
                sentences += getsentences(content)
            for title in d['title']:
                sentences += getsentences(title)
            if 'tags' in d:
                for tag in d['tags']:
                    sentences.append(tag.split(' '))

'''
print(len(sentences))
print(sentences[0:10])
'''

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
numFeatures = 100    # Word vector dimensionality
min_word_count = 1
num_workers = 4
context = 10
downsampling = 1e-3

#build the wordvec model
model = word2vec.Word2Vec(sentences, workers=num_workers, size=numFeatures, min_count = min_word_count,window = context, sample = downsampling)

#save the model for further use
modelName = "wordVec100Features"
model.save(modelName)

#to test - load the model and print the word vectors of some of them and
#find similar words
model = word2vec.Word2Vec.load(modelName)
print(model.most_similar("cooking"))
print(model.most_similar("eat"))
print(model.wv['cook'])
print(len(model.wv['cook']))
print(model.wv.vocab["cook"])