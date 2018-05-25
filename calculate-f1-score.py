from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.models import load_model
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import os
import re
import gensim.models.word2vec as word2vec
from random import randint

model_name = "300features_10context"
model = word2vec.Word2Vec.load(model_name)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def content_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = nltk.sent_tokenize(review.strip().decode('UTF-8'))
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences += content_to_wordlist( raw_sentence, \
              remove_stopwords )
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


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
        entry = review_to_sentences(row['title'], tokenizer) +review_to_sentences(row['content'], tokenizer)
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
            #print "---------------------------Above 200------------------------------------"
            wordvecArray = wordvecArray[0:200]

        for tag in localtags:
            inputVec = []
            inputVec += wordvecArray
	    for i in range(0, 50):
            	inputVec.append(model.wv[tag])
            finalvec = np.array(inputVec)
            #print "***********************************************************************"
            #print finalvec.shape
            #print finalInputVec.shape
            #finalInputVec = np.append(finalInputVec, np.array(finalvec), axis=0)
            inputWordVec.append(np.array(finalvec))
            output.append(1)

        iter = 0
        ratio = 1
        while iter < len(localtags)*ratio:
            tag = tags[randint(0,len(tags)-1)]
            if tag in localtags:
                #print "inlocaltags"
                continue
            iter += 1
            inputVec = []
            inputVec += wordvecArray
            for i in range(0, 50):
            	inputVec.append(model.wv[tag])
            finalvec = np.array(inputVec)
            #print "##############################################################################################"
            #print finalvec.shape
           # print finalInputVec.shape
            inputWordVec.append(np.array(finalvec))
            output.append(0)
        #print "?????????????????????????????????????????????????????????????????????????????????????????"
        c =c +1
        if c == 1000:
            break
        

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


model_cooking = load_model("CookingModel")
x_test,y_test = getWordVecs("biology.csv")
#scores = model_cooking.evaluate(x_biology, y_biology, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
predictions = model_cooking.predict(x_test)
#print round_array(predictions)
#print y_biology
pred_rounded = round_array(predictions)
print "Biology Accuracy: " + str(accuracy_score(y_test, pred_rounded))
print "Biology F1-Score: " + str(f1_score(y_test, pred_rounded))



x_test,y_test = getWordVecs("crypto.csv")
#scores = model_cooking.evaluate(x_biology, y_biology, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
predictions = model_cooking.predict(x_test)
#print round_array(predictions)
#print y_biology
pred_rounded = round_array(predictions)
print "Crypto Accuracy: " + str(accuracy_score(y_test, pred_rounded))
print "Crypto F1-Score: " + str(f1_score(y_test, pred_rounded))



x_test,y_test = getWordVecs("diy.csv")
#scores = model_cooking.evaluate(x_biology, y_biology, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
predictions = model_cooking.predict(x_test)
#print round_array(predictions)
#print y_biology
pred_rounded = round_array(predictions)
print "DIY Accuracy: " + str(accuracy_score(y_test, pred_rounded))
print "DIY F1-Score: " + str(f1_score(y_test, pred_rounded))

x_test,y_test = getWordVecs("robotics.csv")
#scores = model_cooking.evaluate(x_biology, y_biology, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
predictions = model_cooking.predict(x_test)
#print round_array(predictions)
#print y_biology
pred_rounded = round_array(predictions)
print "Robotics Accuracy: " + str(accuracy_score(y_test, pred_rounded))
print "Robotics F1-Score: " + str(f1_score(y_test, pred_rounded))


x_test,y_test = getWordVecs("travel.csv")
#scores = model_cooking.evaluate(x_biology, y_biology, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
predictions = model_cooking.predict(x_test)
#print round_array(predictions)
#print y_biology
pred_rounded = round_array(predictions)
print "Travel Accuracy: " + str(accuracy_score(y_test, pred_rounded))
print "Travel F1-Score: " + str(f1_score(y_test, pred_rounded))


