# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:51:41 2020

@author: Liz
"""
import os
import sklearn 
from sklearn.naive_bayes import MultinomialNB # Import the Naive Bayes Model
from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split 
# This is an essential package to count the words and label them. 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # Stop Words
import nltk # Important to deal with natural language (English)

os.chdir("C:\\Users\\labbi\\OneDrive\\Liz\\Brandeis\\Python\\sentiments analysis\\aclImdb")

#stopwords
#nltk.download('stopwords') 
#nltk.download('punkt')
stop_words = stopwords.words('english') 

trainPath = "train"
testPath = "test"

train = load_files(trainPath)
test = load_files(testPath)

reviews = []
labels = []

for review in train.data:
    reviews.append(review)
    labels.append(train.target[train.data.index(review)])

for review in test.data:
    reviews.append(review)
    labels.append(test.target[test.data.index(review)])
    
# This creates the vectorizer, the object that splits the reviews into words. 
splitParagraphs = CountVectorizer(tokenizer=nltk.word_tokenize,stop_words=stop_words)

# This splits the paragraph into words (tokenizes them), 
#   and count them according to their label
counts=splitParagraphs.fit_transform(reviews)
type(counts)
print(counts)
# Splits the data into training and testing
moviesTrain, moviesTest, labelTrain, labelTest = train_test_split(
        counts, labels, test_size = 0.3, random_state = 814)
clf = MultinomialNB().fit(moviesTrain, labelTrain)
sklearn.metrics.accuracy_score(labelTest,clf.predict(moviesTest))


