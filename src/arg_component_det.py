
import numpy as np
import tensorflow
import tensorflow_addons as tfa
import matplotlib.pyplot as pyplot
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Bidirectional, Input, SpatialDropout1D
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.preprocessing import sequence
import csv 
from sklearn.dummy import DummyClassifier
from sklearn import svm, metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk import word_tokenize, pos_tag, ngrams,classify, bigrams
from nltk.classify import MaxentClassifier, NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from cleantext import clean
import collections
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score

dataset  = pd.read_csv('FINAL_RNN.csv', encoding='ISO-8859-1')

dataset[dataset.columns[0]] = dataset[dataset.columns[0]].str.replace('\d+','')
y = dataset.iloc[:,1]
data = dataset.iloc[:,0]
sample = dataset.iloc[1,0]

# Encoding the labels into acceptable numeric vectors

le = LabelEncoder()
le.fit(["CLAIM", "BACKING","REBUTTAL","REFUTATION", "PREMISE"])

y = le.transform(y)
y = to_categorical(y)



# Tokenizing a piece of raw text, returns a list of tokens.

def tokenize(text):
    tokens = word_tokenize(text)    
    return tokens

#Part-Of-Speech Tagging  for a stream of tokens, returns a list of tuples.

def tagger(tokens):
    tags = pos_tag(tokens)
    return tags

# Pre-processing raw data.

def clean_text(words):
    wnl = nltk.stem.WordNetLemmatizer()
    stwords = stopwords.words('english')
    refined = [wnl.lemmatize(word) for word in words if word not in stwords]
    return refined


data = clean_text(data)


# Creating the GloVe Embeddings Dictionary

embedding_dict = {}
#glove2word2vec('glove.twitter.27B.25d.txt', 'word.txt')
with open('glove.twitter.27B.100d.txt', 'r') as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors
        

glove.close()



# Creating a GloVe Matrix



def Glove_matrix(data):
    matrix = np.zeros( (len(data), 100) )
    n = 0
    for dp in data:
        ab = []
        ab = tokenize(dp)
        y_2 = []
        for b in ab:
            if b in embedding_dict:
                y_2.append((embedding_dict[b]))
            else:
                y_2.append( (np.zeros([100,], dtype = np.float32) ))
        y_3 = [list(i) for i in y_2]   
        y_3 = np.array(y_3)
        col_mean = y_3.mean(axis=0)
        matrix[n] = col_mean
        n= n+1

    return matrix




#Glove Matrix for predictions
def Glove2(dp):
    matrix2 = np.zeros( (len(data), 100) )
    ab = []
    ab = tokenize(dp)
    y_2 = []
    for b in ab:
        if b in embedding_dict:
            y_2.append((embedding_dict[b]))
        else:
            y_2.append( (np.zeros([100,], dtype = np.float32) ))
    y_3 = [list(i) for i in y_2]
    y_3 = np.array(y_3)
    col_mean = y_3.mean(axis=0)
    matrix2[0] = col_mean
    return matrix2


# Converting the training data into the GloVe matrix
matrix =(Glove_matrix(data))
mat_arr = np.array(matrix)
X = matrix

# Reshaping to an appropriate shape for the LSTM model

X = X.reshape(len(X), 1, 100)

# Splitting of data into training and test data.

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, train_size=0.80)

# Defining baseline classifiers to measure the performance of the model

base1 = DummyClassifier(strategy='most_frequent', random_state=0)
base1.fit(X_train, y_train)
base2 = DummyClassifier(strategy='stratified', random_state=0)
base2.fit(X_train, y_train)

print("---------- BASELINE SCORES -------------")
print("BASELINE-1:", base1.score(X_test, y_test))
print("BASELINE-2:", base2.score(X_test, y_test))
print("----------------------------------------")




#The RNN Model 
# Testing the RNN classifier on our test data, and further evaluating on different metrics
# 
# Metrics used: 
# Accuracy - 
# F1_Macro Score - 
# Loss - 
#


model = Sequential()
model.add(Input(shape = (1,100)))
model.add(LSTM(units = 64, activation='relu'))
model.add(Dense(units = 5, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=14 ,validation_data = (X_test, y_test), batch_size = 64)

# Creating a graph to compare the training and validation loss

pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy vs validation accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
scores = model.evaluate(X_test, y_test, verbose = 0)
print("ACCURACY:" , (scores[1]))

y_pred = []
for x  in X_test:
    x = x.reshape(len(x),1,100)
    y_pred.append(np.argmax(model.predict(x)))

y_check =[]
for a in y_test:
    y_check.append(np.argmax(a))


f1 = f1_score(y_check, y_pred, average="weighted")

print("F1 SCORE:", f1)