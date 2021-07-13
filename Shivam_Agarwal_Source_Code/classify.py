import nltk
from nltk import word_tokenize, pos_tag, ngrams,classify, bigrams
from nltk.classify import MaxentClassifier, NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from cleantext import clean
import collections
import csv 
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Extracting the training dataset and storing as a pandas dataframe


dataset = pd.read_csv('training1.csv', encoding='ISO-8859-1')
dataset = dataset[["TEXT", "LABEL"]]

# Testing the model on the Twitter data 
test_df = pd.read_csv('labelled_twitter1.csv', encoding = 'ISO-8859-1')
test_df = test_df.append(pd.read_csv('labelled_twitter2.csv', encoding = 'ISO-8859-1'))
test_df = test_df.astype({'LABEL': 'bool'})
test_df = test_df.dropna()
print("test-df:", test_df.head(10))

# Extracting the text from the pandas dataframe

data= dataset.iloc[:,0] 

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


# POS Tagging of the dataset using the functions defined above.

data_pos = []
for dp in data:
    tags = tagger(tokenize(dp))
    n = 1
    y_2 = [b[n] for b in tags]
    data_pos.append(' '.join(y_2))


#Implementation of the TF-IDF Model, using unigrams and bigrams as features.

matrix2 = TfidfVectorizer( ngram_range=(1,2), lowercase=True)
X = matrix2.fit_transform(data).toarray()


# Display the vocabulary created by the model.
#print("VOCAB:", matrix2.vocabulary_)


# Splitting of the dataset into suitable sizes for training and testing phase.

y= dataset.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, train_size=0.80)


# Determining baseline score using a Dummy Classifier
base1 = DummyClassifier(strategy='most_frequent', random_state=0)
base1.fit(X_train, y_train)
base2 = DummyClassifier(strategy='stratified', random_state=0)
base2.fit(X_train, y_train)

print("---------- BASELINE SCORES -------------")
print("BASELINE-1:", base1.score(X_test, y_test))
print("BASELINE-2:", base2.score(X_test, y_test))
print("----------------------------------------")

# Training a Support Vector Machine(SVM) for classification

sv = svm.SVC()
sv.fit(X_train,y_train)


# Testing the SVM classifier on our test data, and further evaluating on different metrics
# 
# Metrics used: 
# Accuracy - 
# F1_Macro Score - 
# 5-Cross Validation -
#
#


y_pred = []
for x in X_test:
   y_pred.append(sv.predict(x.reshape(1,-1)))

accuracy = accuracy_score(y_test, y_pred)

scores = cross_val_score(sv, X, y, cv = 5, scoring='f1_macro')
sum = 0
for s in scores:
    sum += s
avg_score = sum/5
print("-----SVM--------------")
print('10-CROSS VALIDATION SCORE: ', avg_score)
print('F1_MACRO: ', f1_score(y_test, y_pred, average='macro'))
print("ACCURACY: ", accuracy)
print("-----------------------")
print("")
print("")



### Training an ensemble classifier of an SVM, a Random Forest Classifier a Naive Bayes Model to compare results

print("-----ENSEMBLE CLASSIFIER------")
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
sv2 = svm.SVC()
eclf = VotingClassifier(estimators=[('svm',sv2), ('gnb',clf3), ('rf', clf2)], voting = 'hard')
eclf.fit(X_train, y_train)


y_pred_ensemble = []
for x in X_test:
   y_pred_ensemble.append(eclf.predict(x.reshape(1,-1)))

print("ACCURACY:",eclf.score(X_test, y_test))
print('F1_MACRO: ', f1_score(y_test, y_pred_ensemble, average='macro'))
print("--------------------------------")

#Test on Twitter Data

t_data= test_df.iloc[:,0] 
t_data = clean_text(t_data)
t_X = matrix2.transform(t_data).toarray()
t_y= test_df.iloc[:,1]

# Baseline results for twitter data

print("---------- BASELINE SCORES -------------")
print("BASELINE-1:", base1.score(t_X, t_y))
print("BASELINE-2:", base2.score(t_X, t_y))
print("----------------------------------------")

ty_pred = []
for x in t_X:
   ty_pred.append(sv.predict(x.reshape(1,-1)))

accuracy = accuracy_score(t_y, ty_pred)

scores = cross_val_score(sv, t_X, t_y, cv = 5, scoring='f1_macro')
sum = 0
for s in scores:
    sum += s
avg_score = sum/5
print("-----TWITTER SVM------")
print('10-CROSS VALIDATION SCORE: ', avg_score)
print('F1_MACRO: ', f1_score(t_y, ty_pred, average='macro'))
print("ACCURACY: ", accuracy)

