import re
import glob
import nltk
import random
import os, os.path
import itertools
import numpy as np
from base import *
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import randint as randint
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict, defaultdict
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class preProcess(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = np.recarray(shape=(len(posts),), dtype=[('text', object),('polarity', object)])
        for i, item in enumerate(posts):
            features['text'][i] = item['original']
            #pol = {}
            #for index,value in enumerate(item['polarity']):
            #    pol['p_'+str(index)]=value
            #features['polarity'][i] = pol
            #features['pos'][i] = item['pos']
        return features


def train(X, Y, posWords, commonWords, classifierIn='LogisticRegression'):
    if (classifierIn == 'LogisticRegression'):
        classifier = LogisticRegression(solver='lbfgs',penalty='l2')
    elif (classifierIn == 'SupportVectorMachine'):
        classifier = LinearSVC(penalty='l2',C=7.25)
    elif (classifierIn == 'DecisionTree'):
        classifier = DecisionTreeClassifier()
    elif (classifierIn == 'neuralNetwork'):
        classifier = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(20, 10, 5), random_state=1)
    elif (classifierIn == 'MultinomialNB'):
        classifier = MultinomialNB()

    print("\tBuilding transformer vocabulary...")
    BoF_pipeline = Pipeline([
        ('selector2', ItemSelector(key='text')),
        ('ngramsVect',CountVectorizer(ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(use_idf=True))
    ])
    polarity_pipeline = Pipeline([
        ('selector3', ItemSelector(key='polarity')),
        ('polarityVect', DictVectorizer()),
        ('tfidf2', TfidfTransformer())
    ])
    pos_pipeline = Pipeline([
        ('selector4', ItemSelector(key='text')),
        ('posVect', CountVectorizer(stop_words='english',vocabulary=list(posWords))),
        ('tfidf3', TfidfTransformer())
    ])
    common_pipeline = Pipeline([
        ('selector5', ItemSelector(key='text')),
        ('commonVect', CountVectorizer(stop_words='english',vocabulary=list(commonWords))),
        ('tfidf4', TfidfTransformer())
    ])
    BoF3_pipeline = Pipeline([
        ('selector32', ItemSelector(key='text')),
        ('ngramsVect3',CountVectorizer(ngram_range=(3,3))),
        ('tfidf33', TfidfTransformer())
    ])
    custom_features = FeatureUnion(
        transformer_list=[
            ('BagOfWords', BoF_pipeline),
            ('BagOfWords3', BoF3_pipeline),
            ('Pos', pos_pipeline),
            ('Common', common_pipeline)
        ],
        transformer_weights={'BagOfWords': 2.0,'BagOfWords3':2.5,'Common':1.0}
    )
    pipeline = Pipeline([
        ('preProcess', preProcess()),
        ('custom_features', custom_features),
        #('feature_selection', TruncatedSVD()),
        ('varianceTh',VarianceThreshold()),
        ('norm', Normalizer()),
        ('classifier', classifier)
    ])
    parameters = {
    #'custom_features__BagOfWords__ngramsVect__stop_words': ['english'],
    'custom_features__BagOfWords__ngramsVect__binary': [True, False],
    #'custom_features__BagOfWords__ngramsVect__ngram_range': ((1, 2),(2, 2)),
    #'custom_features__BagOfWords__tfidf__use_idf': [True],
    #'custom_features__BagOfWords__tfidf__sublinear_tf': [True,False],
    #'custom_features__Pos__tfidf3__use_idf': [True,False],
    #'custom_features__Pos__tfidf3__sublinear_tf': [True,False],
    #'custom_features__Pos__posVect__binary': [True, False],
    #'custom_features__Common__tfidf4__use_idf': [True],
    #'custom_features__Common__tfidf4__sublinear_tf': [True,False],
    'custom_features__Common__commonVect__binary': [True, False],
    #'custom_features__Pos__posVect__max_features': [2000, 5000, 10000, 15000, 25000, 50000],
    #'custom_features__Common__commonVect__max_features': [2000, 5000, 10000, 15000, 25000, 50000],
    #'reduce': [SelectKBest(k='all'),SelectKBest(k=10000),SelectKBest(k=50000)],
    'custom_features__transformer_weights': [
        {'BagOfWords':2.0,'BagOfWords3':1.0,'Common':1.5#, 'Polarity': 0.2
        },
        {'BagOfWords':2.1,'BagOfWords3':2.7,'Common':1.6#, 'Polarity': 0.2
        },
        {'BagOfWords':2.5,'BagOfWords3':1.4,'Common':1.1#, 'Polarity': 0.2
        },
        {'BagOfWords':2.5,'BagOfWords3':2.5,'Common':1.0#, 'Polarity': 0.2
        },
        {'BagOfWords':2.5,'BagOfWords3':2.6,'Common':1.0#, 'Polarity': 0.2
        }
        ]
    }

    if (classifierIn == 'SupportVectorMachine'):
        parameters.update({'classifier__C' : [7.15]})
    elif (classifierIn == 'LogisticRegression'):
        parameters.update({'classifier__C' : [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7, 7.1, 7.25, 7.4, 7.5, 8.0, 8.5 ] })
    elif (classifierIn == 'neuralNetwork'):
        parameters.update({'classifier__C' : [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7, 7.1, 7.25, 7.4, 7.5, 8.0, 8.5 ] })
    print(parameters)
    pSearch=pipeline
    scores = cross_val_score(pSearch, X, Y, cv=5)
    print(scores)
    #pSearch = RandomizedSearchCV(pipeline, parameters, cv=3,iid=False, n_jobs=-1,n_iter=10, verbose=20)
    print("\tPipelining and fitting...")
    pSearch.fit(X, y=Y)
    #report(pSearch.cv_results_)
    return pSearch

baseDir = "data/"
train_pos = list(np.load(baseDir + 'Pos.npy'))
train_neg = list(np.load(baseDir + 'Neg.npy'))
posWords = list(np.load(baseDir + 'posWords.npy'))
commonWords = list(np.load(baseDir + 'commonWords.npy'))
Vocabulary = list(np.load(baseDir + 'Vocabulary.npy'))
tSet = ([(item, 1) for item in train_pos] + [(item, 0) for item in train_neg])
#tSet = tSet[:10]
random.shuffle(tSet);
X = []
Y = []
for (d, v) in tSet:
    X.append(d)
    Y.append(v)
print("Dataset is loaded")

#classifiers = {'LogisticRegression','SupportVectorMachine','DecisionTree','neuralNetwork','MultinomialNB'}
classifiers = ['MultinomialNB','LogisticRegression','SupportVectorMachine','DecisionTree']
for classifier in classifiers:
    print("-----------------------------")
    print("######"+classifier+"######")
    estimator = train(X, Y, posWords, commonWords, classifier)
    #print("Estimated. Interpolating...")
    #output = open("predictions_"+classifier+".csv", "w")
    #output.write("Id,Category" + "\n")
    #testSet = list(np.load(baseDir + 'Test.npy'))
    #for item in testSet:
    #    pred = estimator.predict([item])
    #    output.write(str(item['id']) + "," + str(pred[0]) + "\n")
    #output.close()
    print("-----------------------------")
