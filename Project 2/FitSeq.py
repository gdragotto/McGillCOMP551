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
from sklearn.ensemble import VotingClassifier
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict, defaultdict
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import LinearSVC,SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator


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
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

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
        features = np.recarray(shape=(len(posts),), dtype=[('original', object),('tokenized', object),('text', object),('word2vect', object)])
        for i, item in enumerate(posts):
            features['text'][i] = item['processed']
            features['original'][i] = item['original']
            features['tokenized'][i] = item['tokenized']
            #features['pos'][i] = item['pos']
            featVect = {}
            for index,value in enumerate(item['vect']):
                featVect['ve'+str(index)]=value
            features['word2vect'][i] = featVect
            #print(features['word2vect'][i])
        return features


def train(X, Y, posWords, commonWords):

    print("\tBuilding transformer vocabulary...")
    BoF_pipeline = Pipeline([
        ('selector2', ItemSelector(key='original')),
        ('ngramsVect',CountVectorizer(ngram_range=(3,3))),
        ('tfidf', TfidfTransformer(use_idf=True))
    ])
    BoF3_pipeline = Pipeline([
        ('selector32', ItemSelector(key='text')),
        ('ngramsVect3',CountVectorizer(ngram_range=(3,3))),
        ('tfidf33', TfidfTransformer(use_idf=True))
    ])
    common_pipeline = Pipeline([
        ('selector5', ItemSelector(key='original')),
        ('commonVect', CountVectorizer(stop_words='english',ngram_range=(1,1))),
        ('tfidf4', TfidfTransformer(use_idf=True))
    ])
    vect_pipeline = Pipeline([
            ('selector3', ItemSelector(key='word2vect')),
            ('vectVect', DictVectorizer()),
    ])
    custom_features = FeatureUnion(
        transformer_list=[
            ('BagOfWords', BoF_pipeline),
            #('BagOfWords3', BoF3_pipeline),
            ('Common', common_pipeline),
            ('AvgVect', vect_pipeline),
        ]
        ,transformer_weights={'BagOfWords':1.5,'Common':1.75,'AvgVect':0.7}

    )
    np.random.seed(123)
    a = LinearSVC(penalty='l2',C=7.25)
    b = LogisticRegression(solver='lbfgs',penalty='l2',C=5.05,n_jobs=-1,max_iter=200)
    c = DecisionTreeClassifier(random_state=0)
    pipeline = Pipeline([
        ('preProcess', preProcess()),
        ('custom_features', custom_features),
        ('varianceTh',VarianceThreshold()),
        ('norm', Normalizer()),
        #('classifier', VotingClassifier(estimators=[('a', a), ('b', b), ('c', c)], n_jobs=-1,voting='hard',weights=[2, 1, 1]))
        #('classifier', LinearSVC(penalty='l2',C=7.27))ù
         ('to_dense', DenseTransformer())
        ('classifier', LinearDiscriminantAnalysis())
    ])
    parameters = {

    }
    print("\tPipelining and fitting...")
    pSearch = GridSearchCV(pipeline, parameters, cv=3, iid=False, n_jobs=-1, verbose=50)
    pSearch.fit(X, y=Y)
    report(pSearch.cv_results_)
    return pSearch

baseDir = "data/"
train_pos = list(np.load(baseDir + 'Pos.npy'))
train_neg = list(np.load(baseDir + 'Neg.npy'))
posWords = list(np.load(baseDir + 'posWords.npy'))
commonWords = list(np.load(baseDir + 'commonWords.npy'))
Vocabulary = list(np.load(baseDir + 'Vocabulary.npy'))
tSet = ([(item, 1) for item in train_pos] + [(item, 0) for item in train_neg])
random.shuffle(tSet);
#tSet = tSet[:100]
X = []
Y = []
for (d, v) in tSet:
    X.append(d)
    Y.append(v)
print("Dataset is loaded")

print("-----------------------------")
estimator = train(X, Y, posWords, commonWords)
print("Estimated. Interpolating...")
output = open("predictions_multi.csv", "w")
output.write("Id,Category" + "\n")
testSet = list(np.load(baseDir + 'Test.npy'))
for item in testSet:
    pred = estimator.predict([item])
    output.write(str(item['id']) + "," + str(pred[0]) + "\n")
output.close()
print("-----------------------------")
