import re
import glob
import nltk
import random
import os, os.path
import itertools
import numpy as np
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict, defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def cleanHtmlTags(sentence):
    return re.sub(re.compile('<.*?>'), '', sentence)

def pos_wordlist(text):
    postag = nltk.pos_tag(text)
    wordlist_adj = []
    wordlist_adv = []
    wordlist_n = []
    wordlist_vb = []
    for (key,value) in postag:
        if (value in ["JJ",'JJR','JJS'] and value not in wordlist_adj):
            wordlist_adj.append(key)
        elif (value in ['RB','RBR','RBS'] and value not in wordlist_adv):
            wordlist_adv.append(key)
        elif (value in ["NN",'NNS'] and value not in wordlist_n):
            wordlist_n.append(key)
        elif (value in ['VB','VBZ','VBD','VBN','VBP','VBG'] and value not in wordlist_vb):
            wordlist_vb.append(key)
    return (wordlist_adj, wordlist_adv, wordlist_n, wordlist_vb)


def getPolarityScore(sentence):
    # Resulting array result[] has 3 elements
    # first one represents negative (1 true,0 false)
    # second one is neutral
    # third one is positive
    r = SentimentIntensityAnalyzer().polarity_scores(sentence)
    result = np.zeros(3)
    if r["compound"] == 0.0:
        result[1] = 1
    elif r["compound"] > 0.0:
        result[2] = 1
    else:
        result[0] = 1
    return result


# From SciKit-Learn
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plotConfusion(y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, ["(+)", "(-)"], normalize=True, title='Normalized confusion matrix')
    plt.show()


def accuracy(label_test, real_label):
    c = [label_test[i] - real_label[i] for i in range(len(label_test))]
    precision = c.count(0)
    acc = precision / len(c)
    return (acc)
