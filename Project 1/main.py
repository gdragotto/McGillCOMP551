import json
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.text import TextCollection
from nltk.stem import PorterStemmer
from collections import OrderedDict, defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt
import timeit
import time
import functools


def MSEError(x, y, w):
    return ((y - x.dot(w)).T).dot(y - x.dot(w))

def Predict(x, y, w):
    return x.dot(w)

def linearGradientDescent(x, y, w_init, eta, beta, epsilon, verbose=False, plot=False, update = 1):
    # update{1: inverse time decay, 2: exponential decay, 3: cosine decay}
    w = []
    w.append(w_init)
    i = 0
    cond = True
    if plot:
        e = []
        l = []
    x_t = x.T
    x_fix = x_t.dot(x)
    while cond:
        if update == 1:
            alpha = eta / (1+beta*i)
            option = 'inverse time decay'
        elif update == 2:
            alpha = eta * beta ** (i/1000)
            option = 'exponential decay'
        else :
            alpha =  eta * ((1 - beta) *0.5 * (1 + math.cos(math.pi * i / 1000))+beta)
            option = 'cosine decay'

        w.append(w[-1] - 2 * alpha * (x_fix.dot(w[-1]) - x_t.dot(y)))
        err = np.sum((w[-1] - w[-2]) ** 2)
        if i % 100 == 0 and verbose:
            print("\tEpoch ", i, "\te=", err, "\t alpha=", alpha)
            print("\t\t", w[-1])
        if plot:
            e.append(err)
            l.append(alpha)
        if epsilon > err:
            cond = False
        i = i + 1
    if plot:
        df = pd.DataFrame({'x': range(1, i + 1), 'err': e, 'learn': l})
        plt.yscale('log')
        plt.title(r"Linear GDS learning (eta=" + str('{:.2e}'.format(eta)) + ",beta=" + str(
            '{:.2e}'.format(beta)) + ",eps=" + str('{:.2e}'.format(epsilon))+')'+ option, fontsize=10, color='gray')
        plt.plot('x', 'err', data=df, marker='o', markerfacecolor='#5fdbf4', markersize=5, color='#216ae0',
                 linewidth=0.5)
        plt.plot('x', 'learn', data=df, marker='o', markerfacecolor='#7eea3a', markersize=5, color='#15bf42',
                 linewidth=0.5)
        plt.legend()
    return w


def linearExactSolution(x, y):
    x_t = x.T
    return (np.linalg.inv(x_t.dot(x))).dot((x_t).dot(y))


def cleanData(data, better=False):
    for data_point in data:
        data_point['original_text'] = data_point['text']
        if better:
            data_point['text'] = [word.lower() for word in TweetTokenizer().tokenize(data_point['text'])]
        else:
            data_point['text'] = [word.lower() for word in data_point['text'].split()]
    return data


def mostCommonWords(data, count):
    words = defaultdict()
    for data_point in data:
        for word in data_point['text']:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1

    words = OrderedDict(sorted(words.items(), key=itemgetter(1), reverse=True))
    return {k: words[k] for k in list(words)[:count]}


def writeMostCommon(file, common):
    stream = open(file, "w")
    for key, value in common.items():
        stream.write(key + "\t\t" + str(value) + "\n")
    stream.close()


def prepare_ThreeLabels(data):
    X = []
    Y = []
    for data_point in data:
        X.append([data_point['children'], data_point['controversiality'], int(data_point['is_root']), 1])
        Y.append(data_point['popularity_score'])

    return np.array(X), np.array(Y)


def prepare_CommonWordsLabels(data, common, count):
    X = []
    Y = []
    wordsid = ["" for x in range(count)]
    common = {k: common[k] for k in list(common)[:count]}
    default = np.zeros(count)
    i = 0
    for word in common:
        default[i] = 0
        wordsid[i] = word
        i = i + 1

    for data_point in data:
        occur = default
        x = []
        for word in data_point['text']:
            if word in wordsid:
                occur[wordsid.index(word)] += 1
        x.append(data_point['children'])
        x.append(data_point['controversiality'])
        x.append(int(data_point['is_root']))
        for j in range(count):
            x.append(occur[j])
        x.append(1)
        X.append(x)
        Y.append(data_point['popularity_score'])
        print(X)
    return np.array(X), np.array(Y)


def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score['compound']


def mostCommonWords_Improved(data, count):
    words = defaultdict()
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    for data_point in data:
        for word in data_point['text']:
            if word not in stop_words:
                if (len(word) != 1) or (len(word) == 1 and word in ["!", "?", "*"]):
                    w = ps.stem(word)
                    if w in words:
                        words[w] += 1
                    else:
                        words[w] = 1

    words = OrderedDict(sorted(words.items(), key=itemgetter(1), reverse=True))
    return {k: words[k] for k in list(words)[:count]}


def prepare_Custom(data, common, count, training=True):
    X = []
    Y = []
    wordsid = ["" for x in range(count)]
    common = {k: common[k] for k in list(common)[:count]}
    default = np.zeros(count)
    i = 0
    for word in common:
        default[i] = 0
        wordsid[i] = word
        i = i + 1

    wordWeights = []
    for i in range(count):
        wordWeights.append(1 / (i + 1))

    wordWeights = np.array(wordWeights)

    if (training):
        global Text
        Text = []
        for data_point in data:
            Text.append(data_point['original_text'].lower())

    corpus = TextCollection(Text)

    for data_point in data:
        occur = default
        x = []
        Tf_ide = []
        data_point["num_words"] = len(data_point['text'])
        data_point['sentiment'] = np.abs(nltk_sentiment(data_point['original_text']))
        data_point['exclam'] = data_point['original_text'].count('!')
        data_point['hash'] = data_point['original_text'].count('#')
        popularScore = 0
        for word in data_point['text']:
            if word in wordsid:
                occur[wordsid.index(word)] += 1
                popularScore += 1
        for word in common:
            # tf-idf features
            x.append(corpus.tf_idf(word, data_point['original_text']))

        # popularity-frequency (normalized)
        x.append(popularScore / data_point["num_words"])
        # transform(children) feature
        x.append(np.log(data_point['children'] ** 2 + 1))
        # not new
        x.append(data_point['controversiality'])
        # not new
        x.append(int(data_point['is_root']))
        # compute an index of common words
        x.append(occur.dot(wordWeights.T))
        # sentiment analysis transofmed
        x.append(np.log(np.abs(data_point['sentiment']) ** 2 + 1))
        # count of exclamation points
        x.append(data_point['original_text'].count('!'))
        # count of hashtags
        x.append(data_point['original_text'].count('#'))
        x.append(1)
        X.append(x)
        Y.append(data_point['popularity_score'])

    return np.array(X), np.array(Y)


def normalizeX(x):
    for i in range(len(x[0]) - 1):
        mean = np.mean(x[:, i])
        ex = np.max(x[:, i]) - np.min(x[:, i])
        print(ex)
        for j in range(len(x)):
            x[j, i] = (x[j, i] - mean) / ex
    return np.array(x)


# question flag can either be:
# 0: Most common words
# 1: Compare GDS and Exact for 3 simple text features
# 2: Compare different models
# 3: Test custom model

question = 2
########################################
# LOADING AND SPLITTING THE DATA SET
########################################
with open("proj1_data.json") as fp:
    dataImport = json.load(fp)
if question != 3:
    data = cleanData(dataImport, False)
else:
    data = cleanData(dataImport, False)

trainingSet = data[0:10000]
validationSet = data[10000:11000]
testSet = data[11000:-1]

########################################
# MOST COMMON WORDS
########################################
if question == 0:
    common = mostCommonWords(trainingSet, 160)
    writeMostCommon("common.txt", common)

########################################
# QUESTION1: COMPARISON
########################################
if question == 1:
    x_train, y_train = prepare_ThreeLabels(trainingSet)
    x_val, y_val = prepare_ThreeLabels(validationSet)
    x_test, y_test = prepare_ThreeLabels(testSet)

    print("##### EXACT #####")
    w_linear = linearExactSolution(x_train, y_train)
    timeExact = timeit.Timer(functools.partial(linearExactSolution, x_train, y_train))
    print("\tw:\t", w_linear)
    print("\tAvg. Time:\t", timeExact.timeit(100))
    errExact = MSEError(x_val, y_val, w_linear)
    print("\tMSEError:\t", errExact)

    print("##### GDS #####")
    eta = 0.00002
    #eta = 0.0002
    beta = 0.008
    update = 3
    epsilon = 1 * (10 ** (-10))
    print("\tParameters: eta=" + str(eta), "\tbeta=", str(beta), "\teps=", str(epsilon))
    start_time = time.time()
    w_init = np.zeros(len(x_train[0]))
    w_Three = linearGradientDescent(x_train, y_train, w_init, eta, beta, epsilon, False, True)
    timeThree = time.time() - start_time
    print("\tw:\t", w_Three[-1])
    print("\tTime elapsed:\t", timeThree)
    errThree = MSEError(x_val, y_val, w_Three[-1])
    print("\tMSEError:\t", errThree)
    plt.show()

########################################
# QUESTION2: COMPARISON
########################################

#Models are compared with the close form
#In order to perform the comparison with the GDS use linearGradientDescent(args[])
if question == 2:
    common = mostCommonWords(trainingSet, 160)
    x_train, y_train = prepare_ThreeLabels(trainingSet)
    x_val, y_val = prepare_ThreeLabels(validationSet)
    x_test, y_test = prepare_ThreeLabels(testSet)

    x_train160, y_train160 = prepare_CommonWordsLabels(trainingSet, common, 160)
    x_val160, y_val160 = prepare_CommonWordsLabels(validationSet, common, 160)
    x_test160, y_test160 = prepare_CommonWordsLabels(testSet, common, 160)

    x_train60, y_train60 = prepare_CommonWordsLabels(trainingSet, common, 60)
    x_val60, y_val60 = prepare_CommonWordsLabels(validationSet, common, 60)
    x_test60, y_test60 = prepare_CommonWordsLabels(testSet, common, 60)

    print("##### EXACT w=0 #####")
    w_linear = linearExactSolution(x_train, y_train)
    timeExact = timeit.Timer(functools.partial(linearExactSolution, x_train, y_train))
    # print("\tw:\t", w_linear)
    print("\tAvg. Time:\t", timeExact.timeit(100))
    errExact = MSEError(x_val, y_val, w_linear)
    print("\tMSEError:\t", errExact)

    print("##### EXACT w=60 #####")
    w_linear = linearExactSolution(x_train60, y_train60)
    timeExact = timeit.Timer(functools.partial(linearExactSolution, x_train60, y_train60))
    # print("\tw:\t", w_linear)
    print("\tAvg. Time:\t", timeExact.timeit(100))
    errExact = MSEError(x_val60, y_val60, w_linear)
    print("\tMSEError:\t", errExact)

    print("##### EXACT w=160 #####")
    w_linear = linearExactSolution(x_train160, y_train160)
    timeExact = timeit.Timer(functools.partial(linearExactSolution, x_train160, y_train160))
    # print("\tw:\t", w_linear)
    print("\tAvg. Time:\t", timeExact.timeit(100))
    errExact = MSEError(x_val160, y_val160, w_linear)
    print("\tMSEError:\t", errExact)

########################################
# QUESTION3: NEW FEATURES
########################################

if question == 3:
    commonMaster = mostCommonWords_Improved(trainingSet, 160)
    errors = []

    #In order to compare the model for different values of k,
    #change the variable words=k. To compare, envelop in a for loop
    words = 8
    common = {k: commonMaster[k] for k in list(commonMaster)[:words]}
    x_train, y_train = prepare_Custom(trainingSet, common, words)
    # x_train=normalizeX(x_train)
    x_val, y_val = prepare_Custom(validationSet, common, words)
    # x_val=normalizeX(x_val)
    x_test, y_test = prepare_Custom(testSet, common, words,False)
    # x_test=normalizeX(x_test)
    print("##### NEW FEATURES with CLOSED FORM #####")
    w_linear = linearExactSolution(x_train, y_train)
    timeExact = timeit.Timer(functools.partial(linearExactSolution, x_train, y_train))
    print("\tw:\t", w_linear)
    print("\tAvg. Time:\t", timeExact.timeit(10))
    errExact = MSEError(x_val, y_val, w_linear)
    errExactTest = MSEError(x_test, y_test, w_linear)
    print("\tMSEError Exact on Validation:\t", errExact)
    print("\tMSEError Exact on Test:\t", errExactTest)
