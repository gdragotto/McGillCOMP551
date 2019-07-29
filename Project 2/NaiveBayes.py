import glob
import nltk
import random
import numpy as np
from nltk.corpus import stopwords
porter = nltk.PorterStemmer()
stoplist = stopwords.words('english')

# load the data
train_pos = list(np.load('train_pos.npy'))
train_neg = list(np.load('train_neg.npy'))


# create vocabulary set
def createVocabList(dataSet,num_pop):
    vocablist = []
    vocabSet = []
    for text in dataSet:
        b = [w.lower() for w in nltk.word_tokenize(text)]
        vocablist.extend([k for k in b if k not in stoplist])
    #c = [porter.stem(x) for x in vocablist]
    all_words = nltk.FreqDist(vocablist)
    vocablist = all_words.most_common(num_pop)
    for j in range(num_pop):
        vocabSet.append(vocablist[j][0])
    return vocabSet

# train bernoulli naive bayes
def trainBernoulliNB(label, D, num_pop):
    wordlist = createVocabList(D,num_pop)
    N = len(D)
    Nc1 = label.count(1)
    Nc2 = N - Nc1
    prior = Nc1 / N
    p_pos = []
    p_neg = []
    for word in wordlist:
        a = 0
        b = 0
        for j in range(N):
            if word in D[j] and label[j] == 1:
                a += 1
            elif word in D[j] and label[j] == 0:
                b += 1
        p_pos.append((a + 1) / (Nc1 + 2))  # laplace smooth
        p_neg.append((b + 1) / (Nc2 + 2))
    return (prior, p_pos, p_neg, wordlist)


# test Bernouolli model
def applyBernoulliNB(D, prior, p_pos, p_neg, wordlist):
    label_test = []
    for text in D:
        Vd = [w.lower() for w in nltk.word_tokenize(text)]
        Vd = list(set(Vd))
        Vd_f = [k for k in Vd if k not in stoplist]
        score_c1 = prior
        score_c2 = 1 - prior
        for t in range(len(wordlist)):
            if wordlist[t] in Vd_f:
                score_c1 = score_c1 * p_pos[t]
                score_c2 = score_c2 * p_neg[t]
            else:
                score_c1 = score_c1 * (1 - p_pos[t])
                score_c2 = score_c2 * (1 - p_neg[t])
        if score_c1 > score_c2:
            label_test.append(1)
        else:
            label_test.append(0)
    return (label_test)


def accuracy(label_test, real_label):
    c = [label_test[i] - real_label[i] for i in range(len(label_test))]
    precision = c.count(0)
    acc = precision / len(c)
    return (acc)


train_x = ([(name, 1) for name in train_pos] + [(name, 0) for name in train_neg])
random.shuffle(train_x)
train_set, validation_set = train_x[5000:], train_x[:5000]
data_train = []
label_train = []
data_validation = []
label_validation = []
for (d, v) in train_set:
    data_train.append(d)
    label_train.append(v)
for (d, v) in validation_set:
    data_validation.append(d)
    label_validation.append(v)

a=[200,250,300,350,400,450,500,550,600,650,700,750]
acc = []
for i in range(12):
    prior, p_pos, p_neg, wordlist = trainBernoulliNB(label_train, data_train,a[i])
    label_test = applyBernoulliNB(data_validation, prior, p_pos, p_neg, wordlist)
    acc.append(accuracy(label_test, label_validation))
print(acc)
np.save('acc',acc)

