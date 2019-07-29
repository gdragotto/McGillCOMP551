import glob
import nltk
import os
import random
import numpy as np
from base import *
import gensim
from nltk.stem import WordNetLemmatizer

def createVocabList(dataSet, num_pop, num_pos, path):
    stoplist = stopwords.words('english')
    wnl = WordNetLemmatizer()
    vocablist = []
    vocabSet = []
    PosSet = []
    Poslist = []
    for item in dataSet:
        b = item['tokenized']
        vocablist.extend([k for k in b if (k.isalpha())])
        wordlist_adj, wordlist_adv, wordlist_n, wordlist_vb = item['pos']
        Poslist.extend(wordlist_adj + wordlist_adv + wordlist_n + wordlist_vb)

    all_words = nltk.FreqDist(vocablist)
    common_vocablist = all_words.most_common(num_pop)
    all_poswords = nltk.FreqDist(Poslist)
    common_Poslist = all_poswords.most_common(num_pos)
    for j in range(min(num_pop, len(common_vocablist))):
        vocabSet.append(common_vocablist[j][0])
    for k in range(min(num_pos, len(common_Poslist))):
        PosSet.append(common_Poslist[k][0])
    a = vocabSet + PosSet
    np.save(path+'commonWords.npy',list(set(vocabSet)))
    np.save(path+'posWords.npy',list(set(PosSet)))
    np.save(path+'Vocabulary.npy',list(set(a)))
    total_words = list(set(a))
    print("\tVocabulary has size:"+str(len(a)))
    print("\t\tCommon has size:"+str(len(vocabSet)))
    print("\t\tPos has size:"+str(len(PosSet)))
    return total_words


def loadData(directory):
    dataDir = os.listdir(directory)
    wnl = WordNetLemmatizer()
    dataDir = [f for f in dataDir if not f.startswith('.')]
    dataDir.sort(key=lambda x: int(x.split('.')[0]))
    text_set = []
    i = 0
    for filename in dataDir:
        full_path = os.path.join(directory, filename)
        myfile = open(full_path, 'r', encoding='utf-8')
        text= cleanHtmlTags(myfile.read()).lower()
        stoplist = stopwords.words('english')
        tokenized = [wnl.lemmatize(w.lower()) for w in nltk.word_tokenize(text) if (w not in stoplist and w.isalpha())]
        record = {'tokenized':tokenized,'original':text,'processed':" ".join(tokenized),'id':filename.replace(".txt",""),'pos':pos_wordlist(tokenized)}
        #'polarity':getPolarityScore(text)
        text_set.append(record)
        if (i % 1000 == 0):
            print("\tProcessing "+str(i)+"...")
        i+=1
        myfile.close()
    return text_set

def trainWord2Vect(text_set,path):
    #From https://www.kaggle.com/c/word2vec-nlp-tutorial#part-4-comparing-deep-and-non-deep-learning-methods
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    sentences = [x["processed"] for x in text_set]
    model = gensim.models.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)
    model.init_sims(replace=True)
    model.save(path+"Word2VectModel")
    return model

def postProcessSet(set,name,nfeat,model,index2word_set):
    print("\tProcessing set "+name)
    newset = []
    j=0
    for item in set:
        if (j % 1000 == 0):
            print("\t\tProcessing "+str(j)+"...")
        record={}
        j+=1
        featureVec = np.zeros((nfeat,),dtype="float32")
        nwords = 0.
        words = item["tokenized"]
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])

        featureVec = np.divide(featureVec,nfeat)
        record=item
        record["vect"] = featureVec
        newset.append(record)
    print("\tSet"+name+" has size:"+str(len(newset)))
    return newset


# load the data
savepath = "data/"
print("Loading testSet")
testSet = loadData(savepath+"test/")
print("Loading trainSet(1)")
posSet = loadData(savepath+"train/pos/")
print("Loading trainSet(2)")
negSet = loadData(savepath+"train/neg/")

trainSet = posSet + negSet
print("Creating vocabulary")
createVocabList(trainSet, 80000, 80000, savepath)

print("Training word2vect")
model = trainWord2Vect(trainSet,savepath)
index2word_set = set(model.wv.index2word)
nfeat = 300

print("Overriding vord2vect for sets...")
P_trainNeg = postProcessSet(negSet,"Neg",nfeat,model,index2word_set)
P_trainPos = postProcessSet(posSet,"Pos",nfeat,model,index2word_set)
P_testSet = postProcessSet(testSet,"Test",nfeat,model,index2word_set)
np.save(savepath+'Pos.npy',P_trainPos)
np.save(savepath+'Neg.npy',P_trainNeg)
np.save(savepath+'Test.npy',P_testSet)
