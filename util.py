"""
    This file contains utilility functions we might want to use
    across multiple files
"""
import random, sys, os, gensim
from sklearn import metrics

labels = ['id', 'postMedia', 'postText', 'targetCaptions', 'targetParagraphs','targetTitle', 
'postTimestamp', 'targetKeywords', 'targetDescription', 
'truthJudgments', 'truthMean', 'truthClass', 'truthMedian', 'truthMode']
label_dict = {labels[i]: i for i in range(len(labels))}



# This function will return the
# paths to train.josnl and instance.jsonl
def getPaths(pathname):
    instance_path = os.path.join('.', pathname, "instances.jsonl")
    truth_path = os.path.join('.', pathname, "truth.jsonl")
    return instance_path, truth_path

TRAIN_INSTANCE_PATH, TRAIN_TRUTH_PATH = getPaths('clickbait17-train-170331')
VAL_INSTANCE_PATH, VAL_TRUTH_PATH = getPaths('clickbait17-validation-170630')

def getOutputString(_id, clickbaitScore):
    return '{{ "id" : "{}", "clickbaitScore" : {} }}\n'.format(_id, clickbaitScore)

def generateModel():
    print('Generating Model.  Please wait 90s')
    return gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 

# Takes in a dict results to pathname
# results is a dict id -> clickbaitScore
def dumpResults(results, pathname=None):
    if pathname == None:
        for (_id, clickbaitScore) in results.items():
            print(getOutputString(_id, clickbaitScore), end='')
    else:
        with open(pathname, "w") as outfile:
            for (_id, clickbaitScore) in results.items():
                s = getOutputString(_id, clickbaitScore)
                # if random.random() < 0.1:
                #     print(s)
                outfile.write(s)

def testDump():
    results = {str(i) : random.random() for i in range(10)}
    dumpResults(results)
    print('test dump done')

def getInstMean(datum):
    return datum[:9], datum[label_dict['truthMean']]

def testClassifier(func, test_data):
    n = len(test_data)
    pred = []
    y = []
    for i in range(n):
        inst, truth = getInstMean(test_data[i])
        a, b = func(inst), truth
        a, b = round(a), round(b)
        assert(isinstance(a, int))
        assert(isinstance(b, int))
        assert(0 <= a <= 1)
        assert(0 <= b <= 1)
        pred.append(a)
        y.append(b)
    y = list(map(round, y))
    pred = list(map(round, pred))
    num_clickbait = sum(pred)
    print('{} clickbait | {} not clickbait | {} test datapoints'.format(num_clickbait, n - num_clickbait, n))
    print(metrics.classification_report(y, pred))
