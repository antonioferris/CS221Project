"""
    This file contains utilility functions we might want to use
    across multiple files
"""
import random, sys, os

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

# Takes in a dict results to pathname
# results is a dict id -> clickbaitScore
def dumpResults(results, pathname=None):
    if pathname == None:
        for (_id, clickbaitScore) in results.items():
            print(getOutputString(_id, clickbaitScore), end='')
    else:
        with open(pathname, "w") as outfile:
            for (_id, clickbaitScore) in results.items():
                outfile.write(getOutputString(_id, clickbaitScore))

def testDump():
    results = {str(i) : random.random() for i in range(10)}
    dumpResults(results)
    print('test dump done')
