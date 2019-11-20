"""
    This file will contain helper functions used to extract useful pythonic structures
    out of the json training and test data provided by Clickbait Challenge.
"""
import json, os
import util
from stop_words import get_stop_words

# Given a pathname,
# train or val simply tell getRawData to use a pre-assigned path
# Returns instance and truth, both scraped from the json
def getRawData(pathname=None, train=False, val=False):
    if pathname != None:
        instance_path, truth_path = util.getPaths(pathname)
    elif train:
        instance_path, truth_path = util.TRAIN_INSTANCE_PATH, util.TRAIN_TRUTH_PATH
    else:
        instance_path, truth_path = util.VAL_INSTANCE_PATH, util.VAL_TRUTH_PATH
    instance = []
    truth = []
    with open(truth_path, "r") as truth_file:
        for line in truth_file:
            truth.append(json.loads(line))
    with open(instance_path, "r", encoding='utf-8') as instance_file:
        for line in instance_file:
            instance.append(json.loads(line))
    return (instance, truth)

def getTrainTestData(n = 2000):
    # Test data : 0 - 2000
    # validation data : 2000 - 4000
    # train data: train and val 4000+
    instance1, truth1 = getRawData(val=True)
    instance2, truth2 = getRawData(train=True)
    instance, truth = instance1 + instance2, truth1 + truth2
    test_instance, test_truth = instance[n:2*n], truth[n:2*n]
    train_instance, train_truth = instance[2*n:], truth[2*n:]
    return (train_instance, train_truth, test_instance, test_truth)

# This function takes in a list of raw strings
# and outputs a tokenized list of clean words
# clean being lowercase, no stopwords, no punctuation
def getTokenizedWords(text):
    tokens = [word.strip() for word in text]
    words = [word.lower() for word in tokens if word.isalpha()]
    stop_words = get_stop_words('en')
    words = [word for word in words if not word in stop_words]
    return words