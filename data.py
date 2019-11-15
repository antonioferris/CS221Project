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

# This function takes in a list of raw strings
# and outputs a tokenized list of clean words
# clean being lowercase, no stopwords, no punctuation
def getTokenizedWords(text):
    tokens = [word.strip() for word in text]
    words = [word.lower() for word in tokens if word.isalpha()]
    stop_words = get_stop_words('en')
    words = [word for word in words if not word in stop_words]
    return words

instance, truth = getRawData(train=True)
article_one = instance[0]["postText"]
print(article_one)
print(getTokenizedWords(article_one))