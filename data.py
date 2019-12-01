"""
    This file will contain helper functions used to extract useful pythonic structures
    out of the json training and test data provided by Clickbait Challenge.
"""
import json, os
import util
from stop_words import get_stop_words
import pickle

def getTrainValTest():
    raw_train = getPickledData('train_data.pickle')
    raw_val = getPickledData('val_data.pickle')
    total_data = raw_train + raw_val
    n = len(total_data)
    train_sz = 4 * n // 5
    test_val_sz = (n - train_sz) // 2
    train_data = total_data[:train_sz]
    cross_val_data = total_data[train_sz:train_sz + test_val_sz]
    test_data = total_data[train_sz + test_val_sz:]
    return (train_data, cross_val_data, test_data)

def getPickledData(pathname):
    with open(pathname, 'rb') as f:
        D = pickle.load(f)
    return D

def pickleData():
    train_truth_path = 'clickbait17-train-170331\\truth.jsonl'
    train_inst_path = 'clickbait17-train-170331\instances.jsonl'
    val_truth_path = 'clickbait17-train-170331\\truth.jsonl'
    val_inst_path = 'clickbait17-train-170331\instances.jsonl'
    raw_tr_inst = []
    raw_tr_truth = []
    raw_val_inst = []
    raw_val_truth = []
    with open(train_inst_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            raw_tr_inst.append(json.loads(line))
    with open(train_truth_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            raw_tr_truth.append(json.loads(line))
    with open(val_inst_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            raw_val_inst.append(json.loads(line))
    with open(val_truth_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            raw_val_truth.append(json.loads(line))
    n1 = len(raw_tr_inst)
    assert(len(raw_tr_inst) == len(raw_tr_truth))
    n2 = len(raw_val_inst)
    assert(len(raw_val_inst) == len(raw_val_truth))

    to_pickle_train = []
    to_pickle_val = []
    for i in range(n1):
        row = []
        inst = raw_tr_inst[i]
        for j in range(n1):
            if raw_tr_inst[i]["id"] == raw_tr_truth[j]["id"]:
                truth = raw_tr_truth[j]
                break
        for key in util.labels:
            if key in inst:
                row.append(inst[key])
            else:
                row.append(truth[key])
        to_pickle_train.append(row)
    s1 = set()
    for i in range(n1):
        assert(len(to_pickle_train[i]) == len(util.labels))
        s1.add(to_pickle_train[i][util.label_dict["id"]])
    assert(len(s1) == n1)

    for i in range(n2):
        row = []
        inst = raw_val_inst[i]
        for j in range(n2):
            if raw_val_inst[i]["id"] == raw_val_truth[j]["id"]:
                truth = raw_val_truth[j]
                break
        for key in util.labels:
            if key in inst:
                row.append(inst[key])
            else:
                row.append(truth[key])
        to_pickle_val.append(row)
    s2 = set()
    for i in range(n2):
        assert(len(to_pickle_val[i]) == len(util.labels))
        s2.add(to_pickle_val[i][util.label_dict["id"]])
    assert(len(s2) == n2)

    pickle.dump(to_pickle_train, open('train_data.pickle', 'wb'))
    pickle.dump(to_pickle_val, open('val_data.pickle', 'wb'))
        


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