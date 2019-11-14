"""
    This file will contain helper functions used to extract useful pythonic structures
    out of the json training and test data provided by Clickbait Challenge.
"""
import json, os


# Given a pathname,
# Returns instance and truth, both scraped from the json
def getRawData(pathname):
    instance_path = os.path.join('.', pathname, "instances.jsonl")
    truth_path = os.path.join('.', pathname, "truth.jsonl")
    instance = []
    truth = []
    with open(truth_path, "r") as truth_file:
        for line in truth_file:
            truth.append(json.loads(line))
    with open(instance_path, "r", encoding='utf-8') as instance_file:
        for line in instance_file:
            instance.append(json.loads(line))
    # print('BABAGANOUSH') Wonder if Johannes will find this...
    return (instance, truth)