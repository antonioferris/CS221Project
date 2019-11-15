"""
    This file will contain the various models that we will employ, along with helper
    functions to test these models.
"""
import data, main
import eval, subprocess
import util
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

print (set(stopwords.words('english')))
# This classifier will say something is clickbait if it has ! or ? in the title
def dumbClassifier():
    return lambda inst : 1 if set(inst["targetTitle"]) & set("?!") else 0

# Everything is not clickbait classifier
def dumberClassifier():
    return lambda inst : 0

# test tne classifier (function instance -> score)
# with eval.py
def testClassifier(func, name='untitled.testoutput'):
    results = dict()
    instance, truth = data.getRawData(val=True)
    for inst in instance:
        _id = inst["id"]
        results[_id] = str(func(inst))
    util.dumpResults(results, '.tmpdmp')
    subprocess.run(["python", "eval.py", util.VAL_TRUTH_PATH, '.tmpdmp', name])

def classifier(inst):
    title = inst["targetTitle"]
    description = inst["targetDescription"]
    keywords = inst["targetKeywords"]
    paragraphs = inst["targetParagraphs"]

    #Feature: Title punctuation

    #Feature: Average word length

    #Feature: Paragraph length

    #Feature: Number of paragraphs

    #Feature: Number of keywords

    #Feature: Number Named Entities




