"""
    This file will contain the various models that we will employ, along with helper
    functions to test these models.
"""
import data
import util
import numpy as np
import random
# import nltk
# nltk.download('stopwords')
import collections
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from stop_words import get_stop_words
import string
from util import label_dict

# This is the General Classifier model
# You simply need to return a function that classifies, and then we can test you!
class Classifier():
    # gets the classifier's function
    def getFunc(self):
        pass

    def test(self, test_data):
        util.testClassifier(self.getFunc(), test_data)

# This classifier will say something is clickbait if it has ! or ? in the title
class DumbClassifier(Classifier):
    def getFunc(self):
        return lambda inst : 1 if set(inst[label_dict["targetTitle"]]) & set("?!") else 0



