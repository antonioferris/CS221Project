import data
import util
import numpy as np
import random
import collections
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from stop_words import get_stop_words
import string
from util import label_dict
import model

class MultiClassModel(model.Classifier):
    def __init__(self, train_data):
        regr = linear_model.LogisticRegression()
        # model = util.generateModel()
        model = None
        X = []
        y = []
        n = len(train_data)
        for i in range(n):
            inst, truth = util.getInstMean(train_data[i])
            if truth < 1 / 6:
                target = 0
            elif truth < 1 / 2:
                target = 1
            elif truth < 5 / 6:
                target = 2
            else:
                target = 3
            X.append(self.featureExtractorX(inst, model))
            y.append(target)
        X = np.asarray(X)
        print(collections.Counter(y).most_common())
        y = np.asarray(y).reshape(-1, 1)
        regr.fit(X, y)
        def func(inst):
            # if random.random() < 0.005:
            #     print(self.featureExtractorX(inst, model).reshape(1,-1))
            return  float(max(min(regr.predict(self.featureExtractorX(inst, model).reshape(1,-1)), 1), 0))
        #return lambda inst : max(min(regr.predict(featureExtractorX(inst, model).reshape(1,300)), 1), 0)
        self.f = func

    def getFunc(self):
        return self.f

    def featureExtractorX(self, inst, model):
        title = inst[label_dict["targetTitle"]]
        description = inst[label_dict["targetDescription"]]
        keywords = inst[label_dict["targetKeywords"]]
        paragraphs = inst[label_dict["targetParagraphs"]]
        captions = inst[label_dict["targetCaptions"]]

        #Process text
        def processText(text):
            tokens = [word.strip() for word in text]
            words = [word.lower() for word in tokens if word.isalpha()]
            _stopwords = get_stop_words('en')
            words = [word for word in words if not word in _stopwords]
            # tokens = word_tokenize(text)
            # lower = [word.lower() for word in tokens if word.isalpha()]
            # stop_tokens = [word for word in lower if word not in _stopwords]
            return words
        def countPunc(text):
            total_count = 0
            punct_count = collections.defaultdict(int)
            for word in title:
                for c in word:
                    if c in string.punctuation:
                        punct_count[c] += 1
                        total_count += 1
            return punct_count, total_count
        
        #Feature: Title punctuation count
        title_punc, title_punc_count = countPunc(title)
        #Feature: count of !
        title_exclam_count = title_punc["!"]
        #Feature: count of ?
        title_question_count = title_punc['?']
        v2 = np.asarray([title_punc_count, title_exclam_count, title_question_count])
        #return np.concatenate((v1, v2), axis = None)
        return v2