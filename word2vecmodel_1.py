import data
import util
import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from stop_words import get_stop_words
import string
from util import label_dict
import model
import gensim

class Word2VecModel(model.Classifier):
    # Most of our classifiers should have an __init__ just like this one!
    # init youself by training on the training data and then capture your function
    # with a self.f = func to be returned later in self.getFunc()
    # Also, notice that featureExtractorX is a class function now.
    def __init__(self, train_data):
        regr = linear_model.LinearRegression()
        # model = util.generateModel()
        model = gensim.models.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary = True)
        X = []
        y = []
        n = len(train_data)
        for i in range(n):
            inst, truth = util.getInstMean(train_data[i])
            X.append(self.featureExtractorX(inst, model))
            y.append(truth)
        X = np.asarray(X)
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

    def processText(self, text):
        tokens = [word.strip() for word in text]
        words = [word.lower() for word in tokens if word.isalpha()]
        return words

    def featureExtractorX(self, inst, model):
        title = inst[label_dict["targetTitle"]]
        description = inst[label_dict["targetDescription"]]
        keywords = inst[label_dict["targetKeywords"]]
        paragraphs = inst[label_dict["targetParagraphs"]]
        captions = inst[label_dict["targetCaptions"]]

        #Things to go into Word2Vec: processed_title, processed_keywords, title_proper_nouns, keywords_proper_nouns, captions
        processed_title = self.processText(title)
        title_vec = convertToWordVector(title, model)
        #text_vec = 
        #keyword_vec = convertToWordVector(keywords, model)
        #return np.concatenate((title_vec, keyword_vec), axis = None)
        return title_vec

    def convertToWordVector(self, input, model):
        vectors = [model[word] for word in input if word in model.vocab]
    #         percent_vocab = len(vectors) / len(input)
        return np.mean(vectors, axis=0)
    #     else:
    #         return model[input]