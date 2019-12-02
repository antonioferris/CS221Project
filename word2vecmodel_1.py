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
        self.id_set = set()
        regr = linear_model.LinearRegression()
        #model = None
        model = gensim.models.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary = True)
        X = []
        y = []
        n = len(train_data)
        for i in range(n):
            inst, truth = util.getInstMean(train_data[i])
            #print(i, end=' ')
            X.append(self.featureExtractorX(inst, model))
            y.append(truth)
        print(len(X))
        print(X[0].shape)
        n = X[0].shape[0]
        for i in range(len(train_data)):
            assert(X[i].shape[0] == n)
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
        tokens = text.split()
        words = [word.lower() for word in tokens if word.isalpha()]
        return words

    def featureExtractorX(self, inst, model):
        title = inst[label_dict["targetTitle"]]
        description = inst[label_dict["targetDescription"]]
        keywords = inst[label_dict["targetKeywords"]]
        paragraphs = inst[label_dict["targetParagraphs"]]
        captions = inst[label_dict["targetCaptions"]]
        postText = inst[label_dict["postText"]]

        #Things to go into Word2Vec: processed_title, processed_keywords, title_proper_nouns, keywords_proper_nouns, captions
        #curr_id = inst[label_dict["id"]] 
        #assert(curr_id not in self.id_set)
        #self.id_set.add(curr_id)
        processed_title = title.split()
        processed_keywords = title.split(",")
        processed_text = []
        for par in paragraphs:
            processed_text += par.split()
        processed_caption = []
        for cap in captions:
            processed_caption += cap.split()
        processed_description = description.split()

        processed_postText = []
        for post in postText:
            processed_postText += post.split()

        title_vec = self.convertToWordVector(processed_title, model)
        keyword_vec = self.convertToWordVector(processed_keywords, model)
        text_vec = self.convertToWordVector(processed_text, model)
        caption_vec = self.convertToWordVector(processed_caption, model)
        description_vec = self.convertToWordVector(processed_description, model)
        postText_vec = self.convertToWordVector(processed_postText, model)
        description
        return np.concatenate((title_vec, keyword_vec, text_vec, caption_vec, description_vec, postText_vec), axis = None)
        #assert(title_vec.shape == (300,))
        '''
        cosine_sim = []
        #title & text
        if np.linalg.norm(title_vec) == 0 or np.linalg.norm(text_vec) == 0:
            cosine_sim.append(0)
        else:
            cosine_sim.append(np.dot(title_vec, text_vec)/(np.linalg.norm(title_vec) * np.linalg.norm(text_vec)))
        if np.linalg.norm(keyword_vec) == 0 or np.linalg.norm(text_vec) == 0:
            cosine_sim.append(0)
        else:
            cosine_sim.append(np.dot(keyword_vec, text_vec)/(np.linalg.norm(keyword_vec) * np.linalg.norm(text_vec)))
        return np.asarray(cosine_sim)
        '''

    def convertToWordVector(self, input, model):
        vectors = [model[word] for word in input if word in model.vocab]
    #         percent_vocab = len(vectors) / len(input)
        if len(vectors) == 0:
            return np.zeros((300,))
        else:
            return np.mean(vectors, axis=0)
            #vec_sum = np.sum(vectors, axis = 0)
            #return vec_sum/np.linalg.norm(vec_sum)
    #     else:
    #         return model[input]