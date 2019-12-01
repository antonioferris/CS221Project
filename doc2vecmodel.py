import model
import util
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import gensim
import random

class Doc2VecModel(model.Classifier):
    # We will build a Doc2Vec Model for the title
    def __init__(self, train_data, makeModel=False):
        regr = linear_model.LinearRegression()
        self.MODEL_LABELS = ["postText", "targetTitle", "targetDescription", "targetKeywords", "targetParagraphs", "targetCaptions"]
        if not makeModel:
            self.loadModels()
            print('Loaded Model')
        else:
            print('Attempting to create Models')
            self.trainModels(train_data)
            print('DONE creating models!')
        X = []
        y = []
        n = len(train_data)
        for i in range(n):
            inst, truth = util.getInstMean(train_data[i])
            X.append(self.featureExtractorX(inst))
            y.append(truth)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        regr.fit(X, y)
        def func(inst):
            return float(max(min(regr.predict(self.featureExtractorX(inst).reshape(1,-1)), 1), 0))
        self.f = func

    def loadModels(self):
        self.models = dict()
        for label in self.MODEL_LABELS:
            self.models[label] = Doc2Vec.load('d2v_' + label + '.model')

    # This is a 1-time use function.  This will save the models using Doc2Vec so that they can be used again!
    def trainModels(self, train_data):
        self.models = dict()
        for label in self.MODEL_LABELS:
            self.models[label] = self.trainDoc2VecModel(label, train_data, model_name='d2v_' + label + '.model')

    def trainDoc2VecModel(self, data_label, train_data, model_name='d2v.model2'):
        tagged_data = []
        for i in range(len(train_data)):
            doc = train_data[i][util.label_dict[data_label]]
            tagged_doc = TaggedDocument(words=util.processText(doc), tags=[train_data[i][util.label_dict['truthClass']]])
            tagged_data.append(tagged_doc)
        
        max_epochs = 100
        vec_size = 20
        alpha = 0.025
        model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
        
        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('{0} iteration {1}'.format(data_label, epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        
        model.save('d2v.model')
        print("Model Saved")
        return model


    def getFunc(self):
        return self.f

    def featureExtractorX(self, inst):
        doc = inst[util.label_dict['targetTitle']]
        doc_vec = self.models['targetTitle'].infer_vector(util.processText(doc))
        # if random.random() < 0.001:
        #     print(doc_vec)
        return np.asarray(doc_vec)

