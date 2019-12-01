import model
import util
from ntlk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2VecModel(model.Classifier):
    # We will build a Doc2Vec Model for the title
    def __init__(self, train_data):
        self.
        regr = linear_model.LinearRegression()
        X = []
        y = []
        n = len(train_data)
        for i in range(n):
            inst, truth = util.getInstMean(train_data[i])
            X.append(self.featureExtractorX(inst))
            y.append(truth)
        self.model = self.trainDoc2VecModel('targetTitle', train_data)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        regr.fit(X, y)
        def func(inst):
            return float(max(min(regr.predict(self.featureExtractorX(inst).reshape(1,-1)), 1), 0))
        self.f = func

    def trainDoc2VecModel(self, data_label, train_data):
        tagged_data = []
        for i in range(len(train_data)):
            doc = train_data[i][util.label_dict[data_label]]
            tagged_doc = TaggedDocument(words=word_tokenize(doc.lower()), tags=[train_data[i][util.label_dict['truthClass']]])
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
            print('iteration {0}'.format(epoch))
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
