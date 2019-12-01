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
import model
import gensim

class FirstModel(model.Classifier):
    # Most of our classifiers should have an __init__ just like this one!
    # init youself by training on the training data and then capture your function
    # with a self.f = func to be returned later in self.getFunc()
    # Also, notice that featureExtractorX is a class function now.
    def __init__(self, train_data):
        regr = linear_model.LinearRegression()
        # model = util.generateModel()
        model = None
        X = []
        y = []
        n = len(train_data)
        for i in range(n):
            inst, truth = util.getInstMean(train_data[i])
            if i == 1:
                print(inst[label_dict["targetTitle"]])
                print(inst[label_dict["targetKeywords"]])
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
        
        #Gets punctuation counts in title
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
        '''
        #Feature: Number of stopwords in title
        def title_stopwords_feature():
            count = 0
            for word in title:
                if word in _stopwords:
                    count += 1
            return count

        #Feature: Average word length
        def avg_word_len(text):
            total_len = 0
            for word in text:
                total_len += len(word)
            return total_len//len(text)

        #Feature: Paragraph length

        #Feature: Number of paragraphs
        num_pars = len(paragraphs)

        #Feature: Number of keywords

        #Proper Nouns in Title
        tagged_title = pos_tag(title)
        title_proper_nouns = [word for word, pos in tagged_title if pos == 'NNP']

        #Proper Nouns in Keywords
        tagged_keywords = pos_tag(keywords)
        keywords_proper_nouns = [word for word, pos in tagged_keywords if pos == 'NNP']

        #Feature: Number of proper nouns in , keywords
        title_nnp_count = len(title_proper_nouns)
        keyword_nnp_count = len(keywords_proper_nouns)'''

        #Feature: Title Word2Vec
        # processed_title = processText(title)

        #Feature: Keyword Word2Vec
        #processed_keywords = processText(keywords)

        #Things to go into Word2Vec: processed_title, processed_keywords, title_proper_nouns, keywords_proper_nouns, captions
        # v1 = convertToWordVector(processed_title, model)
        v2 = np.asarray([title_punc_count, title_exclam_count, title_question_count])
        #return np.concatenate((v1, v2), axis = None)
        return v2

    # def convertToWordVector(input, model):
    #     if isinstance(input, list):
    #         vectors = [model[word] for word in input if word in model.vocab]
    #         percent_vocab = len(vectors) / len(input)
    #         return np.mean(vectors, axis=0)
    #     else:
    #         return model[input]

    # def convertToWordVectorDimReduce(word_list, dim=2):
    #     vectors = [model[word] for word in word_list if word in model.vocab]
    #     words = [word for word in word_list if word in model.vocab]
    #     percent_vocab = len(words) / len(word_list)
    #     print(str(percent_vocab) + "% of words in vocab")
    #     word_vec_dict = dict(zip(vectors, words))
    #     df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    #     tsne = TSNE(n_components = dim, init = 'random', random_state = 10, perplexity = 100)
    #     tsne_df = tsne.fit_transform(df[:500])
    #     return tsne_df