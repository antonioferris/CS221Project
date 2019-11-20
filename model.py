"""
    This file will contain the various models that we will employ, along with helper
    functions to test these models.
"""
import data, main
import eval, subprocess
import util
import numpy as np
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

# This classifier will say something is clickbait if it has ! or ? in the title
def dumbClassifier():
    return lambda inst : 1 if set(inst["targetTitle"]) & set("?!") else 0

# Everything is not clickbait classifier
def dumberClassifier():
    return lambda inst : 0

def featureExtractorY(truth):
    return float(truth["truthMean"])

# test tne classifier (function instance -> score)
# with eval.py
def testClassifier(func, name='untitled.testoutput', n=2000):
    results = dict()
    instance, truth = data.getRawData(train=True)
    for inst in instance:
        _id = inst["id"]
        out = func(inst)
        while isinstance(out, list):
            out = out[0]
        print(str(out)[2:-2], end='')
        results[_id] = str(out)[2:-2]
    util.dumpResults(results, '.tmpdmp')
    subprocess.run(["python", "eval.py", util.TRAIN_TRUTH_PATH, '.tmpdmp', name])

def createLinearClassifier():
    train_instance, train_truth, test_instance, test_truth = data.getTrainTestData()
    regr = linear_model.LinearRegression()
    model = util.generateModel()
    print('Done')
    X = []
    y = []
    n = len(train_instance)
    for i in range(n):
        X.append(featureExtractorX(train_instance[i], model))
        y.append(featureExtractorY(train_truth[i]))
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    print(X.shape, y.shape)
    regr.fit(X, y)
    def r(inst):
        # print(featureExtractorX(inst, model).reshape(1,300))
        return  max(min(regr.predict(featureExtractorX(inst, model).reshape(1,300)), 1), 0)
    #return lambda inst : max(min(regr.predict(featureExtractorX(inst, model).reshape(1,300)), 1), 0)
    return r

def featureExtractorX(inst, model):
    title = inst["targetTitle"]
    description = inst["targetDescription"]
    keywords = inst["targetKeywords"]
    paragraphs = inst["targetParagraphs"]
    captions = inst["targetCaptions"]

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
        processedText = processText(text)
        punct_count = collections.defaultdict(int)
        for word in processedText:
            if word in string.punctuation:
                punct_count[word] += 1
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
    processed_title = processText(title)

    #Feature: Keyword Word2Vec
    #processed_keywords = processText(keywords)

    #Things to go into Word2Vec: processed_title, processed_keywords, title_proper_nouns, keywords_proper_nouns, captions
    v1 = convertToWordVector(processed_title, model)
    #v2 = np.asarray([title_punc_count, title_exclam_count, title_question_count])
    #return np.concatenate((v1, v2), axis = None)
    return v1

def convertToWordVector(input, model):
    if isinstance(input, list):
        vectors = [model[word] for word in input if word in model.vocab]
        percent_vocab = len(vectors) / len(input)
        return np.mean(vectors, axis=0)
    else:
        return model[input]

def convertToWordVectorDimReduce(word_list, dim=2):
    vectors = [model[word] for word in word_list if word in model.vocab]
    words = [word for word in word_list if word in model.vocab]
    percent_vocab = len(words) / len(word_list)
    print(str(percent_vocab) + "% of words in vocab")
    word_vec_dict = dict(zip(vectors, words))
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    tsne = TSNE(n_components = dim, init = 'random', random_state = 10, perplexity = 100)
    tsne_df = tsne.fit_transform(df[:500])
    return tsne_df



