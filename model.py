"""
    This file will contain the various models that we will employ, along with helper
    functions to test these models.
"""
import data, main
import eval, subprocess
import util
import nltk
nltk.download('stopwords')
import collections
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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
    stopwords = set(stopwords.words('english'))
    title = inst["targetTitle"]
    description = inst["targetDescription"]
    keywords = inst["targetKeywords"]
    paragraphs = inst["targetParagraphs"]

    #Process text
    def processText(text):
        tokens = word_tokenize(text)
        lower = [word.lower() for word in tokens if word.isalpha()]
        stop_tokens = [word in lower if word not in stopwords]
        return stop_tokens
    
    #Gets punctuation counts in title
    def countPunc(text):
        processedText = processText(text)
        punct_count = collections.defaultdict(int)
        for word in processedText:
            if word in string.punctuation:
                punct_count[word] += 1
                total_count += 1
        return punct_count, total_count

    #Feature: Title punctuation
    title_punc, title_punc_count = countPunc(title)

    #Feature: count of !
    title_exclam_count = title_punc["!"]

    #Feature: count of ?
    title_question_count = title_punch['?']

    #Feature: Number of stopwords in title
    def title_stopwords_feature:
        count = 0
        for word in title:
            if word in stopwords:
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

    #Feature: Number of keywords

    #Feature: Number Named Entities

    #Proper Nouns in Title
    tagged_title = pos_tag(title)
    title_proper_nouns = [word for word, pos in tagged_title if pos == 'NNP']

    #Proper Nouns in Keywords
    tagged_keywords = pos_tag(keywords)
    keywords_proper_nouns = [word for word, pos in tagged_keywords if pos == 'NNP']

    #Feature: Title Word2Vec
    processed_title = processText(title)

    #Feature: Keyword Word2Vec
    processed_keywords = processText(keyword)



