import gensim
import util
from util import label_dict


inst, truth = util.getInstMean(test_data[i])
keywords = inst[label_dict["targetKeywords"]]
title = inst[label_dict["targetTitle"]]

print(keywords)
print(title)

#model = gensim.models.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary = True)

print(model["bicycle"])