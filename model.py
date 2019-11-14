"""
    This file will contain the various models that we will employ, along with helper
    functions to test these models.
"""
import data, main
import eval, subprocess
import util

# This classifier will test itself on the truth dataset
def dumbClassifier():
    return lambda inst : 1 if set(inst["targetTitle"]) & set("?!") else 0

def dumberClassifier():
    return lambda inst : 0

def testClassifier(func, name='untitled.testoutput'):
    results = dict()
    instance, truth = data.getRawData(val=True)
    for inst in instance:
        _id = inst["id"]
        results[_id] = str(func(inst))
    util.dumpResults(results, '.tmpdmp')
    subprocess.run(["python", "eval.py", util.VAL_TRUTH_PATH, '.tmpdmp', name])
