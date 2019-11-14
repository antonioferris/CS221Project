"""
    This function will combine functions from data.py and model.py in order
    to fully process clickbait challenge data.
"""
import data, model
import random, sys

def getOutputString(_id, clickbaitScore):
    return '{{ "id" : "{}", "clickbaitScore" : {} }}\n'.format(_id, clickbaitScore)

# Takes in a dict results to pathname
# results is a dict id -> clickbaitScore
def dumpResults(results, pathname=None):
    if pathname == None:
        for (_id, clickbaitScore) in results.items():
            print(getOutputString(_id, clickbaitScore), end='')
    else:
        with open(pathname, "w") as outfile:
            for (_id, clickbaitScore) in results.items():
                outfile.write(getOutputString(_id, clickbaitScore))

def testDump():
    results = {str(i) : random.random() for i in range(10)}
    dumpResults(results)
    print('test dump done')

def main():
    print('')


if __name__ == '__main__':
    main()