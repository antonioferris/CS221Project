"""
    This function will combine functions from data.py and model.py in order
    to fully process clickbait challenge data.
"""
import data, model, json

def main():
    print('Finished Startup')
    # func = model.dumberClassifier()
    # fname = '.trash'
    # print('Testing Dumb Classifier')
    # model.testClassifier(func, fname)
    # func = model.createLinearClassifier()
    # fname = '.trash'
    # print('Testing Linear Classifier')
    # model.testClassifier(func, fname)
    train, val, test, labels = data.getTrainValTest()
    print(len(train), len(val), len(test))
    print(labels)


if __name__ == '__main__':
    main()