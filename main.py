"""
    This function will combine functions from data.py and model.py in order
    to fully process clickbait challenge data.
"""
import data, model, json

def main():
    print('Finished Startup')
    func = model.dumbClassifier()
    fname = '.trash'
    print('Testing Dumb Classifier')
    model.testClassifier(func, fname)
    func = model.dumberClassifier()
    fname = '.trash'
    print('Testing Dumber Classifier')
    model.testClassifier(func, fname)


if __name__ == '__main__':
    main()