"""
    This function will combine functions from data.py and model.py in order
    to fully process clickbait challenge data.
"""
import data,json
import model
import firstmodel, doc2vecmodel
import word2vecmodel_1

def main():
    print('Finished Startup')
    train_data, cross_val_data, test_data, labels = data.getTrainValTest()

    dumbc = model.DumbClassifier()
    print('Dumb Classifier (clickbait if ! or ? in the title')
    dumbc.test(cross_val_data)

    fm = firstmodel.FirstModel(train_data)
    print('First Model (num punctuation features)')
    fm.test(cross_val_data)

    dvm = doc2vecmodel.Doc2VecModel(train_data)
    print('Doc 2 Vec Model')
    dvm.test(cross_val_data)

    # wvm = word2vecmodel_1.Word2VecModel(train_data)
    # print('First Word2Vec Model')
    # wvm.test(test_data)

    # train, val, test, labels = data.getTrainValTest()
    # print(len(train), len(val), len(test))
    # print(labels)


if __name__ == '__main__':
    main()