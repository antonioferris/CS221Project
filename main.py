"""
    This function will combine functions from data.py and model.py in order
    to fully process clickbait challenge data.
"""
import data,json
import model
import firstmodel, doc2vecmodel
import word2vecmodel_1
import multiclassmodel

def main():
    print('Finished Startup')   
    train_data, cross_val_data, test_data = data.getTrainValTest()

    #data.pickleData()

    # dumbc = model.DumbClassifier()
    # print('Dumb Classifier (clickbait if ! or ? in the title')
    # dumbc.test(cross_val_data)

    # fm = firstmodel.FirstModel(train_data)
    # print('First Model (num punctuation features)')
    # fm.test(cross_val_data)

    #dvm = doc2vecmodel.Doc2VecModel(train_data)
    #print('Doc 2 Vec Model')
    #dvm.test(cross_val_data)

    # mcm = multiclassmodel.MultiClassModel(train_data);
    # print('MultiClass Model')
    # mcm.testMulti(test_data)

    print("BABAGANOUSH")
    wvm = word2vecmodel_1.Word2VecModel(train_data)
    print('First Word2Vec Model')
    wvm.test(cross_val_data, threshold = 0.5)

    print("TRAIN ERROR")
    wvm.test(train_data, threshold = 0.5)

    # train, val, test, labels = data.getTrainValTest()
    # print(len(train), len(val), len(test))
    # print(labels)


if __name__ == '__main__':
    main()