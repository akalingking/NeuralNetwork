
#
# Sample classification problem using Titanic dataset from Kaggle
#
""" author :: Ariel Kalingking <akalingking@gmail.com> """

import datetime
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import grid_search
from sklearn import cross_validation as cv
import dataset
import neuralnetwork as N


def process_test_data(params, estimator, x_test_index, x_test):
    y_test_pred = estimator.predict(x_test)
    result = np.c_[x_test_index, y_test_pred].astype(int)
    submission = pd.DataFrame(result, columns=['PassengerId', 'Survived'])

    filename = '../data/submission_' + params['Model'] + '.csv'

    submission.to_csv(filename, index=False)



def main():
    # 1. Generate data
    print 'Running', __file__, '...'

    params =    {
                'Model'                 : 'neuralnetwork',
                'TrainFile'             : '../data/train.csv',
                'TestFile'              : '../data/test.csv',
                'n_fold'                : 5,
                'TrainSize'             : .9
                }

    df = dataset.load_train_data(params)
    train_data = df.values
    
    # Start in the PClass column, we will not be using the passengerid
    X_train = train_data[:,2:]
    Y_train = train_data[:,0].astype(int)
    
    # Partition training data
    trainSize = int(params['TrainSize'] * np.size(Y_train))
    x_train, x_valid = X_train[:trainSize, :], X_train[trainSize:,:]
    y_train, y_valid = Y_train[:trainSize], Y_train[trainSize:]

    df = dataset.load_test_data(params)
    X_test = df.values
    x_test_index = X_test[:,0]
    x_test = X_test[:,1:]


    print 'Analyzing training data ', params['Model'], 'datapoints=', x_train.shape[0], 'features=',x_train.shape[1]
    rng = np.random.RandomState(5000)
    
    classifier = N.NeuralNetwork()

    param_grid = dict(network           = [[9,18,18,1],[9,24,1],[9,45,1]],
                      connection_rate   = [.6,.7],
                      learning_rate     = [.07,.1],
                      learning_momentum = [.005,.05],
                      initial_weight    = [.73,.82],
                      desired_error     = [0.0001],
                      epoch             = [100],
                      hidden_activation = [N.SIGMOID, N.SIGMOID_STEPWISE, N.SIGMOID_SYMMETRIC],
                      output_activation = [N.SIGMOID_SYMMETRIC],
                      training_algorithm = [N.TRAIN_RPROP],
                      show              = [500])
    
    
    cv_ = cv.StratifiedShuffleSplit(y_train, n_iter=params['n_fold'], train_size=params['TrainSize'], random_state=rng)
    
    grid = grid_search.GridSearchCV(classifier, param_grid=param_grid, cv=cv_)
    
    grid.fit(x_train, y_train)
    best_estimator = grid.best_estimator_
    print 'Best estimator:', best_estimator

    scores = cv.cross_val_score(best_estimator, x_train, y_train, cv=params['n_fold'])
    print('Train: (folds=%d) Score for %s accuracy=%0.5f (+/- %0.5f)' % \
          (params['n_fold'], params['Model'], scores.mean(), scores.std()))

    y_valid_pred = best_estimator.predict(x_valid)

    print"Valid:           Score for %s accuracy=%0.5f rmse=%0.5f" % \
        (params['Model'], metrics.accuracy_score(y_valid, y_valid_pred),
        np.sqrt(metrics.mean_squared_error(y_valid, y_valid_pred)))


    print 'Analyzing test data ', params['Model'], 'datapoints=', x_test.shape[0], 'features=',x_test.shape[1]
    process_test_data(params, best_estimator, x_test_index, x_test)



if __name__ == '__main__':
    main()
