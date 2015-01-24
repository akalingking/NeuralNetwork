
""" Data loader for the titanic dataset """

import pandas as pd
import numpy as np
from scipy import stats


def load_from_csv(filename, istraining=True):
    df = pd.read_csv(filename)

    # Normalize
    tickets = df['Ticket'].unique()
    
    # a. Determine if a passenger is with a group
    df['IsInGroup'] = 0
    for ticket in tickets:
        indexes = df[df['Ticket']==ticket].index.tolist()
        for index in indexes:
            df['IsInGroup'][index] = 1

    # b. Normalize fare
    for ticket in tickets:
        indexes = df[df['Ticket']==ticket].index.tolist()
        if len(indexes) > 1:
            for index in indexes:
                df['Fare'][index] = df['Fare'][index] / len(indexes)

    # c. Determine if a passenger is on cabin
    df['CabinType'] = 'X'
    for i in range(len(df['Cabin'])):
        cabin = df['Cabin'][i]
        if type(cabin) == float and np.isnan(cabin):
            df['CabinType'][i] = 'X' # assign the letter X
        else:
            df['CabinType'][i] = cabin[0] # assign the first letter
    
    # classify cabin
    df['CabinClass'] =  df['CabinType'].map({'A':1, 'B':1, 'C':1, 'D':1, 'E':1,'F':1,'G':1,'T':1,'X':0}).astype(int)
    df['Cabin'] = df['CabinClass']
    df = df.drop(['CabinClass', 'CabinType'], axis=1)


    # clean data
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)

    mode_embarked = stats.mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    df['Port'] =  df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
    df = df.drop(['Embarked'], axis=1)

    # classify data
    df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

    df = df.drop(['Sex'], axis=1)

    # Fare(test) has a missing value. Use the pivot 
    # table to determine the mean of fare for each passenger class.
    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df['Fare'] = df[['Fare', 'Pclass']].apply(lambda x:
                                            fare_means[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)

    if istraining:
        df.to_csv('../data/train_mod.csv')
    else:
        df.to_csv('../data/test_mod.csv')

    # Remove unnecessary columns
    df = df.drop(['Name', 'Ticket'], axis=1)

    # arrange columns for training set
    if istraining:
        cols = df.columns.tolist()
        cols = [cols[1]] + cols[0:1] + cols[2:]
        df = df[cols]

    return df


def load_train_data(params):
    return load_from_csv(params['TrainFile'])


def load_test_data(params):
    return load_from_csv(params['TestFile'], istraining=False)




"""
    Test for the data set routines
"""
def test_dataset():
    print 'Running', __file__, '...'
    
    params =    { 'TrainFile': '../data/train.csv', 'TestFile' : '../data/test.csv' }
    
    train_set = load_from_csv(params['TrainFile'])
    print "Training set ..."
    print train_set.info()
    print train_set.columns
    
    test_set = load_from_csv(params['TestFile'], istraining=False)
    print "Test set ..."
    print test_set.info()
    print test_set.columns



if __name__=='__main__':
    test_dataset()