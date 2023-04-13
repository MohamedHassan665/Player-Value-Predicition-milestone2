import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from dython.nominal import associations
from sklearn.preprocessing import MinMaxScaler

def Preproceesingpart(data):
    ##################################################################
    # encoding for string values
    data = data.drop(['id', 'full_name', 'birth_date',
                      'nationality', 'body_type'], axis=1)
    ###################################################################
    # Split Positions In Splited Columns
    new_player_position = data['positions'].str.get_dummies(sep=',').add_prefix('position')
   # print(new_player_position.head())
    data = pd.concat([data, new_player_position], axis=1)

    #print(data.head())
    #print(data.info())
    columns = ['LS', 'ST', 'RS',
               'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
               'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
               'RCB', 'RB']
    #print(data[columns])
    ###################################################################
    # Split Values In Columns (+2)
    for col in columns:
        data[col] = data[col].str.split('+', n=1, expand=True)[0]

    #print(data[columns].head())
    ###################################################################
    # Fill Null Values By 0
    data[columns] = data[columns].fillna(0)
    data[columns] = data[columns].astype(int)
    #print(data[columns].info())
    ###################################################################
    # Fill Null Values By Mean
    data['shot_power'] = data['shot_power'].fillna(data['shot_power'].median())
    data['dribbling'] = data['dribbling'].fillna(data['dribbling'].median())

    data['wage'] = data['wage'].fillna(data['wage'].mean())
    ###################################################################
    # Drop Columns That Have Null Values >= 25%
    for col in data.columns:
        val = data[col].isnull().sum()/data[col].count()*100
        if(val >= 25):
           # print(col + "="+str(val))
            data = data.drop(col, axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, data['PlayerLevel'],test_size=0.2,random_state=20)

    return X_train, X_test, Y_train, Y_test
#############################


def Drawing(data):
    # Draw The Relation Between The Preferred_Foot And (overall_rating,wage,PlayerLevel)
    sns.catplot(x="wage", y="PlayerLevel",
                data=data, aspect=2, kind="bar")
    plt.title("Figure : \n\n Wage vs PlayerLevel",
              fontsize=20)

    plt.show()
    sns.catplot(x="overall_rating", y="PlayerLevel",
                data=data, aspect=2, kind="bar")
    plt.title("Figure : \n\n Overall_rating vs PlayerLevel",
              fontsize=20)

    plt.show()

    sns.catplot(x="potential", y="PlayerLevel",
                data=data, aspect=2, kind="bar")
    plt.title("Figure : \n\n Potential vs PlayerLevel",
              fontsize=20)

    plt.show()
    sns.catplot(x="age", y="PlayerLevel",
                data=data, aspect=2, kind="bar")
    plt.title("Figure : \n\n Age vs PlayerLevel",
              fontsize=20)
    plt.show()
#############################


def corr_matrix(X_train):
    df = X_train[['club_team', 'potential',  'composure', 'reactions',
                  'preferred_foot',  'PlayerLevel']]
   # complete_correlation = associations(        df, filename='complete_correlation.png', figsize=(10, 10))
############################


def Preprocessing_Encoding(X_train, X_test, Y_train, Y_test):
    cols = ['club_team','preferred_foot', 'club_position', 'PlayerLevel']
    for c in cols:
        a = preprocessing.LabelEncoder()
        X_train[c] = a.fit_transform(X_train[c].values)
        X_test[c] = a.transform(X_test[c].values)
    return X_train, X_test, Y_train, Y_test


def Preprocessing_Scaling(X_train, X_test, Y_train, Y_test):
    mx = MinMaxScaler()
    col = []
    for c in X_train.columns:
        if X_train[c].dtype == 'object':
            continue
        else:
            col.append(c)
    X_train_scaled = mx.fit_transform(X_train[col])
    X_test_scaled = mx.transform(X_test[col])
    X_traindf = pd.DataFrame(X_train_scaled, columns=X_train[col].columns)
    X_traindf = X_traindf.fillna(0)
    X_testdf = pd.DataFrame(X_test_scaled, columns=X_test[col].columns)
    X_testdf = X_testdf.fillna(0)

    return X_train_scaled, X_test_scaled, Y_train, Y_test
