from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import pandas as pd
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from dython.nominal import associations
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('player-classification.csv')

# View Data
#print(data.describe())
#print(data.info())
#print(data.head())
#print("The number of rows in the dataset are:", data.shape[0])
#print("The number of columns in the dataset are:", data.shape[1])
# X_train, X_test, Y_train, Y_test = train_test_split(
#     data, data['PlayerLevel'], test_size=0.2, random_state=20)


##################################################################
# encoding for string values
data = data.drop(['id', 'full_name', 'birth_date',
                  'nationality', 'body_type'], axis=1)
###################################################################
# Split Positions In Splited Columns
new_player_position = data['positions'].str.get_dummies(
    sep=',').add_prefix('position')
#print(new_player_position.head())
data = pd.concat([data, new_player_position], axis=1)

#print(data.head())
#print(data.info())

column = ['LS', 'ST', 'RS',
          'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
          'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
          'RCB', 'RB']
###################################################################
# Split Values In Columns (+2)
for col in column:
    data[col] = data[col].str.split('+', n=1, expand=True)[0]

#print(data[column].head())
###################################################################
# Fill Null Values By 0
data[column] = data[column].fillna(0)
data[column] = data[column].astype(int)
#print(data[column].info())

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
 #   print(col + "="+str(val))
    data = data.drop(col, axis=1)

# Draw The Relation Between The Preferred_Foot And (overall_rating,wage,PlayerLevel)
#sns.catplot(x="wage", y="PlayerLevel",
 #           data=data, aspect=2, kind="bar")
#plt.title("Figure : \n\n Wage vs PlayerLevel",
 #         fontsize=20)

#plt.show()
#sns.catplot(x="overall_rating", y="PlayerLevel",
 #           data=data, aspect=2, kind="bar")
#plt.title("Figure : \n\n Overall_rating vs PlayerLevel",
 #         fontsize=20)

#plt.show()

#sns.catplot(x="potential", y="PlayerLevel",
 #           data=data, aspect=2, kind="bar")
#plt.title("Figure : \n\n Potential vs PlayerLevel",
 #         fontsize=20)

#plt.show()
#sns.catplot(x="age", y="PlayerLevel",
 #           data=data, aspect=2, kind="bar")
#plt.title("Figure : \n\n Age vs PlayerLevel",
 #         fontsize=20)
#plt.show()

# df = X_train[['club_team', 'potential',  'composure', 'reactions',
#                 'preferred_foot',  'PlayerLevel']]
# complete_correlation = associations(
#     df, filename='complete_correlation.png', figsize=(10, 10))
X_train, X_test, Y_train, Y_test = train_test_split(
    data, data['PlayerLevel'], test_size=0.2, random_state=20)

cols = ['club_team',
        'preferred_foot', 'club_position', 'PlayerLevel']
for c in cols:
    a = preprocessing.LabelEncoder()
    X_train[c] = a.fit_transform(X_train[c].values)
    X_test[c] = a.transform(X_test[c].values)

mx = MinMaxScaler()
col = []
for c in data.columns:
    if data[c].dtype == 'object':
        continue
    else:
        col.append(c)
X_train.fillna(0)
X_test.fillna(0)

X_train_scaled = mx.fit_transform(X_train[col])
X_test_scaled = mx.transform(X_test[col])
X_trainPD = pd.DataFrame(X_train_scaled, columns=X_train[col].columns)
X_trainPD = X_trainPD.fillna(0)
X_testPD = pd.DataFrame(X_test_scaled, columns=X_test[col].columns)
X_testPD = X_testPD.fillna(0)

#print(X_testPD.head())
clf =svm.SVC(kernel='poly', degree=4).fit(X_trainPD,Y_train)
predict=clf.predict(X_testPD)
print("Testing accurency of SVM ",metrics.accuracy_score(Y_test, predict)*100)
predict=clf.predict(X_trainPD)
print("Training accurency  svm" ,metrics.accuracy_score(Y_train, predict)*100)
with open('SVM model','wb')as f:
    pickle.dump(clf,f)



clf = tree.DecisionTreeClassifier().fit(X_trainPD,Y_train)
predict=clf.predict(X_testPD)
print("Testing accurency of decision Tree ",metrics.accuracy_score(Y_test, predict)*100)
predict=clf.predict(X_trainPD)
print("Training accurency decision Tree " ,metrics.accuracy_score(Y_train, predict)*100)
with open('decision Tree model','wb')as f:
    pickle.dump(clf,f)



#solver decides what solver to use for fitting the model
clf=LogisticRegression(solver='saga', random_state=7).fit(X_trainPD,Y_train)
predict=clf.predict(X_testPD)
print("Testing accurency of LogisticRegression ",metrics.accuracy_score(Y_test, predict)*100)
predict=clf.predict(X_trainPD)
print("Training accurency  LogisticRegression" ,metrics.accuracy_score(Y_train, predict)*100)
with open('LogisticRegression model','wb')as f:
    pickle.dump(clf,f)