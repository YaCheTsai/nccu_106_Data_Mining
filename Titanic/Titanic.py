# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt



SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data_train = pd.read_csv(SCRIPT_PATH + "/train.csv") 


from sklearn.ensemble import RandomForestRegressor

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass', 'Name']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
    
def replace_name(x):  

    if 'Mrs' in x:  

       return '1'  

    elif 'Mr' in x or 'Miss' in x:  

        return '2'  
        
    else:  

        return '3'  

def replace_age(x):  

    if x < 13:  

       return '0'  
       
    else:  

       return '1' 
    
def set_FamilySize(row): 
    x = int(row['Parch']) + int(row['SibSp']) 
    
    if x <=4 or x>=2: 

        return '1' 
    else:  

       return '0' 

       
def replace_Cabin(x): 
    x = str(x)
    if 'B' in x or 'C' in x or 'D' in x or 'E' in x:

       return '1'  
    else:  

        return '0'  
def replace_Ticket(df): 
    for i in [df]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
       
    return df
    
def set_Embarked(df):
    df.loc[ (df.Embarked.isnull()), 'Embarked' ] = "S"
    return df    

data_train['Name']=data_train['Name'].map(lambda x:replace_name(x)) 
data_train, rfr = set_missing_ages(data_train)
#data_train['Cabin']=data_train['Cabin'].map(lambda x:replace_Cabin(x)) 
data_train['Age']=data_train['Age'].map(lambda x:replace_age(x)) 
data_train['FamilySize'] = data_train.apply(set_FamilySize, axis=1)
#data_train = replace_Ticket(data_train)
data_train = set_Embarked(data_train)



#特征因子化[0,1]
#dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

dummies_name = pd.get_dummies(data_train['Name'], prefix= 'Name')

dummies_age = pd.get_dummies(data_train['Age'], prefix= 'Age')

dummies_FamilySize = pd.get_dummies(data_train['FamilySize'], prefix= 'FamilySize')

df = pd.concat([data_train,  dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_name,dummies_age,dummies_FamilySize], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','FamilySize','Parch','SibSp'], axis=1, inplace=True)

df
#特征化到[-1,1]之内

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(df['Fare'].reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1, 1), fare_scale_param)
df
#===================================================================================================================
#逻辑回归建模
from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Name_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]
#
from sklearn.model_selection import train_test_split  

from sklearn.linear_model import LogisticRegression  

lr=linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)  

lr.fit(X, y) 

#print lr.score(X, y)  

from sklearn.svm import SVC  

svc=SVC(C=2, kernel='rbf', decision_function_shape='ovo')  

svc.fit(X, y) 

#print svc.score(X, y)  

from sklearn.ensemble import RandomForestClassifier  

randomf=RandomForestClassifier(n_estimators=250,oob_score=True,max_depth=5,random_state=10,n_jobs=-1)  


randomf.fit(X, y) 

print randomf.score(X, y)  

from sklearn.ensemble import GradientBoostingClassifier  

gdbt=GradientBoostingClassifier(n_estimators=600,max_depth=5,random_state=0)  

gdbt.fit(X, y)  

#print gdbt.score(X, y)  

from sklearn.ensemble import VotingClassifier  

model=VotingClassifier(estimators=[('lr',lr),('svc',svc),('rf',randomf),('gdbt',gdbt)],voting='hard')  

model.fit(X, y)  

#print model.score(X, y)  

#====================================================================================================================
#input test data
data_test = pd.read_csv(SCRIPT_PATH + "/test.csv") 
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄

data_test['Name']=data_test['Name'].map(lambda x:replace_name(x)) 
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass', 'Name']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
#data_test['Cabin']=data_test['Cabin'].map(lambda x:replace_Cabin(x)) 
data_test['Age']=data_test['Age'].map(lambda x:replace_age(x)) 
data_test['FamilySize'] = data_test.apply(set_FamilySize, axis=1)
#data_test = replace_Ticket(data_test)
data_test = set_Embarked(data_test)

#dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
dummies_name = pd.get_dummies(data_test['Name'], prefix= 'Name')
dummies_age = pd.get_dummies(data_test['Age'], prefix= 'Age')
dummies_FamilySize = pd.get_dummies(data_test['FamilySize'], prefix= 'FamilySize')


df_test = pd.concat([data_test,  dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_name,dummies_age,dummies_FamilySize], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','FamilySize','Parch','SibSp'], axis=1, inplace=True)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].reshape(-1, 1), fare_scale_param)
df_test


#预测取结果
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Name_.*')
predictions = randomf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})    
result.to_csv(SCRIPT_PATH + "/submission1.csv", index=False)


