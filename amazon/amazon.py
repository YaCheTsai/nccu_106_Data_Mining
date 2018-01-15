# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data_train = pd.read_csv(SCRIPT_PATH + "/train.csv") 

fig = plt.figure()
fig.set(alpha=0.1)  # 设定图表颜色alpha参数

train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Name_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]