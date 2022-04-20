# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


data_file= r"C:\Users\qarna\Desktop\kaggle\input\spaceship-titanic\train.csv"
data = pd.read_csv(data_file)

def Categorize(X,method=1):
    if(method==1):
        numerical_columns = [col for col in X.columns if X[col].dtype in ['int64','float64']]
        categorial_columns = [col for col in X.columns if X[col].dtype == 'object']
        X_num = X[numerical_columns]
        X_cat = X[categorial_columns]
    elif method == 2:
        from sklearn.compose import make_column_selector as selector
        numerical_columns_selector = selector(dtype_exclude='object')
        numerical_columns = numerical_columns_selector(X)
        X_num = X[numerical_columns]
        categorial_columns_selector = selector(dtype_include='object')
        categorial_columns = categorial_columns_selector(X)
        X_cat = X[categorial_columns]
        X_cat
    elif method == 3:
        X_num = X.select_dtypes(exclude=['object'])
        X_cat = X.select_dtypes(include=['object'])
        numerical_columns = X_num.columns
        categorial_columns = X_cat.columns
    else :
        print('problem')
    
    return X_num,X_cat,numerical_columns,categorial_columns

#select only numerical values
data = pd.read_csv(data_file)
Target_name = 'Transported'
y = data[Target_name]
X = data.drop(columns=[Target_name])

import time
start = time.time()
X_num,X_cat,_,_ = Categorize(X,3)



XX = pd.concat([X_num,y],axis=1)

gr1 = data.PassengerId.apply(lambda x : x[0:4])
dic = gr1.value_counts().to_dict()
group_nb = pd.Series([dic[v] for v in gr1],name='GroupNb')
XX = pd.concat([XX,group_nb],axis=1)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,9))

import seaborn as sns

XX["Luxury"] = XX[['RoomService','Spa','VRDeck']].sum(axis=1)
XX["nbnan"] = XX.isnull().sum(axis=1)
d = XX.corr()
sns.heatmap(data=d,mask=(d==1),annot=True,cmap="RdBu_r",linewidth=2)


# lux_columns = ['RoomService','Spa','VRDeck']
# XX[lu_columns]
# train_data_num.isnull().sum()
# train_data_num[:,'Luxury']=train_data_num.sum(axis=1).copy()


XX = XX.drop(columns=['Transported'])
#XX = XX.drop(columns=['Luxury'])
XX = XX.drop(columns=['RoomService','Spa','VRDeck'])
#XX = XX.drop(columns=['GroupNb'])
#XX = XX.drop(columns=['nbnan'])
#XX = XX.drop(columns=['Age'])
#XX = XX.drop(columns=['FoodCourt'])

# include le preprocessing dans la pipeline c'est plus simple
from sklearn.impute import SimpleImputer
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
# X_imputed = pd.DataFrame(imp_mean.fit_transform(X))
# X_imputed.columns = X.columns
# X_imputed.isnull().sum()

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
machine = LogisticRegression(max_iter=2000)
machine = KNeighborsClassifier(n_neighbors=3)

start = time.time()
model = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'),
                      StandardScaler(),
                      machine)
cv_result = cross_validate(model,XX,y,cv=5)
print(f"{cv_result['test_score'].mean():.3} +/- {cv_result['test_score'].std():.3}")
print(f"evaluated in {time.time()-start:.2} seconds")

# sns.displot(
#     data=data.isnull().melt(value_name="Missing"),
#     y="variable",
#     hue="Missing",
#     multiple="fill",
#     aspect=1.5
# )



# sns.displot(
#     data=data,
#     x="Spa",
#     hue="Transported",
#     kind='hist'
# )
# plt.yscale('log')
#submission_data = pd.read_csv(r"‪C:\Users\qarna\Desktop\kaggle\input\spaceship-titanic\test.csv")


#numerical_columns = [col for col in train_data.columns if train_data[col].dtype in ["int64","float64"]]
#train_data_num = train_data[numerical_columns]
#train_data_num.isnull().sum()

