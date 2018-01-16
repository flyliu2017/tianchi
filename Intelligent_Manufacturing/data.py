import pandas as pd
import os
import re

path="/root/PycharmProjects/tf/data/"

X_train=pd.read_csv(path+'total_train.csv',index_col=0)
y_train=pd.read_csv(path+'label.csv',header=None,index_col=0)
a_test=pd.read_csv(path+'test_a_data.csv',index_col=0)
b_test=pd.read_csv(path+'test_b_data.csv',index_col=0)

filenames=os.listdir(path)
features=pd.Index([])
for name in filenames:
    if re.search('feature_importance_rf_',name):
        fe=pd.read_csv(path+name,header=None,index_col=0)
        fe=fe.sort_values(by=[1],ascending=False)
        features=features.append(fe.index[0:100])

feature_xgb=pd.read_csv(path+'feature_importance_xgb_0.csv',header=None,index_col=0)
features=features.append(feature_xgb.index)
features=features.unique()

reduced_train_data=X_train[features]
reduced_test_data=b_test[features]

reduced_train_data.to_csv(path+'reduced_train_data.csv')
reduced_test_data.to_csv(path+'reduced_test_data.csv')

print(features)