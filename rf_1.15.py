import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time

RANDOM_SEED=int(time.time())
date_string=time.strftime("%Y-%m-%d %X")

path="/root/PycharmProjects/tf/data/"

X_train=pd.read_csv(path+'reduced_train_data.csv',index_col=0)
y_train=pd.read_csv(path+'label.csv',header=None,index_col=0)
a_test=pd.read_csv(path+'test_a_data.csv',index_col=0)
b_test=pd.read_csv(path+'reduced_test_data.csv',index_col=0)

rf=RandomForestRegressor(n_jobs=8,oob_score=True,verbose=1,
                         n_estimators=70,
                         max_depth=6,
                         min_samples_split=10,
                         min_samples_leaf=1,
                         # min_impurity_decrease=0,
                         max_leaf_nodes=30
                         )

# param_test1 = {'min_samples_split':range(10,15,2),'min_samples_leaf':range(1,6,2),'max_depth':range(4,10,2)}
# param_test1 = {'max_depth':range(4,12,2),'n_estimators':range(50,100,10)}
# param_test1 = {'max_depth':range(4,10,2)}
# param_test1 = {'min_impurity_decrease':np.arange(0,0.03,0.01)}
param_test1 = {'max_leaf_nodes':range(26,32,2)}
#
gsearch1 = GridSearchCV(estimator = rf,
                       param_grid = param_test1, scoring='neg_mean_squared_error',cv=5,return_train_score=True,refit=True)
gsearch1.fit(X_train.values, np.reshape(y_train.values,newshape=[-1]))
print(gsearch1.best_params_)
print(gsearch1.best_score_)
# label_b=gsearch1.predict(b_test.values)

rf=gsearch1.best_estimator_
y_predict=rf.predict(X_train.values)
sub=(y_predict-y_train.T).values
mse=np.mean(np.square(sub))
print(mse)

# for i in range(5):
#     rf.fit(X_train.values,np.reshape(y_train.values,newshape=[-1]))
#     feature_importance=rf.feature_importances_
#     fi=pd.Series(feature_importance,index=X_train.keys())
#     fi.to_csv(path+'feature_importance_rf_'+str(i+5)+'.csv')

# fscore=open(path+'score_rf.txt','a')
# fscore.write(date_string+':\n')
# fscore.write(str(gsearch1.best_score_)+'\n')
# fscore.write(str(gsearch1.best_params_)+'\n\n')


# predict_b=pd.Series(label_b,index=b_test.index)
# predict_b.to_csv(path+'predict_b_rf_'+date_string+'.csv')

# fscore.close()
