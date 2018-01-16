import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

RANDOM_SEED=int(time.time())
date_string=time.strftime("%Y-%m-%d %X")

path="/root/PycharmProjects/tf/data/"

# X_train=pd.read_csv(path+'reduced_train_data.csv',index_col=0)
y_train=pd.read_csv(path+'label.csv',header=None,index_col=0)
a_test=pd.read_csv(path+'test_a_data.csv',index_col=0)
# b_test=pd.read_csv(path+'reduced_test_data.csv',index_col=0)

scaled_combined_data=pd.read_pickle(path+'scaled_combined_data.pickle')
X_train=scaled_combined_data[0:600]
b_test=scaled_combined_data[600:]

svr=SVR(
    max_iter=-1,
    verbose=True,
    kernel='rbf',
    # gamma='auto',
    C=0.8,
    epsilon=0.1,
    tol=1.1e-2
)

# param_test1 = {'C':np.arange(0.6,1,0.1),'epsilon':np.arange(0.01,0.12,0.01),'tol':np.arange(1e-3,4e-3,1e-3)}
param_test1={'tol':np.arange(1e-2,2e-2,1e-3)}

gsearch1 = GridSearchCV(estimator =svr ,n_jobs=8,
                       param_grid = param_test1, scoring='neg_mean_squared_error',cv=5,return_train_score=True,refit=True)
gsearch1.fit(X_train, np.reshape(y_train.values,newshape=[-1]))
print(gsearch1.best_params_)
print(gsearch1.best_score_)
label_b=gsearch1.predict(b_test)

svr=gsearch1.best_estimator_
y_predict=svr.predict(X_train)
sub=(y_predict-y_train.T).values
mse=np.mean(np.square(sub))
print(mse)

# feature_importance=svr.feature_importances_
# fi=pd.Series(feature_importance,index=X_train.keys())
# fi.to_csv(path+'feature_importance_rf_'+date_string+'.csv')

# fscore=open(path+'score_svr.txt','a')
# fscore.write(date_string+':\n')
# fscore.write(str(gsearch1.best_score_)+'\n')
# fscore.write(str(gsearch1.best_params_)+'\n\n')
#
#
# predict_b=pd.Series(label_b,index=b_test.index)
# predict_b.to_csv(path+'predict_b_svr_'+date_string+'.csv')

# fscore.close()
