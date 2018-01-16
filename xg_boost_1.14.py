import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

import pickle
import datetime

now = datetime.datetime.now()

date_string=now.strftime('%Y-%m-%d %H:%M:%S')
path="/root/PycharmProjects/tf/data/"

X_train=pd.read_csv(path+'reduced_train_data.csv',index_col=0)
y_train=pd.read_csv(path+'label.csv',header=None,index_col=0)
a_test=pd.read_csv(path+'test_a_data.csv',index_col=0)
b_test=pd.read_csv(path+'reduced_test_data.csv',index_col=0)

clf = XGBRegressor(
silent=False ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.06, # 如同学习率
min_child_weight=1,
n_jobs=8,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=3, # 构建树的深度，越大越容易过拟合
gamma=0.08,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=0.9, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样
reg_lambda=0.5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
reg_alpha=0.07, # L1 正则项参数
# scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
objective= 'reg:linear', #多分类的问题 指定学习任务和相应的学习目标
# num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=110, #树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
)
# clf.fit(X_train,y_train,eval_metric='auc')
# y_true, y_pred = y_test, clf.predict(X_test)
# print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))

param_test1 = {'learning_rate':np.arange(0.05,0.1,0.01)}
# param_test1 = {'gamma':np.arange(0.05,0.15,0.03)}

# param_test1 = {'reg_alpha':np.arange(0.01,0.1,0.03),'min_child_weight':np.arange(0.1,1,0.3)}
# param_test1 = {'reg_alpha':np.arange(0.04,0.11,0.03),'reg_lambda':np.arange(0.4,0.7,0.1),'gamma':np.arange(0.05,0.15,0.03)}
# param_test1 = {'max_depth':range(3,6,1),'min_child_weight':np.arange(1,4,1)}
# param_test1 = {'subsample':[0.9,1],'n_estimators':range(100,130,10)}
gsearch1 = GridSearchCV(estimator = clf,
                       param_grid = param_test1, scoring='neg_mean_squared_error',cv=5,return_train_score=True,refit=True)
gsearch1.fit(X_train,y_train)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
y_predict=gsearch1.predict(X_train)
label_b=gsearch1.predict(b_test)
# clf.fit(X_train,y_train)
# y_predict=clf.predict(X_train)
sub=(y_predict-y_train.T).values
mse=np.mean(np.square(sub))
print(mse)
# feature_importance=clf.get_booster().get_fscore()
# label_b=clf.predict(b_test)
#
#
# fscore=pd.Series(feature_importance)
# fscore.to_csv(path+'feature_importance_xgb_'+str(i)+'.csv')

# predict_b=pd.Series(label_b,index=b_test.index)
# predict_b.to_csv(path+'predict_b_'+date_string+'.csv')
