from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge
from xgboost.sklearn import XGBRegressor
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

RANDOM_SEED=time.time()
date_string=time.strftime("%Y-%m-%d %X")
path="/root/PycharmProjects/tf/data/"

y_train=pd.read_csv(path+'label.csv',header=None,index_col=0)
a_test=pd.read_csv(path+'test_a_data.csv',index_col=0)
b_test=pd.read_csv(path+'reduced_test_data.csv',index_col=0)

scaled_combined_data=pd.read_pickle(path+'scaled_combined_data.pickle')
X_train=scaled_combined_data[0:600]
scaled_b_test=scaled_combined_data[600:]

ridge = Ridge(solver='lsqr',random_state=RANDOM_SEED,
              alpha=17,
              # tol=0.016
              )

svr=SVR(
    verbose=True,
    kernel='rbf',
    C=0.8,
    epsilon=0.1,
    tol=1.1e-2
)

rf=RandomForestRegressor(n_jobs=8,oob_score=True,verbose=1,
                         n_estimators=70,
                         max_depth=6,
                         min_samples_split=10,
                         min_samples_leaf=1,
                         # min_impurity_decrease=0,
                         max_leaf_nodes=30
                         )

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
# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(int(RANDOM_SEED))
stack = StackingCVRegressor(regressors=(rf,clf,svr),
                            meta_regressor=ridge,
                            use_features_in_secondary=True)

# params = {'lasso__alpha': [0.1, 1.0, 10.0],
#           'ridge__alpha': [0.1, 1.0, 10.0]}

# params = {'svr__C': np.arange(0.5,2,0.5),
#           'svr__epsilon': np.arange(0.1,0.4,0.1),
#           'svr__kernel':['rbf','linear']}

params = {'meta-ridge__alpha': np.arange(1450,1600,50),
          # 'meta-ridge__tol': np.arange(0.02,0.2,0.06)
          }

grid = GridSearchCV(
    estimator=stack,
    param_grid=params,
    cv=5,
    refit=True,
    scoring='neg_mean_squared_error'
)

grid.fit(X_train, np.reshape(y_train.values,newshape=[-1]))

print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
# fscore=open(path+'score.txt','a')
# fscore.write(date_string+':\n')
# fscore.write(str(grid.best_score_)+'\n')
# fscore.write(str(grid.best_params_)+'\n\n')

label_b=grid.predict(scaled_b_test)
y_predict=grid.predict(X_train)

predict_b=pd.Series(label_b,index=b_test.index)
predict_b.to_csv(path+'predict_b_'+date_string+'.csv')

sub=(y_predict-y_train.T).values
mse=np.mean(np.square(sub))
print(mse)

# {
#         'lasso__alpha': [x/5.0 for x in range(1, 10)],
#         'ridge__alpha': [x/20.0 for x in range(1, 10)],
#         'meta-randomforestregressor__n_estimators': [10, 100]
#     },