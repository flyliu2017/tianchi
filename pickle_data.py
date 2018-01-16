import pickle
import pandas as pd

path="/root/PycharmProjects/tf/data/"

X_train=pd.read_csv(path+'train_data.csv',index_col=0)
y_train=pd.read_csv(path+'label_data.csv',header=None,index_col=0)
a_test=pd.read_csv(path+'test_a_data.csv',index_col=0)
b_test=pd.read_csv(path+'test_b_data.csv',index_col=0)

f_X_train=open(path+'X_train.pickle','wb')
f_y_train=open(path+'y_train.pickle','wb')
f_a_test= open(path+'a_test.pickle' ,'wb')
f_b_test= open(path+'b_test.pickle' ,'wb')

pickle.dump(X_train,f_X_train)
pickle.dump(y_train,f_y_train)
pickle.dump(a_test,f_a_test)
pickle.dump(b_test,f_b_test)