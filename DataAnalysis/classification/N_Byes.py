from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def N_Byes(table,target,parametrs):
     x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=80, random_state=0)
     gnb = GaussianNB()
     y_pred = gnb.fit(x_train, y_train).predict(x_test)
     return  x_test.assign(Y_True=y_test,Y_Pred=y_pred)
