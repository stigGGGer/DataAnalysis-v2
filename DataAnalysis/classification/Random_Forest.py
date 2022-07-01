from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def Random_Forest(table,target,parametrs):
     x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=80)
     #X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
     clf = RandomForestClassifier(max_depth=2, random_state=0)
     clf.fit(x_train, y_train)
     y_pred = clf.predict(x_test)
     return  x_test.assign(Y_True=y_test,Y_Pred=y_pred)
