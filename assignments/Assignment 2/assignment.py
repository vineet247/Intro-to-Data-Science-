import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import roc_auc_score

start_time = time.time()

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submit_df = pd.read_csv('submit.csv')




test_df = pd.get_dummies(test_df, columns=["COLLEGE","REPORTED_SATISFACTION",
	"REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN"], prefix = ["college","satisfaction","usage","consideration"])#convert categorical data using one-hot-encoding
#print("Test shape: ",test_df.shape)
#print("Test columns: ", test_df.dtypes)
df = pd.get_dummies(df, columns=["COLLEGE","REPORTED_SATISFACTION","REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN"], prefix = ["college","satisfaction","usage","consideration"])
#print("Train shape: ", df.shape)
X = df.drop('LEAVE', axis = 1) #Drop the leave column from train dataset
Y = df.loc[:,'LEAVE']
A = test_df
print("Shape of X", X.shape)
print("Shape of Y", Y.shape)
X = X.values						#Convert train data frame to values
sample_X = X
Y = Y.values
sample_Y = Y

features = SelectKBest(f_classif, k=10).fit(X, Y).get_support(indices=True)	#Select the 10 best features from the dataset. The number of features vary and this number is got after repeated experimentation
X = SelectKBest(f_classif, k=10).fit_transform(X,Y)
print("features:", features)	#printing features 
A = test_df.loc[:,["INCOME","OVERAGE","LEFTOVER","HOUSE","HANDSET_PRICE","OVER_15MINS_CALLS_PER_MONTH","AVERAGE_CALL_DURATION" ,"college_zero","satisfaction_avg"
,"satisfaction_sat"]]			#Convert test data frame to values

#pca = PCA(n_components = 10)
#X = pca.fit_transform(X)
print("New shape of X", X.shape)
#A = pca.fit_transform(A)
print("Test set:", A.shape)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

#Models include Decision Trees, ExtraTrees, AdaBoost, GradientBoost and RandomForestClassifier
"""
#Decision tree
decision_tree = DecisionTreeClassifier(max_depth = 5, min_samples_split = 4, min_samples_leaf = 2, max_features = "auto")
decision_tree.fit(x_train,y_train)
testsing = decision_tree.predict(x_test)
submission = decision_tree.predict(A.values)
submit_df = pd.DataFrame({'LEAVE':submission})
print("submission: ", submit_df.shape)
submit_df.to_csv("submit.csv")
print("Decision Tree Accuracy: ",accuracy_score(testsing,y_test))

dot_data = tree.export_graphviz(decision_tree, out_file='trees.dot')
from subprocess import call
call(['dot', '-Tpng', 'trees.dot', '-o', 'trees.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename = 'trees.png')


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(criterion='entropy',n_estimators = 15000, max_features=9, class_weight= {0:2.3, 1:0.8}, warm_start=True)
clf.fit(x_train,y_train)
predict = clf.predict(x_test)
print("Extra tree: ", clf.score(x_test,y_test))
print("ROC Score:", roc_auc_score(y_test,predict))


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(learning_rate = 0.005, algorithm='SAMME.R', n_estimators = 50000)
clf.fit(x_train,y_train)
predict = clf.predict(x_test)
print("\n\nAda Boost: ", clf.score(x_test,y_test))
print("ROC Score:", roc_auc_score(y_test,predict))
"""
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(loss='deviance', learning_rate = 0.001, n_estimators = 8000, subsample = 1, criterion='friedman_mse'
		, max_features = 'sqrt')
clf.fit(x_train,y_train)
predict = clf.predict(x_test)
print("\n\nGradient Boost: ", clf.score(x_test,y_test))
print("ROC Score:", roc_auc_score(y_test,predict))
submission = clf.predict_proba(A.values)
submit_df = pd.DataFrame({'LEAVE':submission[:,1]})
submit_df.to_csv("submit.csv")


"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
clf = RandomForestClassifier(n_estimators=8000, max_depth=10,random_state=0,criterion='entropy',min_samples_split=4)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
y_predict_proba = clf.predict_proba(x_test)
submission = clf.predict_proba(A.values)
print("shape :", submission.shape)
submit_df = pd.DataFrame({'LEAVE':submission[:,1]})
print("submission: ", submit_df.shape)
submit_df.to_csv("submit.csv")
print("Random forest score: ",clf.score(x_test,y_test))
print("ROC Score:", roc_auc_score(y_test,y_predict))
print("Class probabilities: ", y_predict_proba)
"""