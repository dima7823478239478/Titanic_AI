from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean.csv')

if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)

X = data.drop(['Survived'], axis=1)
y = data['Survived']

X.columns = [f"col_{i}" for i in range(X.shape[1])]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model1 = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, min_samples_split=16, n_estimators=493, max_leaf_nodes=10, min_samples_leaf=5)
model2 = CatBoostClassifier(loss_function='Logloss', random_state=42, random_strength=1, learning_rate= 0.1, l2_leaf_reg= 7, iterations= 500, depth= 4, border_count= 128, verbose=0)
model3 = lgb.LGBMClassifier(objective='binary',learning_rate=0.05,num_leaves=31,max_depth=-1,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,reg_alpha=0.1,reg_lambda=0.2,min_child_samples=20,n_estimators=30,random_state=42,verbose=-1)
# Soft voting
ensemble_soft = VotingClassifier(
    estimators=[('gb', model1), ('catboost', model2), ('lgbm', model3)],
    voting='soft'
)

# Hard voting
ensemble_hard = VotingClassifier(
    estimators=[('gb', model1), ('catboost', model2), ('lgbm', model3)],
    voting='hard'
)
clf1 = model1.fit(X_train, y_train)
clf2 = model2.fit(X_train, y_train)
clf3 = model3.fit(X_train, y_train)
eclf_soft = ensemble_soft.fit(X_train, y_train)
eclf_hard = ensemble_hard.fit(X_train, y_train)

y_pred_soft = eclf_soft.predict(X_test)
y_pred_hard = eclf_hard.predict(X_test)

accuracy_soft = accuracy_score(y_test, y_pred_soft)
accuracy_hard = accuracy_score(y_test, y_pred_hard)
print(f'accuracy in ensemble with soft voting: {accuracy_soft:.3f}')
print(f'accuracy in ensemble with hard voting: {accuracy_hard:.3f}')
