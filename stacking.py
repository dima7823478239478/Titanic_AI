from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_normal.csv')

X = data.drop(['0'], axis=1)
y = data['0']
if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)
X.columns = [f"col_{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model1 = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, min_samples_split=16, n_estimators=493, max_leaf_nodes=10, min_samples_leaf=5)
model2 = RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',random_state=42)
#model3 = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
#model4 = SVC(kernel='linear', C=2.0, probability=True)
model5 = LogisticRegression(max_iter=500, C=1.0, solver='liblinear', penalty='l2', random_state=42)
# Soft voting
ensemble_soft = VotingClassifier(
    estimators=[('GB', model1), ('RF', model2), ('LR', model5)],
    voting='soft'
)

clf1 = model1.fit(X_train, y_train)
clf2 = model2.fit(X_train, y_train)
#clf3 = model3.fit(X_train, y_train)
#clf4 = model4.fit(X_train, y_train)
clf5 = model5.fit(X_train, y_train)
eclf_soft = ensemble_soft.fit(X_train, y_train)

y_pred_soft = eclf_soft.predict(X_test)

accuracy_soft = accuracy_score(y_test, y_pred_soft)
print(f'accuracy in ensemble with soft voting: {accuracy_soft:.3f}')
