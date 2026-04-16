import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier



data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_fe.csv')
X = data.drop(['Survived'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = CatBoostClassifier(loss_function='Logloss', random_state=42, random_strength=1, learning_rate= 0.1, l2_leaf_reg= 7, iterations= 500, depth= 4, border_count= 128, verbose=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: ', accuracy)





