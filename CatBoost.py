import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold



data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_fe.csv')
X = data.drop(['Survived'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = CatBoostClassifier(loss_function='Logloss', random_state=42, random_strength=1, learning_rate= 0.1, l2_leaf_reg= 7, iterations= 500, depth= 4, border_count= 128, verbose=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'accuracy: {accuracy:.3f}')
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"precision recall: {precision:.3f}-{recall:.3f}")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"{scores.mean():.4f} (+/- {scores.std():.4f})")





