import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import lightgbm as lgb

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_fe.csv')

if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)

X = data.drop(['Survived'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Используй LGBMClassifier (sklearn-совместимый)
model = lgb.LGBMClassifier(
    objective='binary',
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.2,
    min_child_samples=20,
    n_estimators=32,  # ← твой best_iteration из предыдущего запуска
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"ROC AUC: {auc:.3f}")
print(f"Precision-Recall: {precision:.3f}-{recall:.3f}")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")