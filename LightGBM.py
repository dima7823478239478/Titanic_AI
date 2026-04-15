import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean.csv')

# Удаляем лишнюю колонку Unnamed: 0 (индекс)
if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)

# Теперь X и y
X = data.drop(['Survived'], axis=1)
y = data['Survived']

#переименовываем колонки в безопасные имена
X.columns = [f"col_{i}" for i in range(X.shape[1])]
print("Новые имена колонок:", X.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "min_data_in_leaf": 20,
    "verbose": -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(50)]
)

y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
