import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_normal.csv')
X = data.drop(['0'], axis=1)
y = data[['0']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500, C=1.0, solver='liblinear', penalty='l2', random_state=42)
model.fit(X, y)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Вероятности для класса 1

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# ROC-кривая и значение AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC: {roc_auc:.4f}")


cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Для логистической регрессии
cv_scores_logreg = cross_val_score(
LogisticRegression(),
X,
y.values.ravel(),
cv=cv,
scoring='roc_auc'
)
print(f"Средний AUC-ROC при кросс-валидации: {cv_scores_logreg.mean():.4f}")
print(f"Стандартное отклонение AUC-ROC: {cv_scores_logreg.std():.4f}")


# Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
# построение ROC кривой
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC: %.3f" % auc)
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

