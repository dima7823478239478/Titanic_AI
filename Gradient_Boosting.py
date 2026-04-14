from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_normal.csv')
X = data.drop(['0'], axis=1)
y = data[['0']]

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.25, random_state=42)

# Создание модели Gradient Boosting для бинарной классификации
model = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, min_samples_split=16, n_estimators=493, max_leaf_nodes=10, min_samples_leaf=5)


# Обучение модели
model.fit(X_train, y_train)

# Получение предсказаний
y_pred = model.predict(X_test)
print(f"Точность: {accuracy_score(y_test, y_pred)}")