import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_normal.csv')
X = data.drop(['0'], axis=1)
y = data[['0']]

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.25, random_state=42)


# Создаем классификатор KNN с 5 соседями
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')

# Обучаем модель на тренировочных данных
knn.fit(X_train, y_train)

# Делаем предсказания на тестовых данных
y_pred = knn.predict(X_test)

# Рассчитываем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")


