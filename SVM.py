from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_normal.csv')
X = data.drop(['0'], axis=1)
y = data[['0']]

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.25, random_state=42)

# 3. Создание и обучение модели (SVM)
model = SVC(kernel='linear', C=2.0)
model.fit(X_train, y_train)

# 4. Прогнозирование
predictions = model.predict(X_test)
print(f"Точность: {accuracy_score(y_test, predictions)}")