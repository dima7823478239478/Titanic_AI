import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sys

train_data_csv_path = ("/Users/dmitrii/PycharmProjects/Titanic_AI_project/titanic/train.csv")
row_train_data = pd.read_csv(train_data_csv_path)
row_train_data = row_train_data.drop('Name', axis=1)
row_train_data = row_train_data.drop('Cabin', axis=1)
row_train_data = row_train_data.drop('Ticket', axis=1)
row_train_data = row_train_data.drop('PassengerId', axis=1)
row_train_data['Sex'] = row_train_data['Sex'].replace({'male': 1, 'female': 0})#привел к виду мужчина - 1, женщина - 0
row_train_data['Embarked'] = row_train_data['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})#привел к численному виду точки отправления
row_train_data['Age'].fillna(0, inplace=True)# заполнение пустых значений в столбце возраст нулем
row_train_data.dropna(inplace=True)#Удалил пустые значения, которые невозможно заполнить
print(row_train_data['Parch'])


plt.figure(figsize=(8, 6))

plt.subplot(3, 3, 1)
#Гистограмма возрастов: Большая часть людей в возрасте от 18 до 48, меньшая от 0 до 18 и от 48 до 80
plt.hist(row_train_data['Age'], bins=10, color='blue', edgecolor='black')
plt.title('Гистограмма Возрастов')
correlation_age = row_train_data['Age'].corr(row_train_data['Survived'])
#print("Коррелляция возраста и выживаемости: ", correlation_age)#Коррелляции возраста и выживаемости нет(-0.07)

plt.subplot(3, 3, 2)
#Гистограмма классов: Большая часть людей была 3 класса
plt.hist(row_train_data['Pclass'], bins=3, color='green', edgecolor='black')
plt.title('Гистограмма классов')
correlation_class = row_train_data['Pclass'].corr(row_train_data['Survived'])
#print("Коррелляция класса и выживаемости: ",correlation_class)#Коррелляции класса и выживаемости особо нет(-0,3)

plt.subplot(3, 3, 3)
#Гистограмма : Большая часть людей была 3 класса
plt.hist(row_train_data['Fare'], bins=5, color='purple', edgecolor='black')
plt.title('Гистограмма Тарифов')
correlation_fare = row_train_data['Fare'].corr(row_train_data['Survived'])
#print("Коррелляция тарифа и выживаемости: ",correlation_fare)#Коррелляции тарифа и выживаемости особо нет(0,25)

correlation_sex = row_train_data['Sex'].corr(row_train_data['Survived'])
#print("Коррелляция пола и выживаемости: ",correlation_sex)#Коррелляции пола и выживаемости есть неплохая(-0,54)

plt.subplot(3, 3, 4)
#Гистограмма Parch: Очень много выбросов(хотя по идее должно быть 3 класса)
plt.hist(row_train_data['Parch'], bins=100, color='orange', edgecolor='black')
plt.title('Гистограмма Parch')
correlation_cabin = row_train_data['Parch'].corr(row_train_data['Survived'])
#print("Коррелляция Parch и выживаемости: ", correlation_cabin)#Коррелляции Parch и выживаемости плохая(0.08)

plt.subplot(3, 3, 5)
#Гистограмма места отправления: Большая часть людей отправилась из Southampton, а меньше всего из Queenstown
plt.hist(row_train_data['Embarked'], bins=3, color='black', edgecolor='black')
plt.title('Гистограмма места отправления')
correlation_Embarked = row_train_data['Embarked'].corr(row_train_data['Survived'])
#print("Коррелляция места отправления и выживаемости: ", correlation_Embarked)#Коррелляции места отправления и выживаемости плохая(-0.16)

correlation_age_and_class = row_train_data['Pclass'].corr(row_train_data['Age'])
#print("Коррелляция возраста и класса: ", correlation_age_and_class)#Коррелляции возраста и класса особо нет(-0.37)

correlation_age_and_fare = row_train_data['Fare'].corr(row_train_data['Age'])
#print("Коррелляция возраста и тарифа: ", correlation_age_and_fare)#Коррелляции возраста и тарифа нет(0.09)

correlation_class_and_fare = row_train_data['Fare'].corr(row_train_data['Pclass'])
#print("Коррелляция класса и тарифа: ", correlation_class_and_fare)#Коррелляции класса и тарифа есть неплохая(-0.55)

correlation_matrix = row_train_data.corr()
high_corr = correlation_matrix[abs(correlation_matrix) > 0.5]
print("Вся матрица корреляций:\n",high_corr)

x = row_train_data
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized  = pd.DataFrame(x_scaled)
print(df_normalized)#дата с нормализацией для модели лог регрессии


# Базовое сохранение (с индексами)
row_train_data.to_csv('data_clean.csv')
df_normalized.to_csv('data_clean_normal.csv')

#plt.show()







