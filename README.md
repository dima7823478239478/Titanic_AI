## Titanic - Machine Learning from Disaster
Start here! Predict survival on the Titanic and get familiar with ML basics
Description
👋🛳️ Ahoy, welcome to Kaggle! You’re in the right place.
This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.

If you want to talk with other users about this competition, come join our Discord! We've got channels for competitions, job postings and career discussions, resources, and socializing with your fellow data scientists. Follow the link here: https://discord.gg/kaggle

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

Read on or watch the video below to explore more details. Once you’re ready to start competing, click on the "Join Competition button to create an account and gain access to the competition data. Then check out Alexis Cook’s Titanic Tutorial that walks you through step by step how to make your first submission!


The Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Data
'''columns description - 
'PassengerId' - Id of Passenger, 
'Survived' - status of survive(1 for survived, 0 for deceased), 
'Pclass' - Ticket class, 
'Name' - Name of Passnger, 
'Sex' - sex of Passenger, 
'Age' - Age in years, 
'SibSp' - of siblings / spouses aboard the Titanic,
'Parch' - of parents / children aboard the Titanic, 
'Ticket' - Ticket number, 
'Fare' - Passenger fare, 
'Cabin' - Cabin number, 
'Embarked' - Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.'''''



## My Decision
I will use logistic regression and there are 2 classes: 1 for survived, 0 for deceased.
    
I create these models and compare accuracy, results:

* Random forest - 0.798(Changed cause syntetic data)
* Logistic regression - 0.781
* KNN - 0.691
* SVM - 0.794
* GradientBoostingClassifier - 0.830
* CatBoost - 0.830(updated with new features in data)
* LightGBM - 0.830(updated with new features in data)
* AdaBoost - 0.789(very unstable)
* XGBoost - 0.812
* Ensemble(GradientBoostingClassifier, CatBoost, LightGBM) with soft voting - 0.825
* Ensemble(GradientBoostingClassifier, CatBoost, LightGBM) with hard voting - 0.821

## Ensemble
I'll create ensemble of the most accurate models. 
There are GradientBoostingClassifier, CatBoost, LightGBM(choice based on accuracy).
* accuracy in ensemble with soft voting: 0.825
* accuracy in ensemble with hard voting: 0.821

Ensemble of strongest models didn't improve accuracy(even didn't beat GradientBoostingClassifier).

## My ideas and realization:
* Idea: Ensemble - Realization: Didn't improve accuracy. Final accuracy - 0.825
* Idea: Feature Engineering - Realization: I tested new data on GradientBoostingClassifier, CatBoost, LightGBM and got such results:GradientBoostingClassifier - 0.821, CatBoost - 0.830, LightGBM - 0.830.(2 of 3 models became more accurate!)

