#USING RANDOM FOREST CLASSIFIER

#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
pd.set_option('display.max_columns',100)
passengerid = test['PassengerId']

print(train.head())
print(train.shape)
print(train.info())
print(train.isnull().sum()) #Age,Cabin,Embarked

print(test.shape)
print(test.isnull().sum()) #Age ,Cabin

#drop name and cabin,passenger_id,ticket
train = train.drop(columns=['Name', 'Cabin'])
test = test.drop(columns=['Name', 'Cabin'])
train.drop(columns='PassengerId', inplace=True)
test.drop(columns='PassengerId', inplace=True)
train.drop(columns='Ticket', inplace=True)
test.drop(columns='Ticket', inplace=True)

#replace the null in Age with the mean value of the column with fillna().
train_age_mean = train['Age'].mean()
test_age_mean = test['Age'].mean()

train['Age'].fillna(train_age_mean, inplace=True)
test['Age'].fillna(test_age_mean, inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

#Embarked
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

#data analysis (Pclass)
Pclass_count = train['Pclass'].value_counts()
Pclass_count.plot(kind='bar', xlabel='Pclass', ylabel='Count')
plt.show()


#men/women on titanic
sex_count = train['Sex'].value_counts()
sex_count.plot(kind='bar', xlabel='Sex', ylabel='Count')
plt.show()


#how many people survived and how many people died.
Survived_count = train['Survived'].value_counts()
Survived_count.plot(kind='bar', xlabel='Survived', ylabel='Count')
plt.show()


#Now using seaborn let’s know how may men died and how many females died.
sns.countplot(train, x='Sex', hue='Survived')
plt.show()


#see how which class of people survived the most.
sns.countplot(train, x='Pclass', hue='Survived')
plt.show()

#handling categorical data
train['Sex'].replace({'male':1, 'female': 0}, inplace=True)
test['Sex'].replace({'male':1, 'female': 0}, inplace=True)

train['Embarked'].replace({'S':2, 'C': 1, 'Q':0}, inplace=True)
test['Embarked'].replace({'S':2, 'C': 1, 'Q':0}, inplace=True)

#features and labels
y = train['Survived']
X = train.drop(columns=['Survived'])
#train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=786)


#model
#Random Forest Classifier
rfc = RandomForestClassifier(max_depth=15)
rfc.fit(Xtrain, ytrain)
rfc_prediction = rfc.predict(Xtest)
rfc_score = accuracy_score(ytest, rfc_prediction)
print(rfc_score) #77%



#improve Random Forest with Grid search

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc = RandomForestClassifier()


grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_rfc = RandomForestClassifier(**best_params)
best_rfc.fit(Xtrain, ytrain)

rfc_prediction = best_rfc.predict(Xtest)


rfc_score = accuracy_score(ytest, rfc_prediction)
print("Accuracy Score:", rfc_score) #78%

