import pandas as pd
import numpy as np


train_dataframe = pd.read_csv('train.csv')

np_train = train_dataframe.values
list_survived = []
list_dead = []

for i in range(np_train.shape[0]):
    if np_train[i][1]==1:
        list_survived.append(np_train[i])
    else:
        list_dead.append(np_train[i])

np_survived = np.array(list_survived)
np_dead = np.array(list_dead)

df_survived = pd.DataFrame.from_records(np_survived)
df_survived.columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df_dead = pd.DataFrame.from_records(np_dead)
df_dead.columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


df_survived.to_csv('lucky_passengers.csv', index=False)
df_dead.to_csv('unlucky_passenger.csv', index=False)

