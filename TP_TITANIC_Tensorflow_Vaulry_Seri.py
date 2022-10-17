#!/usr/bin/env python
# coding: utf-8

# In[78]:


import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")


# In[79]:


train = pd.read_csv("train.csv")


# In[80]:


train.info()


# In[81]:


#suppression des colonnes non utiles
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)


# In[82]:


#Remplacer les valeurs de la Q,S,C par 1, 2 et 3

Q = 0
S = 0
C = 0
N = 0

for i in range(len(train)):
    if train.iloc[i,8] == "Q":
        Q+=1
    elif train.iloc[i,8] == "S":
        S+=1
    elif train.iloc[i,8] == "C":
        C+=1
    else:
        N+=1
print("Q :", Q, " S :",S," C :",C," N :",N)



train['Embarked'] = train['Embarked'].replace(['Q'],'1')
train['Embarked'] = train['Embarked'].replace(['S'],'2')
train['Embarked'] = train['Embarked'].replace(['C'],'3')
train['Embarked'] = train['Embarked'].fillna('2')


# In[83]:


#Remplcaer les données catégoriques "male et female" par 1 et 0
train['Sex'] = train['Sex'].replace(['male'],'1')
train['Sex'] = train['Sex'].replace(['female'],'0')


# In[84]:


#conversion colonne Name

newV = train['Name'].copy()
liste = []

for i in range(len(newV)):
    valeur = newV[i]
    chaine = re.findall(" ([A-Za-z]+)\.", valeur)[0]
    
    if chaine not in liste:
        liste.append(chaine)
    
    if chaine == "Mr":
        newV[i] = 1
    elif chaine == "Miss":
        newV[i] = 2
    elif chaine == "Mrs":
        newV[i] = 3
    elif chaine == "Master":
        newV[i] = 4
    elif chaine == "Dr":
        newV[i] = 5
    elif chaine == "Rev":
        newV[i] = 6
    elif chaine == "Don":
        newV[i] = 7
    elif chaine == "Mme":
        newV[i] = 8
    elif chaine == "Ms":
        newV[i] = 9
    elif chaine == "Major":
        newV[i] = 10
    elif chaine == "Lady":
        newV[i] = 11
    elif chaine == "Sir":
        newV[i] = 12
    elif chaine == "Mlle":
        newV[i] = 13
    elif chaine == "Col":
        newV[i] = 14
    elif chaine == "Capt":
        newV[i] = 15
    elif chaine == "Countess":
        newV[i] = 16
    elif chaine == "Jonkheer":
        newV[i] = 17
    else:
        newV[i] = 18
        

train['Name'] = newV 
print(liste)


# In[85]:


averageAge = round(train['Age'].sum()/len(train))
train['Age'] = train['Age'].fillna(averageAge)


# In[86]:


train['Embarked'] = train['Embarked'].astype('int64')
train['Name'] = train['Name'].astype('int64')
train['Sex'] = train['Sex'].astype('int64')


# In[87]:


AvecValeursManquantes = len(train)
train = train.dropna()
SansValeursManquantes = len(train)

print("Avec Valeurs Manquantes :", AvecValeursManquantes,"  Sans Valeurs Manquantes :",SansValeursManquantes)

y = train['Survived']
train = train.drop(['Survived'], axis=1)
print(y)


# In[88]:


import time
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
train = scaler.fit_transform(train)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)


start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[170]:


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


# In[171]:


train.shape


# In[172]:


#traduction de notre MLP avec le  scoore le plus haut sur KEras

model = Sequential()
model.add(Input(train.shape))
model.add(Dense(units=70, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss="BinaryCrossentropy", optimizer="adam", metrics=["accuracy"])


# In[173]:


model.fit(X_train, y_train, batch_size=400, epochs = 10, verbose =1)
score = model.evaluate(X_test, y_test)


# In[174]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[121]:


model.summary()


# In[ ]:




