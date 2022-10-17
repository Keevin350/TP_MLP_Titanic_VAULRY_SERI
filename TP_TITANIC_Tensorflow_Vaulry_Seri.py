#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")


# In[2]:


train = pd.read_csv("train.csv")


# In[3]:


train.info()


# In[4]:


#suppression des colonnes non utiles
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)


# In[5]:


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


# In[6]:


#Remplcaer les données catégoriques "male et female" par 1 et 0
train['Sex'] = train['Sex'].replace(['male'],'1')
train['Sex'] = train['Sex'].replace(['female'],'0')


# In[7]:


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


# In[8]:


averageAge = round(train['Age'].sum()/len(train))
train['Age'] = train['Age'].fillna(averageAge)


# In[9]:


train['Embarked'] = train['Embarked'].astype('int64')
train['Name'] = train['Name'].astype('int64')
train['Sex'] = train['Sex'].astype('int64')


# In[10]:


AvecValeursManquantes = len(train)
train = train.dropna()
SansValeursManquantes = len(train)

print("Avec Valeurs Manquantes :", AvecValeursManquantes,"  Sans Valeurs Manquantes :",SansValeursManquantes)

y = train['Survived']
train = train.drop(['Survived'], axis=1)
print(y)


# In[11]:


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


# In[12]:


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow import keras


# In[13]:


train.shape


# In[14]:



#traduction de notre MLP avec le  scoore le plus haut sur KEras

model = Sequential()
model.add(Input(train.shape))
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss="BinaryCrossentropy", optimizer= "adam", metrics=["accuracy"])


# In[15]:


model.fit(X_train, y_train, batch_size=200, epochs = 10, verbose =1)
score = model.evaluate(X_test, y_test)


# In[16]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[24]:


epochs = [10,20,30,40,50,100,200]
batch_size = [50,100,200,300,400,500,600]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
meilleurI = 0
meilleurZ = 0
meilleurJ = 0
meilleurA = 0
maximum = 0
loss = 0 

    
for a in range(3):
    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2)
    for i in range(len(optimizer)):

        for j in range(len(batch_size)):

            for z in range(len(epochs)):

                model = Sequential()
                model.add(Input(train.shape))

                if a >= 0:
                    model.add(Dense(units=100, activation="relu"))
                if a >= 1:
                    model.add(Dense(units=100, activation="relu"))
                if a>= 2:
                    model.add(Dense(units=100, activation="relu"))

                model.add(Dense(units=1, activation="sigmoid"))
                model.compile(loss="BinaryCrossentropy", optimizer= optimizer[i], metrics=["accuracy"])

                model.fit(X_train, y_train, batch_size=batch_size[j], epochs = epochs[z], verbose =0)
                score = model.evaluate(X_test, y_test)

                if maximum < score[1]:
                    maximum = score[1]
                    meilleurI = i
                    meilleurZ = z
                    meilleurJ = j
                    meilleurA = a
                    loss = score[0]

    print("tour a : ", a)


# In[25]:


print("Meilleur score : ", maximum, " avec optimizer, batch_size, epochs, a = ",optimizer[meilleurI],", ",batch_size[meilleurJ],", ",epochs[meilleurZ], ", ",meilleurA)
print("Loss : ", loss)


# In[ ]:




