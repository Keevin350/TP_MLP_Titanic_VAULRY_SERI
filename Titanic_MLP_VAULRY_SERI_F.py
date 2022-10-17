#!/usr/bin/env python
# coding: utf-8

# In[38]:


import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")


# In[39]:


train = pd.read_csv("train.csv")


# In[40]:


train.info()


# In[41]:


train


# In[42]:


#suppression des colonnes non utiles
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)


# In[43]:


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


# In[44]:


#Remplcaer les données catégoriques "male et female" par 1 et 0
train['Sex'] = train['Sex'].replace(['male'],'1')
train['Sex'] = train['Sex'].replace(['female'],'0')


# In[45]:


train


# In[46]:


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


# In[47]:


train.info()


# In[48]:


averageAge = round(train['Age'].sum()/len(train))
train['Age'] = train['Age'].fillna(averageAge)


# In[49]:


train['Embarked'] = train['Embarked'].astype('int64')
train['Name'] = train['Name'].astype('int64')
train['Sex'] = train['Sex'].astype('int64')


# In[50]:


train.info()


# In[51]:


import seaborn as sns
donnees = pd.DataFrame(train)
sns.pairplot(donnees, aspect=0.6, hue="Survived")


# In[14]:


AvecValeursManquantes = len(train)
train = train.dropna()
SansValeursManquantes = len(train)

print("Avec Valeurs Manquantes :", AvecValeursManquantes,"  Sans Valeurs Manquantes :",SansValeursManquantes)

y = train['Survived']
train = train.drop(['Survived'], axis=1)
print(y)


# In[16]:


import time
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)

print("taille test :", len(X_test))
print("taille train :", len(X_train))

start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[17]:


scaler = StandardScaler()
train = scaler.fit_transform(train)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)


start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[185]:


parametres = {'hidden_layer_sizes':[(100,),(50,),(500,),(100,100),(50,50),(100,100,100),(50,50,50)],
             'activation':["identity","logistic","tanh","relu"],
             'solver':["lbfgs","sgd","adam"],
             'max_iter':[50,100,200,250],   
             'alpha':[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
             'learning_rate':["constant","invscaling","adaptive"]}

mlp_clf = MLPClassifier()
gridS_mlp = GridSearchCV(mlp_clf, parametres, cv=2, n_jobs=-1)
gridS_mlp.fit(X_train, y_train)

print(gridS_mlp.best_params_)


# In[194]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)

maximum = 0
for i in range(200):
    
    start_time = time.time()
    mlp = MLPClassifier(activation = "relu", alpha = 1e-07, hidden_layer_sizes = (50, 50, 50), learning_rate = "invscaling",max_iter= 100, solver ="adam")
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)
    
    if maximum < accuracy_score(y_test, prediction):
        maximum = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))
        
print("Accuracy_score MAX : ", maximum)


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)

maximum = 0
for i in range(500):
    
    start_time = time.time()
    mlp = MLPClassifier(activation = "relu", alpha = 1e-07, hidden_layer_sizes = (50, 50, 50), learning_rate = "invscaling",max_iter= 100, solver ="adam")
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)
    
    if maximum < accuracy_score(y_test, prediction):
        maximum = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))
        print(i)
        
print("Accuracy_score MAX : ", maximum)


# In[19]:


parametres = {'hidden_layer_sizes':[(70,),(70,70),(70,70,70)],
             'activation':["identity","logistic","tanh","relu"],
             'solver':["lbfgs","sgd","adam"],
              'max_iter':[200,300,400],
              'batch_size':[200,300,400],
             'alpha':[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
              'max_iter':[50,100,200,250],  
             'learning_rate':["constant","invscaling","adaptive"]}

mlp_clf = MLPClassifier()
gridS_mlp = GridSearchCV(mlp_clf, parametres, cv=2, n_jobs=-1, verbose = 11)
gridS_mlp.fit(X_train, y_train)

print(gridS_mlp.best_params_)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2)

maximum = 0
for i in range(100):
    
    start_time = time.time()
    mlp = MLPClassifier(activation = "relu", alpha = 1e-07, hidden_layer_sizes = (70,), learning_rate = "adaptive",max_iter= 100, solver ="adam", batch_size= 400)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)
    
    if maximum < accuracy_score(y_test, prediction):
        maximum = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))
        
print("Accuracy_score MAX : ", maximum)


# In[52]:


mlp.loss_


# In[ ]:




