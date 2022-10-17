#!/usr/bin/env python
# coding: utf-8

# In[609]:


import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")


# In[610]:


train = pd.read_csv("train.csv")


# In[611]:


#suppression des colonnes non utiles
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)


# In[612]:


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


# In[613]:


#Remplcaer les données catégoriques "male et female" par 1 et 0
train['Sex'] = train['Sex'].replace(['male'],'1')
train['Sex'] = train['Sex'].replace(['female'],'0')


# In[614]:


train.info()


# In[615]:


#conversion colonne Name
#le but de la boucle est de récupérer dans des tableaux les ages en fonction de titre
#afin de créer une moyenne

ageMr = []
ageMiss = []
ageMrs = []
ageMaster = []
ageDr = []
ageRev = []
ageDon = []
ageMme = []
ageMs = []
ageMajor = []
ageLady = []
ageSir = []
ageMlle = []
ageCol = []
ageCapt = []
ageCountess = []
ageJonkheer = []

newV = train['Name'].copy()
liste = []

for i in range(len(newV)):
    valeur = newV[i]
    chaine = re.findall(" ([A-Za-z]+)\.", valeur)[0]
    
    if chaine not in liste:
        liste.append(chaine)
    
    if chaine == "Mr":
        newV[i] = 1
        if not pd.isnull(train.iloc[i,4]):  
            ageMr.append(train.iloc[i,4])
            
    elif chaine == "Miss":
        newV[i] = 2
        if not pd.isnull(train.iloc[i,4]):  
            ageMiss.append(train.iloc[i,4])
            
    elif chaine == "Mrs":
        newV[i] = 3
        if not pd.isnull(train.iloc[i,4]):  
            ageMrs.append(train.iloc[i,4])
            
    elif chaine == "Master":
        newV[i] = 4
        if not pd.isnull(train.iloc[i,4]):  
            ageMaster.append(train.iloc[i,4])
            
    elif chaine == "Dr":
        newV[i] = 5
        if not pd.isnull(train.iloc[i,4]):  
            ageDr.append(train.iloc[i,4])
        
    elif chaine == "Rev":
        newV[i] = 6
        if not pd.isnull(train.iloc[i,4]):  
            ageRev.append(train.iloc[i,4])
            
    elif chaine == "Don":
        newV[i] = 7
        if not pd.isnull(train.iloc[i,4]):  
            ageDon.append(train.iloc[i,4])
            
    elif chaine == "Mme":
        newV[i] = 8
        if not pd.isnull(train.iloc[i,4]):  
            ageMme.append(train.iloc[i,4])
            
    elif chaine == "Ms":
        newV[i] = 9
        if not pd.isnull(train.iloc[i,4]):  
            ageMs.append(train.iloc[i,4])
            
    elif chaine == "Major":
        newV[i] = 10
        if not pd.isnull(train.iloc[i,4]):  
            ageMajor.append(train.iloc[i,4])
            
    elif chaine == "Lady":
        newV[i] = 11
        if not pd.isnull(train.iloc[i,4]):  
            ageLady.append(train.iloc[i,4])
            
    elif chaine == "Sir":
        newV[i] = 12
        if not pd.isnull(train.iloc[i,4]):  
            ageSir.append(train.iloc[i,4])
            
    elif chaine == "Mlle":
        newV[i] = 13
        if not pd.isnull(train.iloc[i,4]):  
            ageMlle.append(train.iloc[i,4])
            
    elif chaine == "Col":
        newV[i] = 14
        if not pd.isnull(train.iloc[i,4]):  
            ageCol.append(train.iloc[i,4])
            
    elif chaine == "Capt":
        newV[i] = 15
        if not pd.isnull(train.iloc[i,4]):  
            ageCapt.append(train.iloc[i,4])
            
    elif chaine == "Countess":
        newV[i] = 16
        if not pd.isnull(train.iloc[i,4]):  
            ageCountess.append(train.iloc[i,4])
            
    elif chaine == "Jonkheer":
        newV[i] = 17
        if not pd.isnull(train.iloc[i,4]):  
            ageJonkheer.append(train.iloc[i,4])
    else:
        newV[i] = 18
        

train['Name'] = newV 
print(liste)


# In[616]:


train.info()
print(train.iloc[0,3])


# In[617]:


train


# In[583]:


#pour les données ayant l'age manquant, on attribut la moyenne obtenue
for i in range(len(train)):
    
    
    if train.iloc[i,2] == 1  and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMr) / len(ageMr)
        
    elif train.iloc[i,2] == 2  and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMiss) / len(ageMiss)
        
    elif train.iloc[i,2] == 3 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMrs) / len(ageMrs)
        
    elif train.iloc[i,2] == 4 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMaster) / len(ageMaster)
        
    elif train.iloc[i,2] == 5 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageDr) / len(ageDr)
        
    elif train.iloc[i,2] == 6 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageRev) / len(ageRev)
        
    elif train.iloc[i,2] == 7 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageDon) / len(ageDon)
        
    elif train.iloc[i,2] == 8 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMme) / len(ageMme)
        
    elif train.iloc[i,2] == 9 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMs) / len(ageMs)
        
    elif train.iloc[i,2] == 10 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMajor) / len(ageMajor)
        
    elif train.iloc[i,2] == 11 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageLady) / len(ageLady)
        
    elif train.iloc[i,2] == 12 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageSir) / len(ageSir)
        
    elif train.iloc[i,2] == 13 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageMlle) / len(ageMlle)
        
    elif train.iloc[i,2] == 14 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageCol) / len(ageCol)
        
    elif train.iloc[i,2] == 15 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageCol) / len(ageCol)    
        
    elif train.iloc[i,2] == 16 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageCountess) / len(ageCountess)
        
    elif train.iloc[i,2] == 17 and pd.isnull(train.iloc[i,4]):
        train.iloc[i,4] = sum(ageJonkheer) / len(ageJonkheer)


# In[618]:


train.info()


# In[619]:


train['Embarked'] = train['Embarked'].astype('int64')
train['Name'] = train['Name'].astype('int64')
train['Sex'] = train['Sex'].astype('int64')


# In[ ]:


len(train)


# In[620]:


AvecValeursManquantes = len(train)
train = train.dropna()
SansValeursManquantes = len(train)

print("Avec Valeurs Manquantes :", AvecValeursManquantes,"  Sans Valeurs Manquantes :",SansValeursManquantes)

y = train['Survived']
train = train.drop(['Survived'], axis=1)
print(y)


# In[621]:


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


# In[622]:


scaler = StandardScaler()
train = scaler.fit_transform(train)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)


start_time = time.time()
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print(accuracy_score(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


# In[623]:


parametres = {'hidden_layer_sizes':[(100,),(100,100),(100,100,100),(100,100,100,100)],
             'activation':["identity","logistic","tanh","relu"],
             'solver':["lbfgs","sgd","adam"],
             'batch_size':[50,200,300],
             'max_iter':[100,200,300],
             'alpha':[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
             'learning_rate':["constant","invscaling","adaptive"]}

mlp_clf = MLPClassifier()
gridS_mlp = GridSearchCV(mlp_clf, parametres, cv=2, n_jobs=-1, verbose = 11)
gridS_mlp.fit(X_train, y_train)

print(gridS_mlp.best_params_)


# In[592]:


print(gridS_mlp.best_params_)


# In[624]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2)

maximum = 0
for i in range(50):

    start_time = time.time()
    mlp = MLPClassifier(activation = "relu", alpha = 1e-06, batch_size = 300, hidden_layer_sizes = (100, 100,100,100), learning_rate = "constant", max_iter = 300, solver ="sgd")
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)

    if maximum < accuracy_score(y_test, prediction):
        maximum = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))

print("Accuracy_score MAX : ", maximum)


# In[627]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2)

maximum = 0
for i in range(150):

    start_time = time.time()
    mlp = MLPClassifier(activation = "relu", alpha = 1e-06, batch_size = 300, hidden_layer_sizes = (100, 100,100,100), learning_rate = "constant", max_iter = 300, solver ="sgd")
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)

    if maximum < accuracy_score(y_test, prediction):
        maximum = accuracy_score(y_test, prediction)
        print(accuracy_score(y_test, prediction))
        print("--- %s seconds ---" % (time.time() - start_time))
        print(i)

print("Accuracy_score MAX : ", maximum)


# In[ ]:




