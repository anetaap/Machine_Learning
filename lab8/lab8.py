#!/usr/bin/env python
# coding: utf-8

# # 1. Cel/Zakres
# - Redukcja liczby wymiarów
# - Ocena efektów redukcji wymiarów

# # 2. Przygotowanie danych

# In[1]:


import pickle
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


data_breast_cancer = datasets.load_breast_cancer()
data_iris = load_iris()


# # 3. Ćwiczenie

# #### 1. Przeprowadź analizę PCA.
# Zredukuj liczbę wymiarów dla każdego z w/w zbiorów. Nowa przestrzeń ma pokrywać przynajmniej 90% różnorodności (zmienności) danych i ma mieć jak najmniej wymiarów.

# In[3]:


pca_dbc = PCA(n_components=0.9)
pca_di = PCA(n_components=0.9)

dbc_t = pca_dbc.fit_transform(data_breast_cancer.data)
di_t = pca_di.fit_transform(data_iris.data)


# In[4]:


print(pca_dbc.explained_variance_ratio_)
print(pca_di.explained_variance_ratio_)


# #### 2. Porównaj wyniki.
# - Ćwiczenia przeprowadź najpierw na oryginalnych danych, a później na danych przeskalowanych.
# 
# - W podanych zbiorach są istotnie różne zakresy dla poszczególnych cech. 
# - Aby je przeskalować aby były porównywalne użyj **StandardScaler()**. 
# - Klasa **PCA()** centruje dane automatycznie, ale ich nie skaluje!

# In[5]:


scaler = StandardScaler()

dbc_scaled = scaler.fit_transform(data_breast_cancer.data)
di_scaled =scaler.fit_transform(data_iris.data)

pca_dbc_scaled = PCA(n_components=0.9)
pca_di_scaled = PCA(n_components=0.9)

dbc_scaled_t = pca_dbc_scaled.fit_transform(dbc_scaled.data)
di_scaled_t = pca_di_scaled.fit_transform(di_scaled.data)


# In[6]:


print(list(pca_dbc_scaled.explained_variance_ratio_))
print(pca_di_scaled.explained_variance_ratio_)


# #### 3. Utwórz listę z współczynnikami zmienności nowych wymiarów <br/>(dla danych przeskalowanych).
# W przypadku data_breast_cancer listę zapisz w pliku Pickle o nazwie:
# - pca_bc.pkl
# 
# W przypadku data_iris listę zapisz w pliku Pickle o nazwie:
# - pca_ir.pkl

# In[7]:


pca_bc = list(pca_dbc_scaled.explained_variance_ratio_)
pca_ir = list(pca_di_scaled.explained_variance_ratio_)

with open("pca_bc.pkl", 'wb') as f:
    pickle.dump(pca_bc,f)

with open("pca_ir.pkl", 'wb') as f:
    pickle.dump(pca_ir, f)


# #### 4. Utwórz listę. 
# Indeksów cech (oryginalnych wymiarów), które mają największy udział w
# <br/>nowych cechach (wymiarach), po redukcji (dla danych przeskalowanych).
# 
# ###### Podpowiedź: zob. atrybut components_ klasy PCA.
# 
# W przypadku data_breast_cancer listę zapisz w pliku Pickle o nazwie:
# - idx_bc.pkl
# 
# W przypadku data_iris listę zapisz w pliku Pickle o nazwie:
# - idx_ir.pkl

# In[9]:


idx_bc = []

for row in pca_dbc_scaled.components_:
    idx = np.argmax(abs(row))
    idx_bc.append(idx)

print(idx_bc)


# In[10]:


idx_ir = []

for row in pca_di_scaled.components_:
    idx = np.argmax(abs(row))
    idx_ir.append(idx)

print(idx_ir)


# In[11]:


with open("idx_bc.pkl", 'wb') as f:
    pickle.dump(idx_bc, f)

with open("idx_ir.pkl", 'wb') as f:
    pickle.dump(idx_ir, f)


# # 4 Prześlij raport
# Prześlij plik o nazwie **lab8.py** realizujący ww. ćwiczenia.
# <br/>Sprawdzane będzie, czy skrypt Pythona tworzy wszystkie wymagane pliki,<br/> oraz czy ich zawartość
# jest poprawna.
# 
