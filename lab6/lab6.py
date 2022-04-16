#!/usr/bin/env python
# coding: utf-8

# # 1. Cel / Zakres
#     - Metody zespołowe
#         - równoległe
#         - sekwencyjne
#     - Hard / soft voting
#     - Bagging
#     - Boosting

# # 2. Przygotowanie danych

# In[1]:


import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[2]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# # 3. Ćwiczenie:
#     - Uwaga: stosuj domyślne wartości parametrów dla użytych klas,
#       chyba, że z opisu danego ćwiczenia wynika inaczej.

# #### 1. Podział Zbioru
# Podziel zbiór data_breast_cancer na uczący i testujący w proporcjach 80:20.

# In[3]:


X_cancer = data_breast_cancer.data
y_cancer = data_breast_cancer.target


# In[4]:


X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)


# #### 2. Zbuduj ensemble
# Używając klasyfikatorów binarnych, których używałeś(aś) w poprzednich
# ćwiczeniach, tj.:
# 
#     - drzewa decyzyjne,
#     - regresja logistyczna,
#     - k najbliższych sąsiadów,
# do klasyfikacji w oparciu o cechy:
# 
#     - mean texture,
#     - mean symmetry.
# Użyj domyślnych parametrów.
# 

# In[5]:


X_cancer_train_texture_symmetry = X_cancer_train[["mean texture", "mean symmetry"]]
X_cancer_test_texture_symmetry = X_cancer_test[["mean texture", "mean symmetry"]]


# In[6]:


tree_clf = DecisionTreeClassifier(max_depth=3)
log_clf = LogisticRegression(solver="lbfgs")
knn_clf = KNeighborsClassifier()

voting_clf_hard = VotingClassifier(
    estimators=[('lr', log_clf),
                ('tr', tree_clf),
                ('knn', knn_clf)],
    voting='hard'
)

voting_clf_soft = VotingClassifier(
        estimators=[('lr', log_clf),
                ('tr', tree_clf),
                ('knn', knn_clf)],
    voting='soft'
)


# #### 3. Porównaj dokładność (accuracy)
# ww. klasyfikatorów z zespołem z głosowaniem typu:
#     
#     - hard
#     - soft.

# ***TRAIN SET***

# In[7]:


tree_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
log_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
knn_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
voting_clf_hard.fit(X_cancer_train_texture_symmetry, y_cancer_train)
voting_clf_soft.fit(X_cancer_train_texture_symmetry, y_cancer_train)


# In[8]:


y_train_tree_pred = tree_clf.predict(X_cancer_train_texture_symmetry)
y_train_log_pred = log_clf.predict(X_cancer_train_texture_symmetry)
y_train_knn_pred = knn_clf.predict(X_cancer_train_texture_symmetry)
y_train_vh_pred = voting_clf_hard.predict(X_cancer_train_texture_symmetry)
y_train_vs_pred = voting_clf_soft.predict(X_cancer_train_texture_symmetry)


# In[9]:


acc_tree_train = accuracy_score(y_cancer_train, y_train_tree_pred)
acc_log_train = accuracy_score(y_cancer_train, y_train_log_pred)
acc_knn_train = accuracy_score(y_cancer_train, y_train_knn_pred)
acc_vh_train = accuracy_score(y_cancer_train, y_train_vh_pred)
acc_vs_train = accuracy_score(y_cancer_train, y_train_vs_pred)


# In[10]:


print([acc_tree_train, acc_log_train, acc_knn_train, acc_vh_train, acc_vs_train])


# ***TEST SET***

# In[11]:


y_test_tree_pred = tree_clf.predict(X_cancer_test_texture_symmetry)
y_test_log_pred = log_clf.predict(X_cancer_test_texture_symmetry)
y_test_knn_pred = knn_clf.predict(X_cancer_test_texture_symmetry)
y_test_vh_pred = voting_clf_hard.predict(X_cancer_test_texture_symmetry)
y_test_vs_pred = voting_clf_soft.predict(X_cancer_test_texture_symmetry)


# In[12]:


acc_tree_test = accuracy_score(y_cancer_test, y_test_tree_pred)
acc_log_test = accuracy_score(y_cancer_test, y_test_log_pred)
acc_knn_test = accuracy_score(y_cancer_test, y_test_knn_pred)
acc_vh_test = accuracy_score(y_cancer_test, y_test_vh_pred)
acc_vs_test = accuracy_score(y_cancer_test, y_test_vs_pred)


# In[13]:


print([acc_tree_test, acc_log_test, acc_knn_test, acc_vh_test, acc_vs_test])


# #### 4. Zapisz rezultaty jako listę par
#     - (dokładność_dla_zb_uczącego, dokładność_dla_zb_testującego)
# dla każdego z w/w klasyfikatorów i umieść ją w pliku Pickle o nazwie:
#     
#     - acc_vote.pkl (razem 5 elementów)
# Zapisz klasyfikatory jako listę w pliku Pickle o nazwie:
#     
#     - vote.pkl (5 obiektów).

# In[14]:


accuracy =[(acc_tree_train, acc_tree_test),
           (acc_log_train, acc_log_test),
           (acc_knn_train, acc_log_test),
           (acc_vh_train, acc_vh_test),
           (acc_vs_train, acc_vs_test)]


# In[15]:


with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(accuracy, f)


# In[16]:


clf = [tree_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft]


# In[17]:


with open('vote.pkl', 'wb') as f:
    pickle.dump(clf, f)


# #### 5. Wykonaj na zbiorze uczącym 
#     - wykorzystując 30 drzew decyzyjnych:
# - Bagging,
# - Bagging z wykorzystaniem 50% instancji,
# - Pasting,
# - Pasting z wykorzystaniem 50% instancji, oraz
# - Random Forest,
# - AdaBoost,
# - Gradient Boosting.
# 
# ###### - Dlaczego Random Forest daje inne rezultaty niż Bagging + drzewa decyzyjne?

# In[18]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            max_samples=1.0, bootstrap=True)

bag_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                                 max_samples=0.5, bootstrap=True)

pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            max_samples=1.0, bootstrap=False)

pas_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            max_samples=0.5, bootstrap=False)

rnd_clf = RandomForestClassifier(n_estimators=30)

ada_clf = AdaBoostClassifier(n_estimators=30)

gb_clf = GradientBoostingClassifier(n_estimators=30)


# In[19]:


bag_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
bag_clf_half.fit(X_cancer_train_texture_symmetry, y_cancer_train)
pas_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
pas_clf_half.fit(X_cancer_train_texture_symmetry, y_cancer_train)
rnd_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
ada_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)
gb_clf.fit(X_cancer_train_texture_symmetry, y_cancer_train)


# #### 6. Oblicz dokładności oraz zapisz je jako listę par
#     -  (dokładność_dla_zb_uczącego, dokładność_dla_zb_testującego)
# dla każdego z ww. estymatorów w pliku Pickle o nazwie:
# 
#     - acc_bag.pkl (razem 7 elementów)
# Zapisz klasyfikatory jako listę w pliku Pickle o nazwie:
#     
#     - bag.pkl (razem 7 elementów)

# ***TRAIN SET***

# In[20]:


y_train_b_pred = bag_clf.predict(X_cancer_train_texture_symmetry)
y_train_bh_pred = bag_clf_half.predict(X_cancer_train_texture_symmetry)
y_train_p_pred = pas_clf.predict(X_cancer_train_texture_symmetry)
y_train_ph_pred = pas_clf_half.predict(X_cancer_train_texture_symmetry)
y_train_r_pred = rnd_clf.predict(X_cancer_train_texture_symmetry)
y_train_a_pred = ada_clf.predict(X_cancer_train_texture_symmetry)
y_train_g_pred = gb_clf.predict(X_cancer_train_texture_symmetry)


# In[21]:


acc_bag_train = accuracy_score(y_cancer_train, y_train_b_pred)
acc_bagh_train = accuracy_score(y_cancer_train, y_train_bh_pred)
acc_pas_train = accuracy_score(y_cancer_train, y_train_p_pred)
acc_pash_train = accuracy_score(y_cancer_train, y_train_ph_pred)
acc_rnd_train = accuracy_score(y_cancer_train, y_train_r_pred)
acc_ada_train = accuracy_score(y_cancer_train, y_train_a_pred)
acc_gb_train = accuracy_score(y_cancer_train, y_train_g_pred)


# ***TEST SET***

# In[22]:


y_test_b_pred = bag_clf.predict(X_cancer_test_texture_symmetry)
y_test_bh_pred = bag_clf_half.predict(X_cancer_test_texture_symmetry)
y_test_p_pred = pas_clf.predict(X_cancer_test_texture_symmetry)
y_test_ph_pred = pas_clf_half.predict(X_cancer_test_texture_symmetry)
y_test_r_pred = rnd_clf.predict(X_cancer_test_texture_symmetry)
y_test_a_pred = ada_clf.predict(X_cancer_test_texture_symmetry)
y_test_g_pred = gb_clf.predict(X_cancer_test_texture_symmetry)


# In[23]:


acc_bag_test = accuracy_score(y_cancer_test, y_test_b_pred)
acc_bagh_test = accuracy_score(y_cancer_test, y_test_bh_pred)
acc_pas_test = accuracy_score(y_cancer_test, y_test_p_pred)
acc_pash_test = accuracy_score(y_cancer_test, y_test_ph_pred)
acc_rnd_test = accuracy_score(y_cancer_test, y_test_r_pred)
acc_ada_test = accuracy_score(y_cancer_test, y_test_a_pred)
acc_gb_test = accuracy_score(y_cancer_test, y_test_g_pred)


# In[24]:


acc_bag = [(acc_bag_train, acc_bag_test),
       (acc_bagh_train, acc_bagh_test),
       (acc_pas_train, acc_pas_test),
       (acc_pash_train, acc_pash_test),
       (acc_rnd_train, acc_rnd_test),
       (acc_ada_train, acc_ada_test),
       (acc_gb_train, acc_gb_test)]


# In[25]:


with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(acc_bag, f)


# In[26]:


bag = [bag_clf, bag_clf_half, pas_clf, pas_clf_half, rnd_clf, ada_clf, gb_clf]


# In[27]:


with open('bag.pkl', 'wb') as f:
    pickle.dump(bag, f)


# #### 7. Przeprowadź sampling 2 cech z wszystkich dostepnych
#     - bez powtórzeń cech z wykorzystaniem 30 drzew decyzyjnych,
#     - wybierz połowę instancji dla każdego z drzew z powtórzeniami.
#     
# ###### Notatka:
# ###### Odnosnie zad 7: Tutaj BaggingClassifier z decision tree i max_features=2, Reszta ustawien domyslna trzeba uzyc

# In[28]:


bag_clf_s = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            max_samples=0.5, bootstrap=True, max_features=2, bootstrap_features=False)


# In[29]:


bag_clf_s.fit(X_cancer_train, y_cancer_train)


# In[30]:


y_train_bs_pred = bag_clf_s.predict(X_cancer_train)
y_test_bs_pred = bag_clf_s.predict(X_cancer_test)


# In[31]:


acc_bag_s_train = accuracy_score(y_cancer_train, y_train_bs_pred)
acc_bag_s_test = accuracy_score(y_cancer_test, y_test_bs_pred)


# #### 8. Zapisz dokładności ww estymatora jako listę :
#     - dokładność_dla_zb_uczącego, dokładność_dla_zb_testującego
# w pliku Pickle o nazwie:
# 
#     - acc_fea.pkl.
# Zapisz klasyfikator jako jednoelementową listę w pliku Pickle o nazwie:
# 
#     - fea.pkl

# In[32]:


acc_fea = [acc_bag_s_train, acc_bag_s_test]

with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(acc_fea, f)


# In[33]:


fea = [bag_clf_s]

with open('fea.pkl', 'wb') as f:
    pickle.dump(fea, f)


# #### 9. Sprawdź, które cechy dają najwięszą dokładność.
# Dostęp do poszczególnych estymatorów, aby obliczyć dokładność,
# możesz uzyskać za pmocą:
#     
#     - BaggingClasifier.estimators_.
# Cechy wybrane przez sampling dla każdego z estymatorów znajdziesz w:
#     
#     - BaggingClassifier.estimators_features_.
#     
# Zbuduj ranking estymatorów jako DataFrame, który będzie mieć w kolejnych kolumnach:
#     
#     - dokładność dla zb. uczącego,
#     - dokładnośc dla zb. testującego,
#     - lista nazw cech.
#     
#     - Każdy wiersz to informacje o jednym estymatorze.
#     - DataFrame posortuj malejąco po dokładności dla zbioru testującego i uczącego
# 
# Zapisz w pliku Pickle o nazwie:
#     
#     - acc_fea_rank.pkl

# In[34]:


fea_rank = []


# In[35]:


for estimator_, estimator_features_ in zip(bag_clf_s.estimators_, bag_clf_s.estimators_features_):
    y_train_fea_pred = estimator_.predict(X_cancer_train.iloc[:, estimator_features_])
    y_test_fea_pred = estimator_.predict(X_cancer_test.iloc[:, estimator_features_])

    acc_fea_train = accuracy_score(y_cancer_train, y_train_fea_pred)
    acc_fea_test = accuracy_score(y_cancer_test, y_test_fea_pred)

    rank = [acc_fea_train, acc_fea_test, list(X_cancer.columns[estimator_features_])]

    fea_rank.append(rank)


# In[36]:


acc_fea_rank = pd.DataFrame(fea_rank, columns=["accuracy train", "accuracy test", "features"])


# In[40]:


acc_fea_rank.sort_values(by=["accuracy test", "accuracy train"], ascending=False, inplace=True)


# In[41]:


acc_fea_rank


# In[42]:


with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(acc_fea_rank, f)


# ## 4. Prześlij raport
# Prześlij plik o nazwie lab6.py realizujący ww. ćwiczenia.
# 
# Sprawdzane będzie, czy skrypt Pythona tworzy wszystkie wymagane pliki oraz czy ich zawartość
# jest poprawna.
