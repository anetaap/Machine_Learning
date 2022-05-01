"""lab7.ipynb

# 1. Cel/ Zakres
- Klasteryzacja
- Znajdywanie parametrów dla algorytmów klasteryzacji.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
import numpy as np
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import pickle

"""# 2. Przygotowanie Danych"""

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

"""# 3. Ćwiczenie

Pozyskane dane (zmienna X) reprezentują zeskanowane znaki nieznanego alfabetu.

Celem ćwiczenia jest identyfikacja ile tych znaków jest i jak mogą one wyglądać.

Zakładając, że możemy mieć do czynienia z 8–12 różnymi znakami uzyj metody centroidów do ich
klasteryzacji.

### 1. Przeprowadź klasteryzację dla 8, 9, 10, 11 i 12 skupisk.
"""

clusters = [8, 9, 10, 11, 12]
silhouette_scores = [] 

for k in clusters:
    kmeans = KMeans(n_clusters = k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

"""### 2. Wylicz wartośc wskaźnika sylwetkowego dla każdego z ww. skupisk.
Zapisz wartość wszystkich wskaźników sylwetkowych jako listę w pliku Pickle o nazwie:
- kmeans_sil.pkl.
"""

with open("kmeans_sil.pkl", 'wb') as f:
    pickle.dump(silhouette_scores, f)

"""### 3. Wartości wskaźnika sylwetkowego.
Znany lingwista prof. Talent twierdzi, że w zbiorze X można zidentyfikować 10 różnych
znaków. <br /> Czy wartości wskaźnika sylwetkowego potwierdzają tą obserwację?

### 4. Policz macierz błędów.
Prof. Talent dostarczył swoich wyników klasyfikacji w postaci zbioru y. Policz macierz błędów
pomiędzy danymi otrzymanymi z procesu klasteryazacji dla 10 skupisk i zbioru y.
"""

kmeans_10 = KMeans(n_clusters = 10, random_state=42)
y_pred10 = kmeans_10.fit_predict(X)

conf_matrix10 = confusion_matrix(y, y_pred10)

conf_matrix10

"""### 5. Dla każdego wiersza ww. macierzy znajdź indeks o najwyższej wartości
(np. numpy.argmax() albo pandas.Series.argmax()).<br/> Wartości umieść na posortowanej rosnąco liście bez duplikatów (użyj np. set()). <br/>Listę zapisz w pliku Pickle o nazwie:
- kmeans_argmax.pkl.
"""

kmeans_argmax = []

for row in conf_matrix10:
    index = np.argmax(row)
    if index not in kmeans_argmax:
        kmeans_argmax.append(index)

kmeans_argmax = np.sort(kmeans_argmax)

with open("kmeans_argmax.pkl", 'wb') as f:
    pickle.dump(kmeans_argmax, f)

"""### 6. Znajdź sensowne wartości parametru eps dla DBSCAN.
Heurystyka dla określenia wartości parametru eps oparta jest o odległość eulidesową pomiędzy instancjami. Policz odległości dla pierwszych 300 elementów ze zbioru X. <br/>(użyj np. numpy.linalg.norm(x1-x2), gdzie x1 i x2 to punkty w przestrzeni wielowymiarowej), <br/>a następnie wyświetl 10 najmniejszych. Ww.<br/><br/>
10 wartości umieść na liście w kolejności rosnącej, a listę zapisz w pliku Pickle o nazwie:
- dist.pkl.
"""

norms = []

for i in range(300):
    for j in range(len(X)):
        if i == j:
            continue
        norm = np.linalg.norm(X[i] - X[j]) 
        norms.append(norm)

norm_sorted = np.unique(np.sort(norms))
dist = norm_sorted[:10]

with open("dist.pkl", 'wb') as f:
    pickle.dump(dist, f)

"""### 7. Policz średnią s z 3 najmniejszych wartości z ww. listy.
Przyjmij kolejno wartości eps od s do s+10%*s z krokiem co 4%*s i wykonaj klasteryzacje.

"""

s = np.mean(dist[:3])

"""### 8. Dla każdej klasteryzacji (dla kolejnych wartości eps) policz ile jest unikalnych etykie zidentyfikowanych przez algorytm DBSCAN.
Wartości umieść na liście i zapisz w pliku Pickle o nazwie:
- dbscan_len.pkl.

"""

eps_values = np.arange(s, 1.1*s, step=0.04*s)
dbscan_len = []

for e in eps_values:
    dbscan = DBSCAN(eps=e)
    dbscan.fit(X)
    unique_labels = set(dbscan.labels_)
    dbscan_len.append(len(unique_labels))

dbscan_len

dbscan_len = "dbscan_len.pkl"

with open(dbscan_len, 'wb') as f:
    pickle.dump(dbscan_len, f)

"""# 4. Prześlij raport

Prześlij plik o nazwie:
- lab7.py
realizujący ww. ćwiczenia.

Sprawdzane będzie, czy skrypt Pythona tworzy wszystkie wymagane pliki oraz czy ich zawartość jest poprawna.
"""