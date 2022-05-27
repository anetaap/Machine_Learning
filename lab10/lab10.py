#!/usr/bin/env python
# coding: utf-8

# # <center> Laboratorium: Klasyfikacja i regresja w Keras </center>

# ### 1. Zakres ćwiczeń
# - Przypomnienie zasad budowy prostych sieci neuronowych dla klasyfikacji oraz regresji.
# - Budowanie sieci przy pomocy API Sequential Keras.
# - Zbieranie danych procesu uczenia i jego wizualizacja przy pomocy TensorBoard.
# - Korzystanie z mechanizmu early stopping do zatrzymania procesu uczenia w odpowiednim
# czasie.

# ### 2. Zadania
# ##### 2.1 Klasyfikacja obrazów
# - Pobierz zbiór danych _Fashion MNIST_.

# In[1]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# - Przeskaluj wartości z zakresu 0–255 do zakresu 0–1.

# In[4]:


X_train = X_train/255
X_test = X_test/255


# - Wyświetl przykładowy rysunek używany do klasyfikacji:

# In[5]:


plt.imshow(X_train[142], cmap="binary")
plt.axis('off')
plt.show()


# - Utwórz listę nazw kategorii zgodnie ze specyfikacją zbioru danych:

# In[6]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# - Stwórz model sekwencyjny zawierający warstwy gęste:
#     • warstwę spłaszczającą dane, tj. przekształcającą z postaci (28,28) do postaci (784,),
#     • warstwę ukrytą zawierającą 300 neuronów,
#     • warstwę ukrytą zawierającą 100 neuronów,
#     • warstwę wyjściową odpowiednią dla problemu klasyfikacji przy 10 klasach.

# In[7]:


model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))


# - Wyświetl podsumowanie i graficzną reprezentację struktury sieci.

# In[8]:


model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# - Skompiluj model, podając rzadką entropię krzyżową jako funkcję straty, SGD jako opymalizator i
# dokładność jako metrykę

# In[9]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics='accuracy')


# - Przygotuj callback Tensorboard do zbierania historii uczenia w katalogu image_logs, będącym
# podkatalogim bieżącego katalogu. Można do tego użyć takiej funkcji jak poniżej, która przy każdym
# wywołaniu będzie zwracała nowy katalog o nazwie opartej o aktualny czas. Alternatywnie możesz
# nazywać katalogi np. oznaczając w nazwie informacje o strukturze sieci. Pamiętaj, że TensorBoard
# zawsze szuka w podanym katalogu podkatalogów odpowiadających kolejnym procesom uczenia
# sieci.

# In[10]:


root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# - Przyucz model przez 20 epok. Wykorzystaj 10% zbioru uczącego jako zbioru walidacyjnego – zwróć
# uwagę, że nie został on wcześniej wydzielony ręcznie. Pamiętaj o dołączeniu callbacku TensorBoard.

# In[11]:


history = model.fit(X_train, y_train, epochs=20,
    validation_split = 0.1, callbacks=[tensorboard_cb])


# - Wykonaj kilka przykładowych predykcji dla losowych elementów zbioru testowego. Czy podpisy
# odpowiadają zawartości obrazków?

# In[12]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# - Przeanalizuj proces uczenia uruchamiając TensorBoard

# In[13]:


# %tensorboard --logdir ./image_logs


# - Zapisz model w pliku fashion_clf.h5:

# In[14]:


model.save('fashion_clf.h5')


# #### _2.2 Regresja_
# - Pobierz zbiór danych California Housing z pakietu scikit-learn:

# In[15]:


housing = fetch_california_housing()


# - Podziel zbiór na uczący, walidacyjny i testowy. Zauważ, że tym razem wydzielamy zbiór walidacyjny
# _explicite_ i nie będziemy korzystali z argumentu **validation_split** metody **fit**.

# In[16]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full, random_state=42)


# - Przeskaluj wszystkie zbiory cech, kalibrując funkcję normalizacyjną do zbioru uczącego:

# In[17]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# - Utwórz model, zawierający jedną warstwę ukrytą z 30 neuronami, odpowiedni do regresji jednego parametru wyjściowego obecnego w zbiorze danych.
# - Skompiluj go używając błędu średniokwadratwego jako funkcji straty i SGD jako optymalizatora.

# In[18]:


modelReg = keras.models.Sequential()

modelReg.add(keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]))
modelReg.add(keras.layers.Dense(1))


# In[19]:


modelReg.compile(loss="mean_squared_error", optimizer='sgd')


# - Przygotuj callback _early stopping_ o cierpliwości równej 5 epok, minimalnej wartości poprawy
# wynoszącej 0.01 i włączając wyświetlanie komunikatów o przerwaniu uczenia na ekranie:

# In[20]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# - Podobnie jak w poprzednim ćwiczeniu, przygotuj callback Tensorboard, tak aby zbierał logi do
# katalogu **housing_logs**.

# In[21]:


root_logdir_ = os.path.join(os.curdir, "housing_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir_, run_id)
run_logdir_ = get_run_logdir()
tensorboard_cb_ = tf.keras.callbacks.TensorBoard(run_logdir_)


# - Przeprowadź uczenie modelu korzystając z obu callbacków (_early stopping_ i TensorBoard). Jaką
# liczbę epok uczenia należy podać w tym przypadku?

# In[22]:


historyReg1 = modelReg.fit(X_train, y_train, epochs=30,
    validation_data=(X_valid, y_valid), callbacks=[es, tensorboard_cb_])


# In[23]:


modelReg.save('reg_housing_1.h5')


# In[24]:


modelReg2 = keras.models.Sequential()

modelReg2.add(keras.layers.Dense(50, activation="relu", input_shape=X_train.shape[1:]))
modelReg2.add(keras.layers.Dense(20, activation="tanh"))
modelReg2.add(keras.layers.Dense(10, activation="relu"))
modelReg2.add(keras.layers.Dense(1))


# In[25]:


modelReg2.compile(loss="mean_squared_error", optimizer='sgd')


# In[26]:


root_logdir_h2 = os.path.join(os.curdir, "./housing_logs/housing_logs_2")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir_h2, run_id)
run_logdir_ = get_run_logdir()
tensorboard_cb_ = tf.keras.callbacks.TensorBoard(run_logdir_)


# In[27]:


historyReg2 = modelReg2.fit(X_train, y_train, epochs=30,
    validation_data=(X_valid, y_valid), callbacks=[es, tensorboard_cb_])


# In[28]:


modelReg2.save('reg_housing_2.h5')


# In[29]:


modelReg3 = keras.models.Sequential()

modelReg3.add(keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]))
modelReg3.add(keras.layers.Dense(10, activation="sigmoid"))
modelReg3.add(keras.layers.Dense(1))


# In[30]:


modelReg3.compile(loss="mean_squared_error", optimizer='sgd')


# In[31]:


root_logdir_h3 = os.path.join(os.curdir, "./housing_logs/housing_logs_3")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir_h3, run_id)
run_logdir_ = get_run_logdir()
tensorboard_cb_ = tf.keras.callbacks.TensorBoard(run_logdir_)


# In[32]:


historyReg3 = modelReg3.fit(X_train, y_train, epochs=30,
    validation_data=(X_valid, y_valid), callbacks=[es, tensorboard_cb_])


# In[33]:


modelReg3.save('reg_housing_3.h5')

