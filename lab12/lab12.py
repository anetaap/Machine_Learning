# -*- coding: utf-8 -*-
"""lab12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-1dlrM9lrs7oG-MWmc5El5Q2ztUV5NOV

## Analiza obrazów przy pomocy sieci konwolucyjnych

### 1. Zakres ćwiczeń
- Wykorzystanie konwolucyjnych sieci neuronowych (CNN) do analizy obrazu.
- Pobieranie gotowego modelu przy pomocy biblioteki Tensorflow Datasets.
- Przetwarzanie i udostępnianie danych przy pomocy Dataset API.
- Wykorzystanie gotowych modeli do uczenia transferowego.

### 2. Ćwiczenia
#### 2.1 Ładowanie danych
Do załadowania danych skorzystamy z pakietu Tensorflow Datasets, który udostępnia wiele zbiorów
przydatnych do uczenia maszynowego. Aby utrzymać względnie krótkie czasy uczenia, do ćwiczeń
będziemy używać zbioru `tf_flowers`:
"""

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import pickle

[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
"tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
as_supervised=True,
with_info=True)

info

"""- Możemy łatwo wyekstrahować istotne parametry zbioru:"""

class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples

"""- Wyświetlmy kilka przykładowych obrazów:"""

plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9)
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")

"""#### 2.2 Budujemy prostą sieć CNN
W tym ćwiczeniu zbudujemy sieć o nieskompikowanej strukturze.
##### 2.2.1 Przygotowanie danych
Sieć będzie przetwarzała obrazy o rozmiarze 224 × 224 pikseli, a więc pierwszym krokiem będzie
przetworzenie. Obiekty Dataset pozwalają na wykorzystanie metody map, która przy uczeniu
nadzorowanym będzie otrzymywała dwa argumenty (cechy, etykieta) i powinna zwracać je w postaci
krotki po przetworzeniu.

Najprostsza funkcja będzie po prostu skalowała obraz do pożądanego rozmiaru:
"""

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label

"""Aplikujemy ją do pobranych zbiorów:"""

batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

"""Wykorzystujemy tu dodatkowe metody Dataset API tak aby dostarczanie danych nie stało się wąskim gardłem procesu uczenia:
- `shuffle` losowo ustawia kolejność próbek w zbiorze uczącym,
- `batch` łączy próbki we wsady o podanej długości (idealnie, powinna to być wielkość miniwsadu podczas uczenia),
- `prefetch` zapewnia takie zarządzanie buforem, aby zawsze przygotowane było n próbek gotowych do pobrania (w tym przypadku chcemy, aby podczas przetwarzania miniwsadu przez algorytm uczenia zawsze czekał jeden przygotowany kolejny miniwsad).

Wyświetlmy próbkę danych po przetworzeniu:
"""

plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")

"""##### 2.2.2 Budowa sieci
Zaprojektuj prostą sieć konwolucyjną, która pozwoli na uzyskanie przyzwoitej dokładności klasyfikacji przetwarzanego zbioru.
Pamiętaj o istotnych zasadach:
1. W przypadku naszych danych, ponieważ składowe RGB pikseli mają wartości z zakresu 0–255, musimy pamiętać o normalizacji danych; można użyć do tego warstwy skalującej wartości.
2. Część wykrywająca elementy obrazu składa się z warstw konwolucyjnych, najczęściej przeplatanych warstwami zbierającymi:
     - głównymi parametrami warstw konwolucyjnych są liczba filtrów i rozmiar filtra; zazwyczaj zaczynamy od względnie niskiej liczby filtrów (np. 32) o większym rozmiarze (np. 7 × 7), aby wykryć elementarne komponenty obrazu, a na kolejnych warstwach łączymy je w bardziej złożone struktury – kombinacji jest więcej, a więc mamy coraz więcej filtrów, ale mogą być mniejszego rozmiaru (np. 3 × 3),
     - zwyczajowo na jedną warstwę konwolucyjną przypadała jedna warstwa zbierająca (zmniejszająca rozmiar „obrazu”), ale często stosujemy też kilka (np. 2) warstw konwolucyjnych bezpośrednio na sobie.
3. Po części konwolucyjnej typowo następuje część gęsta, złożona z warstw gęstych i opcjonalnie regularyzacyjnych (dropout?):
    - część gęsta musi być poprzedzona warstwą spłaszczającą dane, gdyż spodziewa się 1- wymiarowej struktury,
    - ostatnia warstwa musi być dostosowana do charakterystyki zbioru danych.

"""

from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3,
                        activation='relu',
                        padding="SAME")

train_set

n_classes

model = keras.models.Sequential([
    DefaultConv2D(filters=32, kernel_size=7,
                  input_shape=[224, 224, 3]),
    keras.layers.Rescaling(scale=1./127.5, offset=-1),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=5, activation='softmax')])

"""Skompiluj model z odpowiednimi parametrami, tak aby zbierana była metryka dotycząca dokładności."""

model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])

"""Przeprowadź uczenie przez 10 epok."""

history = model.fit(train_set, epochs=10, validation_data=valid_set)

"""Sprawdź jakość modelu. Model powinien zapewniać dokładność dla zbioru testowego co najmniej na poziomie ok. 60% – zbiór z kwiatami jest trudnym zadaniem dla tak prostej sieci."""

acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]

simple_cnn_acc = (acc_test, acc_valid, acc_test)

"""Zapisz wynik ewaluacji dla zbioru uczącego, walidacyjnego i testowego w postaci krotki
`(acc_train, acc_valid, acc_test)` do pikla simple_cnn_acc.pkl.
"""

with open('simple_cnn_acc.pkl', 'wb') as f:
    pickle.dump(simple_cnn_acc, f)

"""#### 2.3 Uczenie transferowe
Tym razem wykorzystamy gotową, dużo bardziej złożoną sieć. Dzięki temu, że sieć będzie zainicjalizowana wagami, możemy znacząco skrócić czas uczenia.

Jako bazową wykorzystamy względnie nowoczesną sieć Xception. Jest ona dostępna w pakiecie `tf.keras.applications.xception`.
Wykorzystamy wcześniej już załadowane surowe zbiory danych (..._set_raw).

##### 2.3.1 Przygotowanie danych
Gotowe modele często dostarczają własnych funkcji przygotowujących wejście w sposób zapewniający optymalne przetwarzanie. Musimy więc zmienić nieco funkcję przygotowującą dane, dodając
wywołanie odpowiedniej metody.
"""

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

"""Zobaczmy jak tym razem wyglądają wstępnie przetworzone dane; zwróć uwagę, że ponieważ teraz wartości należą już do zakresu (−1, 1), musimy je odpowiednio przeskalować (ale w sieci nie
będziemy potrzebowali warstwy skalującej):
"""

plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")

"""##### 2.3.2 Budowa sieci
Utwórz model bazowy przy pomocy odpowiedniej metody:
"""

batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

base_model = tf.keras.applications.xception.Xception(
    weights="imagenet",
    include_top=False)

"""Wyjaśnienie:
- argument `weights` zapewnia inicjalizację wag sieci wynikami uczenia zbiorem ImageNet,
- argument `include_top` sprawi, że sieć nie będzie posiadała górnych warstw (które musimy sami dodać, gdyż są specyficzne dla danego problemu).
Możesz wyświetlić strukturę załadowanej sieci:
"""

for index, layer in enumerate(base_model.layers):
    print(index, layer.name)

"""Korzystając z API funkcyjnego Keras dodaj warstwy:
- uśredniającą wartości wszystkich „pikseli”,
- wyjściową, gęstą, odpowiednią dla problemu.
"""

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)

"""Utwórz model korzystając z odpowiedniego konstruktora, podając jako wejścia (inputs) wejście modelu bazowego, a jako wyjścia – utworzoną warstwę wyjściową"""

model = keras.models.Model(inputs=base_model.input, outputs=output)

"""Przeprowadź uczenie w dwóch krokach:
1. Kilka (np. 5) iteracji, podczas których warstwy sieci bazowej będą zablokowane; ten krok jest konieczny aby zapobiec „zepsuciu” wag dostarczonych wraz z siecią bazową ze względu na spodziewane duże błędy wynikające z braku przyuczenia „nowych” warstw:
"""

for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=5)

"""2. Kolejne iteracje (np. 10), już z odblokowanymi do uczenia warstwami bazowymi"""

for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=10)

"""Aby ograniczyć czas uczenia (i oceniania), ustaw niezbyt wysoką liczbę iteracji w drugiej fazie.
Przeprowadź ewaluację modelu analogicznie do tej w poprzednim zadaniu.
"""

acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]

xception_acc = (acc_test, acc_valid, acc_test)

"""Zapisz wynik ewaluacji dla zbioru uczącego, walidacyjnego i testowego w postaci krotki
`(acc_train, acc_valid, acc_test)` do ***pikla xception_acc.pkl***.
"""

with open('xception_acc.pkl', 'wb') as f:
    pickle.dump(xception_acc, f)