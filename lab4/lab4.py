#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.metrics import accuracy_score


# In[4]:


import pickle


# # ***Cancer dataset***

# _Data preparation_

# In[5]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[6]:


X_cancer = data_breast_cancer.data
y_cancer = data_breast_cancer.target


# In[7]:


print(X_cancer.size, y_cancer.size)


# In[8]:


X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)


# In[9]:


print(X_cancer_train.size,X_cancer_test.size )


# _SVM Classification_

# In[10]:


X_cancer_train_area_smooth = X_cancer_train[["mean area", "mean smoothness"]]
X_cancer_test_area_smooth = X_cancer_test[["mean area", "mean smoothness"]]


# In[11]:


svm_clf = Pipeline([("linear_svc", LinearSVC(C=1,  loss="hinge", random_state=42)),])


# In[12]:


svm_clf_scaler = Pipeline([("scaler", StandardScaler()),
                            ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
                            ])


# In[13]:


svm_clf.fit(X_cancer_train_area_smooth, y_cancer_train)
svm_clf_scaler.fit(X_cancer_train_area_smooth, y_cancer_train)


# In[14]:


y_cancer_train_svm_pred = svm_clf.predict(X_cancer_train_area_smooth)
y_cancer_train_svm_scal_pred = svm_clf_scaler.predict(X_cancer_train_area_smooth)


# In[15]:


y_cancer_test_svm_pred = svm_clf.predict(X_cancer_test_area_smooth)
y_cancer_test_svm_scal_pred = svm_clf_scaler.predict(X_cancer_test_area_smooth)


# In[16]:


acc_svm_clf_train = accuracy_score(y_cancer_train, y_cancer_train_svm_pred)
acc_svm_clf_scaler_train = accuracy_score(y_cancer_train, y_cancer_train_svm_scal_pred)


# In[17]:


acc_svm_clf_test = accuracy_score(y_cancer_test, y_cancer_test_svm_pred)
acc_svm_clf_scaler_test = accuracy_score(y_cancer_test, y_cancer_test_svm_scal_pred)


# In[18]:


bc_acc = [acc_svm_clf_train, acc_svm_clf_test, acc_svm_clf_scaler_train, acc_svm_clf_scaler_test]


# In[19]:


print(bc_acc)


# In[20]:


with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(bc_acc, f)


# # ***Iris dataset***

# _Data preparation_

# In[21]:


data_iris = datasets.load_iris(as_frame=True)


# In[22]:


print(data_iris['DESCR'])


# In[23]:


X_iris = data_iris.data
y_iris = (data_iris.target == 2).astype(np.int8)


# In[24]:


X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)


# In[25]:


print(X_iris_train.size, X_iris_test.size)


# _SVM Classification_

# In[26]:


X_iris_train_length_width = X_iris_train[["petal length (cm)", "petal width (cm)"]]
X_iris_test_length_width = X_iris_test[["petal length (cm)", "petal width (cm)"]]


# In[27]:


svm_clf_iris = Pipeline([("linear_svc", LinearSVC(C=1,  loss="hinge", random_state=42)),])


# In[28]:


svm_clf_scaler_iris = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                 random_state=42)),])


# In[29]:


svm_clf_iris.fit(X_iris_train_length_width, y_iris_train)
svm_clf_scaler_iris.fit(X_iris_train_length_width, y_iris_train)


# In[30]:


y_iris_train_svm_pred = svm_clf_iris.predict(X_iris_train_length_width)
y_iris_train_svm_scal_pred = svm_clf_scaler_iris.predict(X_iris_train_length_width)


# In[31]:


y_iris_test_svm_pred = svm_clf_iris.predict(X_iris_test_length_width)
y_iris_test_svm_scal_pred = svm_clf_scaler_iris.predict(X_iris_test_length_width)


# In[32]:


acc_svm_clf_iris_train = accuracy_score(y_iris_train, y_iris_train_svm_pred)
acc_svm_clf_scaler_iris_train = accuracy_score(y_iris_train, y_iris_train_svm_scal_pred)


# In[33]:


acc_svm_clf_iris_test = accuracy_score(y_iris_test, y_iris_test_svm_pred)
acc_svm_clf_scaler_iris_test = accuracy_score(y_iris_test, y_iris_test_svm_scal_pred)


# In[34]:


iris_acc = [acc_svm_clf_iris_train, acc_svm_clf_iris_test, acc_svm_clf_scaler_iris_train, acc_svm_clf_scaler_iris_test]


# In[35]:


print(iris_acc)


# In[36]:


with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(iris_acc, f)

