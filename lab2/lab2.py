#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', version=1)


# In[3]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[4]:


X, y = mnist["data"], mnist["target"].astype(np.uint8)
y = y.sort_values()
X = X.reindex(y.index)
X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[5]:


y_train.unique()


# In[6]:


y_test.unique()


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[8]:


np.unique(y_train)


# In[9]:


np.unique(y_test)


# In[10]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))


# In[11]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42) 
sgd_clf.fit(X_train, y_train_0)


# In[12]:


print(sgd_clf.predict([mnist["data"].loc[0], mnist["data"].loc[1]]))


# In[13]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)

accuracy = [accuracy_score(y_train_0, y_train_pred), accuracy_score(y_test_0, y_test_pred)]


# In[14]:


print(accuracy)


# In[16]:


import pickle
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(accuracy, f)


# In[17]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_0,
                        cv=3, scoring="accuracy",
                        n_jobs=-1)
print(score)


# In[18]:


sgd_clf.predict(mnist["data"])


# In[19]:


with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(score, f)


# In[22]:


sgd_m_clf = SGDClassifier(random_state=42,n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)


# In[23]:


print(sgd_m_clf.predict([mnist["data"].loc[0], mnist["data"].loc[1]]))


# In[24]:


print(cross_val_score(sgd_m_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1))
y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)


# In[25]:


conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)


# In[26]:


with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(conf_mx, f)

