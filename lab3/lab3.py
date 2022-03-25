#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y}) 
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[2]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # ***TRAIN SET***

# ***Linear Regression Train set***
# 

# In[4]:


X_b = np.c_[np.ones(X_train.shape), X_train]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
print(theta_best)


# In[5]:


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
print(y_predict := X_new_b.dot(theta_best))


# In[6]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
X_train = X_train.reshape(-1, 1)
lin_reg.fit(X_train, y_train)
print(lin_reg.intercept_, lin_reg.coef_, "\n", lin_reg.predict(X_new))
lin_reg_pred = lin_reg.predict(X_train)


# ***KNN Train set k=3***

# In[7]:


import sklearn.neighbors
knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train.reshape(-1,1), y_train)
print(knn_3_reg.predict(X_new))
knn_3_reg_pred = knn_3_reg.predict(X_train)


# ***k=5***

# In[8]:


knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train.reshape(-1,1), y_train)
print(knn_5_reg.predict(X_new))
knn_5_reg_pred = knn_5_reg.predict(X_train)


# ***Polynomial Regression Train set***

# In[9]:


from sklearn.preprocessing import PolynomialFeatures
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features_2.fit_transform(X_train)
print(X_train[0], X_poly[0])
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_poly, y_train)
print(poly_2_reg.intercept_, poly_2_reg.coef_)
print(poly_2_reg.predict(poly_features_2.fit_transform([[0],[2]])))
print(poly_2_reg.coef_[1] * 2**2 + poly_2_reg.coef_[0] * 2 + poly_2_reg.intercept_)
poly_2_reg_pred = poly_2_reg.predict(X_poly)


# In[10]:


poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features_3.fit_transform(X_train)
print(X_train[0], X_poly[0])
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_poly, y_train)
print(poly_3_reg.intercept_, poly_3_reg.coef_)
print(poly_3_reg.predict(poly_features_3.fit_transform([[0],[2]])))
print(poly_3_reg.coef_[2] * 2**3 + poly_3_reg.coef_[1] * 2**2 + poly_3_reg.coef_[0] * 2 + poly_3_reg.intercept_)
poly_3_reg_pred = poly_3_reg.predict(X_poly)


# In[11]:


poly_features_4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features_4.fit_transform(X_train)
print(X_train[0], X_poly[0])
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_poly, y_train)
print(poly_4_reg.intercept_, poly_4_reg.coef_)
print(poly_4_reg.predict(poly_features_4.fit_transform([[0],[2]])))
print(poly_4_reg.coef_[3] * 2**4 + poly_4_reg.coef_[2] * 2**3 + poly_4_reg.coef_[1] * 2**2 + poly_4_reg.coef_[0] * 2 + poly_4_reg.intercept_)
poly_4_reg_pred = poly_4_reg.predict(X_poly)


# In[12]:


poly_features_5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly_features_5.fit_transform(X_train)
print(X_train[0], X_poly[0])
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_poly, y_train)
print(poly_5_reg.intercept_, poly_5_reg.coef_)
print(poly_5_reg.predict(poly_features_5.fit_transform([[0],[2]])))
print(poly_5_reg.coef_[4] *2**5 + poly_5_reg.coef_[3] * 2**4 + poly_5_reg.coef_[2] * 2**3 + poly_5_reg.coef_[1] * 2**2 + poly_5_reg.coef_[0] * 2 + poly_5_reg.intercept_)
poly_5_reg_pred = poly_5_reg.predict(X_poly)


# # ***TEST SET***

# ***Linear Regression Test set***

# In[13]:


X_b = np.c_[np.ones(X_test.shape), X_test]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_test)
print(theta_best)


# In[14]:


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
print(y_predict_ := X_new_b.dot(theta_best))


# In[15]:


lin_reg = LinearRegression()
X_test = X_test.reshape(-1, 1)
lin_reg.fit(X_test, y_test)
lin_reg.fit(X_test, y_test)
print(lin_reg.intercept_, lin_reg.coef_, "\n", lin_reg.predict(X_new))
lin_reg_pred_t = lin_reg.predict(X_test)


# ***KNN Test set k=3***

# In[16]:


knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_test.reshape(-1,1), y_test)
print(knn_3_reg.predict(X_new))
knn_3_reg_pred_t = knn_3_reg.predict(X_test)


# ***k=5***

# In[17]:


knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_test.reshape(-1,1), y_test)
print(knn_5_reg.predict(X_new))
knn_5_reg_pred_t = knn_5_reg.predict(X_test)


# ***Polynomial Regression Test set***

# In[18]:


from sklearn.preprocessing import PolynomialFeatures
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features_2.fit_transform(X_test.reshape(-1,1))
print(X_test[0], X_poly[0])
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_poly, y_test)
print(poly_2_reg.intercept_, poly_2_reg.coef_)
print(poly_2_reg.predict(poly_features_2.fit_transform([[0],[2]])))
print(poly_2_reg.coef_[1] * 2**2 + poly_2_reg.coef_[0] * 2 + poly_2_reg.intercept_)
poly_2_reg_pred_t = poly_2_reg.predict(X_poly)


# In[19]:


poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features_3.fit_transform(X_test.reshape(-1,1))
print(X_test[0], X_poly[0])
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_poly, y_test)
print(poly_3_reg.intercept_, poly_3_reg.coef_)
print(poly_3_reg.predict(poly_features_3.fit_transform([[0],[2]])))
print(poly_3_reg.coef_[2] * 2**3 + poly_3_reg.coef_[1] * 2**2 + poly_3_reg.coef_[0] * 2 + poly_3_reg.intercept_)
poly_3_reg_pred_t = poly_3_reg.predict(X_poly)


# In[20]:


poly_features = PolynomialFeatures(degree=4, include_bias = False)
X_poly = poly_features.fit_transform(X_test.reshape(-1,1))
print(X_test[0].reshape(-1, 1), X_poly[0])
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_poly, y_test)
print(poly_4_reg.intercept_, poly_4_reg.coef_)
print(poly_4_reg.predict(poly_features.fit_transform([[0],[2]])))
print(poly_4_reg.coef_[1] * 2**2 + poly_4_reg.coef_[0] * 2 + poly_4_reg.intercept_)
poly_4_reg_pred_t = poly_4_reg.predict(X_poly)


# In[21]:


poly_features = PolynomialFeatures(degree=5, include_bias = False)
X_poly = poly_features.fit_transform(X_test.reshape(-1,1))
print(X_test[0].reshape(-1, 1), X_poly[0])
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_poly, y_test)
print(poly_5_reg.intercept_, poly_5_reg.coef_)
print(poly_5_reg.predict(poly_features.fit_transform([[0],[2]])))
print(poly_5_reg.coef_[1] * 2**2 + poly_5_reg.coef_[0] * 2 + poly_5_reg.intercept_)
poly_5_reg_pred_t = poly_5_reg.predict(X_poly)


# ***DATA FRAME***

# In[22]:


from sklearn.metrics import mean_squared_error
mse = {'lin_reg':[mean_squared_error(y_train, lin_reg_pred), mean_squared_error(y_test, lin_reg_pred_t)],
       'knn_3_reg':[mean_squared_error(y_train, knn_3_reg_pred), mean_squared_error(y_test, knn_3_reg_pred_t)],
       'knn_5_reg':[mean_squared_error(y_train, knn_5_reg_pred), mean_squared_error(y_test, knn_5_reg_pred_t)],
       'poly_2_reg':[mean_squared_error(y_train, poly_2_reg_pred), mean_squared_error(y_test, poly_2_reg_pred_t)],
       'poly_3_reg':[mean_squared_error(y_train, poly_3_reg_pred), mean_squared_error(y_test, poly_3_reg_pred_t)],
       'poly_4_reg':[mean_squared_error(y_train, poly_4_reg_pred), mean_squared_error(y_test, poly_4_reg_pred_t)],
       'poly_5_reg':[mean_squared_error(y_train, poly_5_reg_pred), mean_squared_error(y_test, poly_5_reg_pred_t)]}

mse = pd.DataFrame.from_dict(mse, orient = 'index', columns = ['train_mse', 'test_mse'])


# In[23]:


mse


# In[24]:


import pickle


# In[25]:


with open('mse.pkl', 'wb') as f:
    pickle.dump(mse, f)


# In[26]:


reg = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (poly_2_reg, poly_features_2 ), (poly_3_reg, poly_features_3), (poly_4_reg, poly_features_4), (poly_5_reg, poly_features_5)]
with open('reg.pkl', 'wb') as f:
    pickle.dump(reg, f)


# In[29]:


reg

