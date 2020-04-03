
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


com = pd.read_csv('Computer_data.csv')


# In[3]:


com.shape


# In[4]:


com.describe()


# In[5]:


com.corr()


# In[6]:


sns.pairplot(com)


# In[7]:


com.isnull().any()


# In[8]:


com.columns


# In[9]:


com['multi'] = com['multi'].map({'yes': 1, 'no': 0})
print(com)
com['cd'] = com['cd'].map({'yes': 1, 'no': 0})
print(com)
com['premium'] = com['premium'].map({'yes': 1, 'no': 0})
print(com)


# In[10]:


X = com[['Unnamed: 0', 'speed', 'hd', 'ram', 'screen', 'cd', 'multi',
       'premium', 'ads', 'trend']].values.reshape(-1,10)
print(X)


# In[11]:


Y= com['price'].values.reshape(-1,1)
print(Y)


# In[12]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[13]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model = lm.fit(X_train,Y_train)


# In[14]:


print(lm.intercept_)


# In[15]:


print(lm.coef_)


# In[16]:


model.score(X_train,Y_train)  #R^2 value 


# In[17]:


predictions = model.predict(X_test)


# In[18]:


plt.scatter(Y_test,predictions)


# In[19]:


from sklearn import metrics


# In[20]:


print('MAE',metrics.mean_absolute_error(Y_test,predictions))


# In[21]:


print('MSE',metrics.mean_squared_error(Y_test,predictions))


# In[22]:


print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))


# In[23]:


import pylab


# In[24]:


Y1 = np.sin(X)
pylab.plot(X,Y1)
print(Y1)


# In[25]:


Y3 = np.log(X)
pylab.plot(X,Y3)
print(Y3)


# In[26]:


import statsmodels.api as sm
#4.Normal distribution of error terms:
model = sm.OLS(Y_train,X_train).fit()
res = model.resid #residuals
fig = sm.qqplot(res,fit=True,line='45')
plt.show()  #Q-Qplot for the advertising data set


# In[27]:


mod= sm.OLS(Y_train,X_train) #5.Little or No autocorrelation in the residuals:
results = mod.fit()
print(results.summary())  #Summary of the fitted Linear Model

