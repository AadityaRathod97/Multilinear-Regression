
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# In[2]:


dataset = pd.read_csv('50_Startups.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.describe()


# In[5]:


dataset.corr()


# In[6]:


import seaborn as sns
sns.pairplot(dataset)


# In[7]:


dataset.isnull().any()


# In[8]:


dataset.columns


# In[9]:


X= dataset[['R&DSpend','Administration','MarketingSpend']].values.reshape(-1,3)
print(X)


# In[10]:


Y= dataset['Profit'].values.reshape(-1,1)
print(Y)


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[12]:


lm= LinearRegression()
lm.fit(X_train,Y_train)


# In[13]:


print(lm.coef_)


# In[14]:


Y_pred = lm.predict(X_test)


# In[15]:


print(lm.intercept_)


# In[16]:


plt.scatter(dataset['Profit'],dataset['R&DSpend'])


# In[17]:


dataset.corr()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4)


# In[20]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model = lm.fit(X_train,Y_train)


# In[21]:


print(lm.intercept_)


# In[22]:


print(lm.coef_)


# In[23]:


model.score(X_train,Y_train)  #R^2 value 


# In[24]:


predictions = model.predict(X_test)


# In[25]:


plt.scatter(Y_test,predictions)


# In[26]:


from sklearn import metrics


# In[27]:


print('MAE',metrics.mean_absolute_error(Y_test,predictions))


# In[28]:


print('MSE',metrics.mean_squared_error(Y_test,predictions))


# In[29]:


print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))


# In[30]:


import pylab


# In[31]:


Y1 = np.sin(X)
pylab.plot(X,Y1)
print(Y1)


# In[32]:


Y3 = np.log(X)
pylab.plot(X,Y3)
print(Y3)


# In[33]:


import statsmodels.api as sm
#4.Normal distribution of error terms:
model = sm.OLS(Y_train,X_train).fit()
res = model.resid #residuals
fig = sm.qqplot(res,fit=True,line='45')
plt.show()  #Q-Qplot for the advertising data set


# In[34]:


mod= sm.OLS(Y_train,X_train) #5.Little or No autocorrelation in the residuals:
results = mod.fit()
print(results.summary())  #Summary of the fitted Linear Model


# In[35]:


dataset.dropna()


# In[36]:


fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")   #influence plot


# In[37]:


fig, ax = plt.subplots(figsize=(12,8))  #Partial Regression Plots (Duncan)
fig = sm.graphics.plot_partregress("Profit", "MarketingSpend", ["Profit", "MarketingSpend"], data=dataset, ax=ax) 


# In[38]:


fig, ax = plt.subplots(figsize=(12,8))   #Partial Regression Plots (Duncan)
fig = sm.graphics.plot_partregress("Profit","Administration", ["Profit", "Administration"], data=dataset, ax=ax)


# In[39]:


fig = plt.figure(figsize=(12,8))  #Partial Regression Plot
fig = sm.graphics.plot_partregress_grid(model, fig=fig)


# In[40]:


fig = plt.figure(figsize=(12, 8))   #CCPR plot
fig = sm.graphics.plot_ccpr_grid(model, fig=fig)


# In[41]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[42]:


# Get variables for which to compute VIF and add intercept term
X = dataset[['R&DSpend','MarketingSpend','Administration']]
X['dataset'] = 1

# Compute and view VIF
#vif = pd.____
#vif["variables"] = X.____
#vif["VIF"] = [____(X.values, i) for i in range(X.shape[1])]

