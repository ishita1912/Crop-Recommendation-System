#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Crop_recommendation.csv')
df.head()


# In[3]:


df.info()


# In[4]:


if df['N'].all()>90:
    print(df['N'])


# In[5]:


df.isnull().sum()


# In[8]:


df.tail()


# In[13]:


df.info()


# In[14]:


df['label'].value_counts()


# In[15]:


x = df.drop('label', axis = 1)
y = df['label']


# In[16]:


x.info()


# In[17]:


y.info()


# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 1, test_size = 0.2)


# In[19]:


x_train.info()


# In[20]:


x_test.info()


# In[21]:


y_train.info()


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[26]:


y_pred1 = model.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score
logistic_acc = accuracy_score(y_test, y_pred1)
print("Accuracy of logistic regression is " + str(logistic_acc))


# In[29]:


from sklearn.tree import DecisionTreeClassifier
model_2 = DecisionTreeClassifier()
model_2.fit(x_train, y_train)
y_pred2 = model_2.predict(x_test)


# In[30]:


decision_acc = accuracy_score(y_test, y_pred2)
print("Accuracy of decision  tree is " + str(decision_acc))


# In[35]:


from sklearn.ensemble import RandomForestClassifier
model_3 = RandomForestClassifier()
model_3.fit(x_train, y_train)
y_pred4 = model_3.predict(x_test)


# In[36]:


random_fore_acc = accuracy_score(y_test, y_pred4)
print("Accuracy of Random Forest is " + str(random_fore_acc))


# In[37]:


import joblib 


# In[44]:


file_name = 'crop_app'


# In[45]:


joblib.dump(model_3,'crop_app')


# In[46]:


app = joblib.load('crop_app')


# In[42]:


arr = [[90,42,43,20.879744,82.002744,6.502985,202.935536]]
y_pred5 = app.predict(arr)


# In[43]:


y_pred5


# In[ ]:




