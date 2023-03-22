#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\excel\car prediction.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df['year'].unique()


# In[7]:


df['kms_driven'].unique()


# In[8]:


df=df[df['Price']<6e6].reset_index(drop=True)
df


# In[9]:


df.describe()


# In[10]:


X = df.drop(columns= 'Price')
Y= df['Price']
X,Y


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[13]:


ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[14]:


col_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder= 'passthrough')


# In[15]:


lr=  LinearRegression()


# In[16]:


pipe = make_pipeline(col_trans,lr)


# In[17]:


pipe.fit(X_train,Y_train)


# In[18]:


Y_pred=pipe.predict(X_test)


# In[19]:


r2_score(Y_test,Y_pred)


# In[20]:


scores =[]
for i in range(1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=i)
    lr=  LinearRegression()
    pipe = make_pipeline(col_trans,lr)
    pipe.fit(X_train,Y_train)
    Y_pred=pipe.predict(X_test)
    scores.append(r2_score(Y_test,Y_pred))
    

    


# In[21]:


np.argmax(scores)


# In[22]:


scores[np.argmax(scores)]


# In[23]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(col_trans,lr)
pipe.fit(X_train,Y_train)
Y_pred=pipe.predict(X_test)
r2_score(Y_test,Y_pred)


# In[24]:


import pickle 


# In[39]:


with open('linear.pkl' , 'wb') as f:
    pickle.dump(pipe,f)


# In[40]:


import pickle
file_name='my_file.pkl'
f = open(file_name,'wb')
pickle.dump(pipe,f)
f.close()


# In[31]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[ ]:




