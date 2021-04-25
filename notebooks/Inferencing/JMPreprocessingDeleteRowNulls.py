#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('../../data/infdata_stage1.csv')
pd.set_option('display.max_rows', 200)
len(df.index)


# In[3]:


#removing fights before 2009
# df.drop(df.index[df['location'] < 2009], inplace = True)
# len(df.index)


# In[4]:


#removing 5 round fights
# df.drop(df.index[df['title_bout'] == True], inplace = True)
# len(df.index)


# In[5]:


#removing instances of draws
df.drop(df.index[df['Winner'] == 'Draw'], inplace = True)
len(df.index)


# In[6]:


#df.drop(columns=['R_age', 'B_age', 'date'], inplace=True)


# In[7]:


##rather than filling null values with the median, we will delete all rows that have a null
NullList = df.isnull().any(axis=1).tolist()
IndexList = [i for i, x in enumerate(NullList) if x]
IndexList


# In[8]:


df.drop(df.index[IndexList], inplace=True)


# In[9]:


NullList = df.isnull().any(axis=1).tolist()
NullList


# In[10]:


#one hot encoding categorical variables
#df = pd.concat([df, pd.get_dummies(df[['weight_class', 'B_Stance', 'R_Stance']])], axis=1)
#df.drop(columns=['weight_class', 'B_Stance', 'R_Stance'], inplace=True)


# In[11]:


#converting classifications to numerical value. Blue is 1, red is 0
df['Winner'] = df['Winner'].map({'Blue': 1, 'Red': 0})


# In[14]:


#dropping unuseful features
#df.drop(columns=['Referee','location','R_fighter', 'B_fighter', 'title_bout'], inplace=True)
df.drop(columns=['title_bout'], inplace=True)

# In[ ]:





# In[15]:


df.to_csv('../../data/infdata_stage3.csv', index=False)


# In[ ]:




