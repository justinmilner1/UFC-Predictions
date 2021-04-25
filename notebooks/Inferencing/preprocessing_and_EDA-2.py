#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


df = pd.read_csv('../../data/infdata.csv')


# In[27]:


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_rows', 200)


# In[28]:


df.T


# In[29]:


df.describe()


# ## Dealing with NaNs

# In[30]:


for column in df.columns:
    if df[column].isnull().sum()!=0:
        print(f"Nan in {column}: {df[column].isnull().sum()}")


# * Looks like the blue fighter stats have 1387 missing rows and the red fighter stats have 704 missing rows, i.e those fighters must not have had any previous fights. We should replace these values with the median.

# In[31]:


df2 = df.copy()


# * Referee doesn't look like an important column. Let's delete that.
# * Let's see if height and reach have a correlation
# * The rest i.e. Age, Stance and Height, let's fill with the median of that column.

# In[32]:


df2.drop(columns=['Referee'], inplace=True)


# In[33]:


df2['R_Reach_cms']


# In[34]:


# Set style of scatterplot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

# Create scatterplot of dataframe
sns.lmplot('R_Height_cms', # Horizontal axis
           'R_Reach_cms', # Vertical axis
           data=df2, # Data source
           fit_reg=True # fix a regression line
           ) # S marker size


# * We can see there is a positive correlation between height and reach. So we'll replace reach with height

# In[35]:


#df2['R_Reach_cms'].fillna(df2['R_Height_cms'], inplace=True)
#df2['B_Reach_cms'].fillna(df2['B_Height_cms'], inplace=True)
df2.fillna(df2.median(), inplace=True)


# In[36]:


df2['B_Stance'].value_counts()


# In[37]:


df2['R_Stance'].value_counts()


# In[38]:


df2['R_Stance'].fillna('Orthodox', inplace=True)
df2['B_Stance'].fillna('Orthodox', inplace=True)


# In[39]:


for column in df2.columns:
    if df2[column].isnull().sum()!=0:
        print(f"Nan in {column}: {df2[column].isnull().sum()}")


# ## Removing non essential columns

# In[40]:


df2['location'].value_counts()


# * Since we don't have home-country of each fighter, location is useless
# * Date of the fight is also not essential since we already created age with it
# * Draws are incredibly rare and should be removed from the target variable so it becomes a binary classification task
# * Fighter names are also to be removed

# In[41]:


df2['Winner'].value_counts()


# In[42]:


df2.drop(df2.index[df2['Winner'] == 'Draw'], inplace = True)
df2.drop(columns=['location', 'date', 'R_fighter', 'B_fighter'], inplace=True)


# In[19]:


df2.dtypes


# * Weight class and Stance are categories, Winner is out target variable.
# * We can one hot encode the categories

# In[20]:


df2 = pd.concat([df2, pd.get_dummies(df2[['weight_class', 'B_Stance', 'R_Stance']])], axis=1)
df2.drop(columns=['weight_class', 'B_Stance', 'R_Stance'], inplace=True)


# In[21]:


df2


# ## Saving the data

# In[22]:


df2.to_csv('../../data/infdata_stage1.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




