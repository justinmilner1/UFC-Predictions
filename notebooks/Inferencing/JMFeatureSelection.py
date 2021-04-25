#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.model_selection import StratifiedKFold, KFold, train_test_split



# ## Prepping Data

# In[2]:


original_df = pd.read_csv('../../data/infdata_stage1.csv')
df = original_df
pd.set_option('display.max_rows', 200)
len(df.index)


# In[3]:


y = df['Winner']
X = df.drop(columns = ['Winner'])
X.fillna(df.median(), inplace=True)
print(y.head(20))



# ## Univariate Selection

# In[4]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(k=10)


# In[5]:


fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[6]:


len(dfcolumns)


# In[7]:


#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores = featureScores.sort_values(by='Score', kind="quicksort", ascending=False)

# print(featureScores)


# In[8]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.bar(featureScores['Specs'],featureScores['Score'])
plt.show()


# In[9]:


smallest = featureScores[featureScores['Score'] < 50]
smallest


# In[10]:


#removing columns that have low scores
print(len(df.columns))
for col in smallest['Specs']:
    df = df.drop(columns = col)
print(len(df.columns))


# In[11]:


df.to_csv('../../data/infdata_stage2.csv', index=False)


# ## Feature Importance

# In[ ]:


# df = original_df
# len(df.columns)


# In[ ]:


# model = ExtraTreesClassifier()
# model.fit(X,y)
# # print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[ ]:


# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(len(df.columns)).plot(kind='barh')
# plt.show()


# In[ ]:


# smallest = feat_importances[feat_importances < .0078]
# smallest


# In[ ]:


# #removing columns that have low scores
# print(len(df.columns))
# for col in smallest.index:
#     df = df.drop(columns = col)
# print(len(df.columns))


# In[ ]:


# df.to_csv('../data/JMpreprocessed_FeatureImportance_<.0078.csv', index=False)


# ## Correlation Matrix w Heatmap

# In[ ]:


# corrmat = df.corr()
# top_corr_features = corrmat.index
# c = df[top_corr_features].corr().abs()

# s = c.unstack()
# so = s.sort_values(kind="quicksort", ascending=False, na_position='last')
# so = so[so != 1]
# so[0:20]


# In[ ]:


# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])

# ax.bar(range(0, len(so)), so)
# plt.show()


# In[ ]:




