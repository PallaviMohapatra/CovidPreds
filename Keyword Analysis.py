#!/usr/bin/env python
# coding: utf-8

# In[43]:


#Importing required libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# ## Keyword Analysis of Training dataset

# In[56]:


#Reading relevant keywords from training keyword set

key_df=pd.read_pickle("train_keyword_search.pkl")


# In[57]:


key_df


# ### Visualising term frequency across brands in training data

# In[29]:


#NHS

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['nhs'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[58]:


#Partner

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['partner'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[32]:


#Donate

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['donate'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[33]:


#Charities

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['charities'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[34]:


#Support

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['support'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[35]:


#Face

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['face'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[37]:


#Government

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['government'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[38]:


#Win

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['win'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[42]:


#Lockdown

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['lockdown'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# ## Keyword Analysis of Test Dataset

# In[59]:


#Reading relevant keywords from test keyword set

key_df=pd.read_pickle("test_keyword_search.pkl")


# In[60]:


key_df


# ### Visualizing term frequency across brands in test data

# In[50]:


#NHS

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['nhs'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[49]:


#Support

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['support'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[48]:


#Win

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['win'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[55]:


#Partner

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['partner'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()


# In[61]:


#Donate

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=key_df['donate'].plot(kind='bar', color='blue')
chart.set_xticklabels(key_df.index, rotation=90)
plt.show()

