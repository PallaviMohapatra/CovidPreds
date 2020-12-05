#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing required libraries

import pandas as pd
import numpy as np
import json
from gensim import matutils, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import scipy.sparse
import re
import sklearn
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


#Displaying all columns and rows which otherwise are partially hidden

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# ## Converting raw json data into compiled text files

# In[4]:


#Retrieving brand addresses on system

import os
root = r"C:\Users\PallaviM777\Documents\Data Dissertation"
folders=[]
brands=[]
for item in os.listdir(root):
    folders.append(os.path.join(root, item))
    brands.append(item)


# In[ ]:


#Compiling raw tweets for each brand into separate spreadsheets

directory_add=r'C:\Users\PallaviM777\Documents\Condensed Data'
for i in range(1):
    os.chdir(folders[i])
    x=[]
    m=folders[i]+"\\"
    for root, dirs, files in os.walk(".", topdown = True):
        for name in files:
            m=folders[i]+"\\"+name
            x.append(m)
    data=pd.DataFrame()
    for j in range(len(x)):
        d=pd.read_json(x[j], typ='series')
        y=d.to_frame(j).transpose()
        data=data.append(y)
        if(j%500==0):
            print("Reached "+str(j))
    filename=directory_add+'\\'+brands[i]+'.csv'
    data.to_csv(filename)


# In[5]:


#Splitting into training and test datasets and removing unnecessary columns

source=r"C:\Users\PallaviM777\Documents\Condensed Data\\"
train_dest=r"C:\Users\PallaviM777\Documents\Timeline Data\\"
test_dest=r"C:\Users\PallaviM777\Documents\Timeline Test Data\\"
for i in brands:
    filename=source+'\\'+i+'.csv'
    d1=pd.read_csv(filename, parse_dates=['datetime'], dayfirst=True)
    d1.sort_values(by=['datetime'],inplace=True)
    d1.drop(columns=['Unnamed: 0', 'ID','has_media','medias','nbr_favorite','nbr_reply','nbr_retweet','url','user_id'], inplace=True)
    
    d2=d1[d1['is_reply']==False]
    d2=d2[d2['datetime']>'2020-05-04']
    d3=d2[d2['datetime']<'2020-08-05']
    d4=d2[d2['datetime']>'2020-08-04']
    
    trainfile=train_dest+'\\'+i+'.csv'
    testfile=test_dest+'\\'+i+'.csv'
    
    d3.to_csv(trainfile)
    d4.to_csv(testfile)


# In[6]:


#Function to remove non ascii characters from text

ascii = set(string.printable)   

def remove_non_ascii(s):
    encoded_string = s.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    return decode_string


# In[7]:


#Writing the training dataset tweets for each brand from the spreadsheet into a text file

source=r"C:\Users\PallaviM777\Documents\Timeline Data\\"
destination=r"C:\Users\PallaviM777\Documents\Timeline_text\\"
for i in brands:
    filename=source+'\\'+i+'.csv'
    d1=pd.read_csv(filename, parse_dates=['datetime'], dayfirst=True)
    filename=destination+'\\'+i+'.txt'
    with open(filename,'w') as f:
        for j in d1['text']:
            english_check = re.compile(r'[a-zA-Z][+]')
            j=j.replace('https://www. ','https://www.')
            j=j.replace('http://www. ','http://www.')
            j=j.replace('https:// ','https://')
            j=j.replace('http:// ','http://')
            j=j.replace(' se?recipient_id=', '')
            j=j.replace('pic.twitter.com/', '')
            j=re.sub('http://\S+|https://\S+', '',j)
            j=j.replace('/', '')
            j=re.sub(r'\b\w{1,2}\b', '', j)
            if english_check.match(j):
                f.write(j+' ')
            else:
                j=remove_non_ascii(j)
                f.write(j+' ')


# In[8]:


#Writing the test dataset tweets for each brand from the spreadsheet into a text file

source=r"C:\Users\PallaviM777\Documents\Timeline Test Data\\"
destination=r"C:\Users\PallaviM777\Documents\Timeline_test_text\\"
for i in brands:
    filename=source+'\\'+i+'.csv'
    d1=pd.read_csv(filename, parse_dates=['datetime'], dayfirst=True)
    filename=destination+'\\'+i+'.txt'
    with open(filename,'w') as f:
        for j in d1['text']:
            english_check = re.compile(r'[a-zA-Z][+]')
            j=j.replace('https://www. ','https://www.')
            j=j.replace('http://www. ','http://www.')
            j=j.replace('https:// ','https://')
            j=j.replace('http:// ','http://')
            j=j.replace(' se?recipient_id=', '')
            j=j.replace('pic.twitter.com/', '')
            j=re.sub('http://\S+|https://\S+', '',j)
            j=j.replace('/', '')
            j=re.sub(r'\b\w{1,2}\b', '', j)
            if english_check.match(j):
                f.write(j+' ')
            else:
                j=remove_non_ascii(j)
                f.write(j+' ')


# In[9]:


#Function to remove stopwords and punctuation

def clean(input_text):
    result = ""
    remove_words = stopwords.words('english')
    for t in input_text:
        if t not in remove_words and t not in string.punctuation:
            result = result + " " + t
    return result


# In[10]:


#Converting training dataset text files into pickles for later use

source=r"C:\Users\PallaviM777\Documents\Timeline_text\\"
destination=r"C:\Users\PallaviM777\Documents\Corpus Pickles\\"
for i, c in enumerate(brands):
    f=open(source+c+".txt","r")
    text=clean(f.readlines())
    with open(destination+c+".txt", "wb") as file:
        pickle.dump(text, file)


# In[11]:


#Converting test dataset text files into pickles for later use

source=r"C:\Users\PallaviM777\Documents\Timeline_test_text\\"
destination=r"C:\Users\PallaviM777\Documents\Corpus Test Pickles\\"
for i, c in enumerate(brands):
    f=open(source+c+".txt","r")
    text=clean(f.readlines())
    with open(destination+c+".txt", "wb") as file:
        pickle.dump(text, file)


# In[12]:


#Reading training dataset pickles into a dictionary

source=r"C:\Users\PallaviM777\Documents\Corpus Pickles\\"
train_data = {}
for i, c in enumerate(brands):
    with open(source + c + ".txt", "rb") as file:
        train_data[c] = pickle.load(file)


# In[13]:


#Reading test dataset pickles into a dictionary

source=r"C:\Users\PallaviM777\Documents\Corpus Test Pickles\\"
test_data = {}
for i, c in enumerate(brands):
    with open(source + c + ".txt", "rb") as file:
        test_data[c] = pickle.load(file)


# In[14]:


#Displaying the values in the training data dictionary

next(iter(train_data.values()))


# In[15]:


#Function to take a list of text and compile them into one large chunk of text

def combine_text(list_of_text):
    combined_text = ''.join(list_of_text)
    return combined_text


# In[16]:


#Converting list of tweets for each brand into one running chunk of text for both datasets

train_data_combined = {key: [combine_text(value)] for (key, value) in train_data.items()}
test_data_combined = {key: [combine_text(value)] for (key, value) in test_data.items()}


# In[17]:


#Converting both training and test dataset dictionaries into dataframes

train_data_df = pd.DataFrame.from_dict(train_data_combined).transpose()
train_data_df.columns = ['Tweet']
train_data_df = train_data_df.sort_index()
train_data_df.index.rename('Brand',inplace=True)

test_data_df = pd.DataFrame.from_dict(test_data_combined).transpose()
test_data_df.columns = ['Tweet']
test_data_df = test_data_df.sort_index()
test_data_df.index.rename('Brand',inplace=True)


# ## Cleaning

# In[18]:


#Function to make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.

def clean_text_round1(text):
    
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)


# In[19]:


#First round of cleaning

train_data_clean = pd.DataFrame(train_data_df.Tweet.apply(round1))
test_data_clean = pd.DataFrame(test_data_df.Tweet.apply(round1))


# In[20]:


#Function to get rid of some additional punctuation and non-sensical text that was missed the first time around.

def clean_text_round2(text):
    
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)


# In[21]:


#Second round of cleaning and pickling of both dataframes for later use

train_data_clean = pd.DataFrame(train_data_clean.Tweet.apply(round2))
test_data_clean = pd.DataFrame(test_data_clean.Tweet.apply(round2))

train_data_df.to_pickle(r"train_corpus.pkl")
train_data_clean.to_pickle('train_data_clean.pkl')

test_data_df.to_pickle(r"test_corpus.pkl")
test_data_clean.to_pickle('test_data_clean.pkl')

