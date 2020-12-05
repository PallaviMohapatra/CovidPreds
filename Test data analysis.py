#!/usr/bin/env python
# coding: utf-8

# In[50]:


#Importing required libraries

import pandas as pd
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag


# In[51]:


#Reading test data from pickle in Data Gathering workbook

test_data_clean = pd.read_pickle('test_data_clean.pkl')
test_data_clean.head()


# In[52]:


#Converting into document term matrix and pickling

cv = CountVectorizer(stop_words='english')
test_data_cv = cv.fit_transform(test_data_clean.Tweet)
test_data_dtm = pd.DataFrame(test_data_cv.toarray(), columns=cv.get_feature_names())
test_data_dtm.index = test_data_clean.index
test_data_dtm.to_pickle(r"test_dtm.pkl")
pickle.dump(cv, open("cv.pkl","wb"))


# In[53]:


#Reading from pickle and transposing the dataframe

test_data = pd.read_pickle('test_dtm.pkl')
test_data = test_data.transpose()
test_data.head()


# In[54]:


#List of stop words identified in Training data EDA and Topic Modelling Workbook

add_stop_words=['thanks', 'thank', 'store', 'new', 'stores', 'help', 'know', 'make', 'sorry', 'like', 'look', 
                'just', 'order', 'today', 'day', 'time', 'home', 'like', 'make', 'know', 'hey', 'ill', 'im', 
                'today', 'dont', 'sarka', 'aisle', 'kaileigh', 'yasmineevans', 'ikea', 'jasmine', 'oliver', 
                'look', 'going', 'whichuk', 'yes', 'right', 'heres', 'kenneth', 'youve', 'entering', 'pic', 
                'said', 'weve', 'youre', 'youd', 'theyre', 'mcdonald', 'flwrt', 'did', 'got', 'erfeedback', 
                'middle', 'excitableedgar', 'franco', 'jlandpartners', 'poundland', 'really', 'kiril', 
                'evening', 'theyll', 'whats', 'james', 'becky', 'liam', 'jasonmanford', 'kr', 'thats', 
                'including', 'thiskm', 'ive', 'colin', 'chris', 'zoe', 'daniel', 'spar', 'nick', 'hear', 
                'great', 'greg', 'thank', 'nick', 'don', 'team', 'doo', 'nando', 'let', 'argosfasttrack', 
                'steph', 'aagfsvupuv', 'subway', 'polly', 'ms', 'tc', 'know', 'mark', 'nandos', 'aldi', 'bm', 
                'lovewilko', 'silvana', 'wilko', 'asda', 'debenhams', 'coop', 'greggs', 'homedepot', 'ikea', 
                'ikeauk', 'goodnight', 'iceland', 'kfc', 'morrisons', 'brandon', 'tesco', 'waitrose', 'walmart', 
                'byggleck']


# In[55]:


# Reading in cleaned data

test_data_clean = pd.read_pickle('test_data_clean.pkl')

# Adding the stop words

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreating document-term matrix

cv = CountVectorizer(stop_words=stop_words)
test_data_cv = cv.fit_transform(test_data_clean.Tweet)
test_data_stop = pd.DataFrame(test_data_cv.toarray(), columns=cv.get_feature_names())
test_data_stop.index = test_data_clean.index

# Pickling it for later use

pickle.dump(cv, open("cv_stop.pkl", "wb"))
test_data_stop.to_pickle("test_dtm_stop.pkl")


# In[56]:


#Reading from pickle

test_data = pd.read_pickle('test_dtm.pkl')
test_data.head()


# In[57]:


#List of keywords to count occurrences of those words across brands for the test dataset

keyword_list=['trust','donation','donate','rainbow','lockdown',
              'charities','nhs','funds','support','nurses',
              'pets','face','safe','recycle','plastic',
              'coronavirus','cancer','partner','everylittlehelps',
              'safety','win','disability','government','lockdown']

##The words below are not in the test dataset

#'mentalhealth,'feedthenation', 'foodheroes', 'fundraiser', 'redcross', 
#'mentalhealthawareness', 'poverty', 'togetherathome', 'facemask', 'eatin'


# ['lgbt', 'awareness', 'pride', 'distancing', 'thankyoutogether', 'homelessness', 
# 'kindness', 'covering', 'mandatory', 'mask', 'netzero', 'clapfornhs', 'challenge', 
# 'protective', 'guidelines', 'shelter', 'hungry', 'applause']


# In[58]:


#Converting the keyword specific dataframe on test data into pickle

test_key_df=test_data[keyword_list]
test_key_df.to_pickle("test_keyword_search.pkl")

test_key_df.head()


# In[59]:


#Reading in pickle of document term matrix with stop words test dataset

test_data = pd.read_pickle('test_dtm_stop.pkl')

#Converting into term-document matrix

test_tdm = test_data.transpose()


# In[60]:


#Function to pull out nouns and adjectives from a string of text

def nouns_adj(text):

#Given a string of text, tokenizing the text and pulling out only the nouns and adjectives.
    
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)


# In[61]:


# Applying the nouns and adjectives function to the transcripts to filter only on nouns and adjectives

test_data_nouns_adj = pd.DataFrame(test_data_clean.Tweet.apply(nouns_adj))

test_data_nouns_adj.head(5)


# In[62]:


# Creating a new document-term matrix using only nouns and adjectives, also remove common words with max_df

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
test_data_cvna = cvna.fit_transform(test_data_nouns_adj.Tweet)
test_data_dtmna = pd.DataFrame(test_data_cvna.toarray(), columns=cvna.get_feature_names())
test_data_dtmna.index = test_data_nouns_adj.index

test_data_dtmna.head(5)


# In[63]:


# Creating the gensim corpus

test_corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(test_data_dtmna.transpose()))

# Creating the vocabulary dictionary

id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())


# In[64]:


# LDA model for test dataset

ldana = models.LdaModel(corpus=test_corpusna, num_topics=4, id2word=id2wordna, passes=80)
ldana.print_topics()


# In[65]:


#Function to sort topic composition

def sort_topic_assignment(corpus_transformed):
    lst = len(corpus_transformed)
    for i in range(0, lst):
        for j in range(0, lst-i-1):
            if (corpus_transformed[j][1] < corpus_transformed[j + 1][1]):
                temp = corpus_transformed[j]
                corpus_transformed[j]= corpus_transformed[j + 1]
                corpus_transformed[j + 1]= temp
    return corpus_transformed


# In[66]:


#Creating topic assignment and composition dataframe for test data

ind=list(test_data_dtmna.index)
test_topic_assignment=pd.DataFrame(columns=['Brand','Topic','Topic Composition'])
for i in range(len(ind)):
    corpus_transformed = ldana[test_corpusna[i]]
    sorted_assignment= sort_topic_assignment(corpus_transformed)
    test_topic_assignment = test_topic_assignment.append({'Brand': ind[i],'Topic':int(sorted_assignment[0][0])+1,'Topic Composition':sorted_assignment[0][1]}, ignore_index=True)


# In[67]:


test_topic_assignment


# In[68]:


#visualising the number of brands for each topic

test_topic_allocation=pd.DataFrame(data=test_topic_assignment.groupby("Topic").size(), columns=['Count'])
plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=test_topic_allocation['Count'].plot(kind='bar', color='blue')
chart.set_xticklabels(labels,rotation=0)
plt.show()

