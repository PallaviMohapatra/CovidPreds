#!/usr/bin/env python
# coding: utf-8

# In[86]:


#Importing required libraries

import pandas as pd
import pickle
from gensim import matutils, models
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import scipy.sparse
from nltk import word_tokenize, pos_tag
import seaborn as sns
from wordcloud import WordCloud


# In[87]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[88]:


#Reading clean training dataset from pickle created in Data Gathering Workbook

train_data_clean = pd.read_pickle('train_data_clean.pkl')
train_data_clean.head()


# In[89]:


#Converting clean training dataset into document term matrix

cv = CountVectorizer(stop_words='english')
train_data_cv = cv.fit_transform(train_data_clean.Tweet)
train_data_dtm = pd.DataFrame(train_data_cv.toarray(), columns=cv.get_feature_names())
train_data_dtm.index = train_data_clean.index
train_data_dtm.to_pickle(r"train_dtm.pkl")
pickle.dump(cv, open("cv.pkl","wb"))


# In[90]:


#Converting the document term matrix into a transposed dataframe

train_data = pd.read_pickle('train_dtm.pkl')
train_data = train_data.transpose()
train_data.head()


# In[91]:


#Displaying top 30 words used by each brand in training dataset

top_dict = {}
for c in train_data.columns:
    top = train_data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict


# In[92]:


# Print the top 30 words said by each brand

for brand, top_words in top_dict.items():
    print(brand)
    print(', '.join([word for word, count in top_words[0:29]]))
    print('---')


# In[93]:


# Looking at the most common top words and adding them to the stop word list
from collections import Counter

# Pulling out the top 30 words for each brand
words = []
for brand in train_data.columns:
    top = [word for (word, count) in top_dict[brand]]
    for t in top:
        words.append(t)
        
# Aggregating the list to identify the most common words along with how many timelines they occur in
Counter(words).most_common()


# In[94]:


# Excluding top words, proper nouns, specific brand related terms and irrelevant words

add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
manual_stop =['like','make','know','hey','ill', 'im', 'today', 'dont', 'sarka','aisle', 
              'kaileigh','yasmineevans','ikea','jasmine','oliver','look','going','whichuk',
              'yes','right', 'heres','kenneth', 'youve', 'entering', 'pic', 'said', 'weve', 
              'youre', 'youd', 'theyre','mcdonald','flwrt', 'did', 'got', 'erfeedback','middle',
              'excitableedgar','franco','jlandpartners','poundland', 'really','kiril','evening',
              'theyll', 'whats', 'james', 'becky', 'liam', 'jasonmanford','kr', 'thats', 'including',
              'thiskm', 'ive', 'colin', 'chris', 'zoe', 'daniel','spar', 'nick', 'hear','great',
              'greg','thank','nick','don','team','doo','nando','let','argosfasttrack','steph','aagfsvupuv',
              'subway','polly','ms','tc','know', 'mark','nandos','aldi','bm','lovewilko','silvana',
              'wilko','asda','debenhams','coop','greggs','homedepot','ikea','ikeauk','goodnight',
              'iceland','kfc','morrisons','brandon','tesco','waitrose','walmart','byggleck']

#Adding the stop words identified manually to the stop word list

for i in manual_stop:
    add_stop_words.append(i)


# In[95]:


#Displaying the complete list of stop words

print(add_stop_words)


# In[96]:


# Reading in cleaned data
train_data_clean = pd.read_pickle('train_data_clean.pkl')

# Adding new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreating document-term matrix
cv = CountVectorizer(stop_words=stop_words)
train_data_cv = cv.fit_transform(train_data_clean.Tweet)
train_data_stop = pd.DataFrame(train_data_cv.toarray(), columns=cv.get_feature_names())
train_data_stop.index = train_data_clean.index

# Pickling it for later use
pickle.dump(cv, open("cv_stop.pkl", "wb"))
train_data_stop.to_pickle("train_dtm_stop.pkl")


# In[97]:


#Visualizing word clouds for training dataset

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [25, 12]

brands=train_data_clean.index

# Create subplots for each brand
for index, brand in enumerate(train_data.columns):
    wc.generate(train_data_clean.Tweet[brand])
    plt.subplot(5,5,index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(brands[index])
    
plt.show()


# In[100]:


#Reading data again from pickle

train_data = pd.read_pickle('train_dtm.pkl')

train_data.head()


# In[101]:


#Manually creating a list of keywords identified through cursory observations on a set of three timelines

keyword_list=['trust','donation','donate','rainbow','hungry','protective',
              'charities','applause','nhs','awareness','lgbt','pride',
              'funds','kindness','challenge','shelter','support','nurses',
              'pets','face','covering','mask','guidelines','clapfornhs',
              'safe','thankyoutogether', 'distancing','recycle','plastic',
              'homelessness','coronavirus','cancer','partner',
              'everylittlehelps','safety','win','mandatory','disability',
              'government','netzero','lockdown']

###The words below do not occur in the training dataset

# 'mentalhealth,'feedthenation', 'foodheroes', 'fundraiser', 'redcross', 
# 'mentalhealthawareness', 'poverty', 'togetherathome', 'facemask', 'eatin'


# In[102]:


#Computing occurrences of keywords across brands in training data

train_key_df=train_data[keyword_list]
train_key_df.to_pickle("train_keyword_search.pkl")

train_key_df.head()


# In[103]:


#Reading the document term matrix of training data considering the stopwords

train_data = pd.read_pickle('train_dtm_stop.pkl')


# In[104]:


# Converting into term-document matrix
train_tdm = train_data.transpose()


# In[105]:


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


# In[106]:


# Putting the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus

train_sparse_counts = scipy.sparse.csr_matrix(train_tdm)
train_corpus = matutils.Sparse2Corpus(train_sparse_counts)


# In[107]:


# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix

cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())


# In[108]:


# Using corpus (term-document matrix) and id2word (dictionary of location: term)
# Specifying the number of topics and the number of passes

lda = models.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=2, passes=10)
lda.print_topics()


# In[109]:


# LDA for num_topics = 3

lda = models.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=3, passes=10)
lda.print_topics()


# In[110]:


# LDA for num_topics = 4

lda = models.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=4, passes=10)
lda.print_topics()


# In[111]:


#Function to pull out nouns from a string of text
   
def nouns(text):
    #Given a string of text, tokenize the text and pull out only the nouns.
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)


# In[112]:


# Reading the cleaned data, before the CountVectorizer step

train_data_clean = pd.read_pickle('train_data_clean.pkl')

train_data_clean.head(5)


# In[113]:


# Applying the nouns function to the transcripts to filter only on nouns

train_data_nouns = pd.DataFrame(train_data_clean.Tweet.apply(nouns))

train_data_nouns.head(5)


# In[114]:


# Creating a new document-term matrix using only nouns
# Re-adding the additional stop words since we are recreating the document-term matrix

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreating a document-term matrix with only nouns

cvn = CountVectorizer(stop_words=stop_words)
train_data_cvn = cvn.fit_transform(train_data_nouns.Tweet)
train_data_dtmn = pd.DataFrame(train_data_cvn.toarray(), columns=cvn.get_feature_names())
train_data_dtmn.index = train_data_nouns.index

train_data_dtmn.head(5)


# In[115]:


# Creating the gensim corpus

train_corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(train_data_dtmn.transpose()))

# Creating the vocabulary dictionary

id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())


# In[116]:


# LDA nouns with 2 topics

ldan = models.LdaModel(corpus=train_corpusn, num_topics=2, id2word=id2wordn, passes=10)
ldan.print_topics()


# In[117]:


# LDA nouns with 3 topics

ldan = models.LdaModel(corpus=train_corpusn, num_topics=3, id2word=id2wordn, passes=10)
ldan.print_topics()


# In[118]:


# LDA nouns with 4 topics

ldan = models.LdaModel(corpus=train_corpusn, num_topics=4, id2word=id2wordn, passes=10)
ldan.print_topics()


# In[119]:


# Function to pull out nouns and adjectives from a string of text

def nouns_adj(text):

#Given a string of text, tokenizing the text to pull out only the nouns and adjectives.
    
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)


# In[120]:


# Applying the nouns and adjectives function to the transcripts to filter only on nouns and adjectives

train_data_nouns_adj = pd.DataFrame(train_data_clean.Tweet.apply(nouns_adj))

train_data_nouns_adj.head(5)


# In[121]:


# Creating a new document-term matrix using only nouns and adjectives, also remove common words with max_df=0.8

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
train_data_cvna = cvna.fit_transform(train_data_nouns_adj.Tweet)
train_data_dtmna = pd.DataFrame(train_data_cvna.toarray(), columns=cvna.get_feature_names())
train_data_dtmna.index = train_data_nouns_adj.index

train_data_dtmna.head(5)


# In[122]:


# Creating the gensim corpus

train_corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(train_data_dtmna.transpose()))

# Creating the vocabulary dictionary

id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())


# In[123]:


# LDA nouns and adjectives with 2 topics

ldana = models.LdaModel(corpus=train_corpusna, num_topics=2, id2word=id2wordna, passes=10)
ldana.print_topics()


# In[124]:


# LDA nouns and adjectives with 3 topics

ldana = models.LdaModel(corpus=train_corpusna, num_topics=3, id2word=id2wordna, passes=10)
ldana.print_topics()


# In[125]:


# LDA nouns and adjectives with 4 topics

ldana = models.LdaModel(corpus=train_corpusna, num_topics=4, id2word=id2wordna, passes=10)
ldana.print_topics()


# In[126]:


#Final LDA model

lda_final = models.LdaModel(corpus=train_corpusna, num_topics=4, id2word=id2wordna, passes=80)
lda_final.print_topics()


# In[127]:


#Function to sort topic assignments

def sort_topic_assignment(corpus_transformed):
    lst = len(corpus_transformed)
    for i in range(0, lst):
        for j in range(0, lst-i-1):
            if (corpus_transformed[j][1] < corpus_transformed[j + 1][1]):
                temp = corpus_transformed[j]
                corpus_transformed[j]= corpus_transformed[j + 1]
                corpus_transformed[j + 1]= temp
    return corpus_transformed


# In[128]:


#Creating dataframe to store topic assignment and composition

ind=list(train_data_dtmna.index)
train_topic_assignment=pd.DataFrame(columns=['Brand','Topic','Topic Composition'])
for i in range(len(ind)):
    corpus_transformed = lda_final[train_corpusna[i]]
    sorted_assignment= sort_topic_assignment(corpus_transformed)
    train_topic_assignment = train_topic_assignment.append({'Brand': ind[i],'Topic':str(int(sorted_assignment[0][0])+1),'Topic Composition':sorted_assignment[0][1]}, ignore_index=True)


# In[130]:


train_topic_assignment


# In[132]:


#Printing number of brands under each topic in training data

train_topic_allocation=pd.DataFrame(data=train_topic_assignment.groupby("Topic").size(), columns=['Count'])
plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams.update({'font.size': 16})
chart=train_topic_allocation['Count'].plot(kind='bar', color='blue')
chart.set_xticklabels(labels, rotation=0)
plt.show()

