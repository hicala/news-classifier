#!/usr/bin/env python
# coding: utf-8

# ## The Dataset
# 
# The dataset we’ll use for this python project- we’ll call it <em>news.csv</em>. 
# 
# This dataset has a shape of 7796×4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE. 

# Import all the needed Libraries

# In[10]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Read the data into a DataFrame, and get the shape of the data and the first 5 records.

# In[18]:


#Read the data
df=pd.read_csv("data/news.csv")

#Get shape and head
df.shape
df.head()


# Now, let's extract and get the labels from the DataFrame.

# In[19]:


#DataFlair - Get the labels
labels=df.label
labels.head()


# After the definition of the classifiers, Split the dataset into training and testing sets.

# In[20]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# Initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7:
# 
# * (terms with a higher document frequency will be discarded).
# 
# * Stop words need to be filtered out before processing the natural language data. 
# 
# * TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
# 
# 

# In[21]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# Initialize a PassiveAggressiveClassifier. 
# 
# Work sequence:
# 
# * Fit this on tfidf_train and y_train.
# * Predict on the test set from the TfidfVectorizer
# * Calculate the accuracy with accuracy_score() from sklearn.metrics.
# 

# In[22]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# The accuracy of the model is 92.42%. Now once we defined this charatceristic we are ready to print a cond=fusion matrix to get a better insgihts into the ocurrency of false and true negatives and positives.

# In[23]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# ## Results
# 
# 1. With model we have:
#     + True Positives = 589
#     + False Positive = 42
#     + True Negatives =587
#     + False Negatives = 49
# 

# ## Conclusions
# 
# 1. With Python to we created a model which able to detect fake news. 
# 2. We took a political dataset, 
# 3. We implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model.
# 3. We ended up obtaining an accuracy of 92.82% in magnitude.
