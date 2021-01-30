#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Online-Shoppers-Purchasing-Intention/
# dataset link : http://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
# email : amirsh.nll@gmail.com


# In[1]:


import pandas
df = pandas.read_csv("O_S_I_train.csv")


# In[2]:


from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df, df['Revenue'], random_state = 0)


# In[3]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


# In[4]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

