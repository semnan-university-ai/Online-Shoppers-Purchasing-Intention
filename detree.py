#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Online-Shoppers-Purchasing-Intention/
# dataset link : http://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
# email : amirsh.nll@gmail.com


# In[1]:


import pandas
trainingData = pandas.read_csv("O_S_I_train.csv")
                


# In[14]:


# We are now ready to train our Decision Tree classifier
from sklearn import tree
import numpy as np

clf=tree.DecisionTreeClassifier(max_leaf_nodes=20)
X=np.array(trainingData[0])
y=np.array(trainingData[1])
clf=clf.fit(X,y)


# In[ ]:


print(trainingData[0])


# In[ ]:


import graphviz
with open("MTTTEST.dot","w") as f:
    f = tree.export_graphviz(clf,
                            feature_names=features,out_file=f)


# In[ ]:


clf.feature_importances_


# In[ ]:


def transformTestDataMTT(testFile,features):

    transformData=[]
    ids=[]
    blank=""
    with open(testFile,"r") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',')
        lineNum=1
        for row in lineReader:
            if lineNum==1:
                header=row
            else: 
                allFeatures=list(row)
                featureVector = [allFeatures[header.index(feature)] for feature in features]
                #featureVector=list(map(lambda x:0 if x=="" else x, featureVector))
                transformData.append(featureVector)
                ids.append(row[0])
            lineNum=lineNum+1 
    return transformData,ids


# In[ ]:


def MTTTest(classifier,resultFile,transformDataFunction=transformTestDataMTT):
    testFile="O_S_I_test.csv"
    testData=transformDataFunction(testFile,features)
    result=classifier.predict(testData[0])
    with open(resultFile,"w") as mf:
        ids=testData[1]
        lineWriter=csv.writer(mf,delimiter=',')
        lineWriter.writerow(["ShopperId","Revenue"])
        for rowNum in range(len(ids)):
            try:
                lineWriter.writerow([ids[rowNum],result[rowNum]])
            except Exception as e:
                print (e)
# Let's take this for a spin! 
resultFile="result.csv"
MTTTest(clf,resultFile)

