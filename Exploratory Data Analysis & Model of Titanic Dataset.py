#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[190]:


data = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\Exploratory Data Analysis of Titanic Dataset\\train.csv')
data.head()


# In[191]:


data.shape


# In[192]:


data.isnull()


# ### Visualize and Handling Missing Values

# In[193]:


data.isnull().sum()


# In[194]:


sns.heatmap(data.isnull(),yticklabels=False)


# #### As we can see in above heatmap shows Age has some white bars shows that their are approx 20% missing values and Cabin has a very large number of missing values

# In[195]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data)


# #### Above Diagram depicts that the approx 600 were not survived and approx 360 survived

# In[196]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=data)


# #### Above diagram depicts that Male Passenger died much as compared to Female and Female Survived much as compared to Male Passenger

# In[197]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=data)


# #### Above diagram depicts that Passenger having class 3 died much and Passenger having class 1 survived larger than both class

# In[198]:


sns.distplot(data['Age'].dropna(),kde=False,color='darkred',bins=40)


# #### As we can see in above diagram, Age from 20 to 40 was there at titanic and less number of people are age between 70 to 80 

# In[199]:


data['Age'].hist(bins=40,color='blue',alpha=0.5)


# In[200]:


sns.countplot(x='SibSp',data=data)


# #### As above diagram depicts, approx 600 having sibling or spouse is 0 and approx 200 have sibling or spouse is 1 as on...

# In[201]:


data['Fare'].hist(color='green',bins=30,alpha=0.8)


# ## Data Cleansing

# In[202]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')


# #### As above diagram depicts, Passenger class 1 having mean age is approx 36-37 and Passenger class 2 having mean age is around 28-29 and Passenger class 3 having mean age around 24-25

# #### Imputation of Age

# In[203]:


def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


# In[204]:


data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)


# In[205]:


sns.heatmap(data.isnull(),yticklabels=False)


# #### As we can see in above heatmap age is not showing any bar because no missing value is left due to imputation based on pclass

# #### As we can see 'Cabin' Feature contains many missing values so I decided to drop it as of now

# In[206]:


data.drop('Cabin',axis=1,inplace=True)


# In[207]:


data.head()


# In[208]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[209]:


data.shape


# In[210]:


data.dropna(inplace=True)
data.reset_index(drop=True)
data.head()


# In[211]:


data.shape


# In[212]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[213]:


data.info()


# In[214]:


data.head()


# In[215]:


sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'],drop_first=True)


# In[216]:


sex.head()


# In[217]:


embark.head()


# In[220]:


data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data.head()


# In[221]:


data = pd.concat([data,sex,embark],axis=1)
data.head()


# In[222]:


data.drop('Survived',axis=1)
data.head()


# In[224]:


data['Survived'].head()


# ### Split Train and Test Data

# In[225]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,data['Survived'],test_size=0.3,random_state=101)


# ### Building and Train Logistic Regression Model

# In[226]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)


# ### Prediction

# In[227]:


y_predict = model.predict(x_test)
y_predict


# ### Confusion Matrix to check accuracy of model

# In[230]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
cm


# In[231]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_predict)
accuracy


# ### Model Evaluation

# In[237]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

