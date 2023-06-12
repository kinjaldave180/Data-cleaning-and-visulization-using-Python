#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df =pd.read_csv('Airbnb Dataset 19.csv')


# In[3]:


df.head(5)


# In[4]:


df.info()


# In[5]:


df['last_review'].isnull().sum()


# In[6]:


df['last_review'].fillna('-',inplace=True )


# In[7]:


df['last_review'].isnull().sum()


# In[8]:


df['reviews_per_month'].isnull().sum()


# In[9]:


df['reviews_per_month'].fillna('-',inplace=True )


# In[10]:


df['reviews_per_month'].isnull().sum()


# In[11]:


df.isnull().sum()


# In[12]:


df.duplicated().sum()


# In[13]:


df.describe()


# In[14]:


corr = df.corr(method='kendall')
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True)
df.columns


# In[15]:


sns.countplot(df['neighbourhood_group'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Neighbourhood Group')


# In[16]:


sns.countplot(df['neighbourhood'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(25,6)
plt.title('Neighbourhood')


# In[17]:


sns.countplot(df['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')


# In[ ]:





# In[18]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=df, x='neighbourhood_group',y='availability_365',palette='plasma')


# In[21]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)
plt.ioff()


# In[23]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.room_type)
plt.ioff()


# In[24]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.availability_365)
plt.ioff()


# In[25]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=df, x='room_type',y='pr',palette='plasma')


# In[27]:


sns.countplot(df['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('avaibility_365')


# In[ ]:





# In[ ]:





# In[30]:


df1 =pd.read_csv('HRDataset_v14.csv')


# In[44]:


df1.head(15)


# In[32]:


df1.info()


# In[33]:


df1['DateofTermination'].isnull().sum()


# In[34]:


df1['DateofTermination'].fillna('-',inplace=True )


# In[35]:


df1['DateofTermination'].isnull().sum()


# In[36]:


df1['ManagerID'].isnull().sum()


# In[37]:


df1['ManagerID'].fillna('-',inplace=True )


# In[38]:


df1['ManagerID'].isnull().sum()


# In[39]:


df1.describe()


# In[41]:


plt.figure(figsize=(12,7))
sns.distplot(df1.PerfScoreID,bins=30,kde=False)
plt.title("Perfomance of an employee")
plt.xlabel("Performance score",fontsize=12)
plt.ylabel("Count",fontsize=12)
plt.show()


# In[45]:


plt.figure(figsize=(12,5))
sns.distplot(df1.EmpSatisfaction,kde=False)
plt.title("Distribution of Satisfaction Level",fontsize=16)
plt.xlabel("Satisfaction Level",fontsize=12)
plt.ylabel("Count",fontsize=12)


# In[50]:


sns.countplot(df1['EmploymentStatus'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Count')


# In[53]:


plt.figure(figsize=(12,7))
sns.distplot(df1.Salary,bins=30,kde=False)
plt.title("Employee status")
plt.xlabel("Salary",fontsize=12)
plt.ylabel("count",fontsize=12)
plt.show()


# In[60]:


f,ax = plt.subplots(figsize=(15,15))
sns.boxplot(x='EmpSatisfaction', y='Salary', data=df1, hue='Position',palette='Set3')
plt.legend(loc='best')
plt.show()


# In[66]:


sns.violinplot(x="EmpID", y="EmploymentStatus", hue="Sex", data=df1, palette="muted", split=True,
               inner="stick")


# In[67]:


corr = df1.corr(method='kendall')
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True)
df.columns


# In[69]:


sns.jointplot(df1.DOB,df1.Salary, kind = "scatter")   
plt.show()


# In[81]:


fig,ax = plt.subplots(2,2, figsize=(10,10))               # 'ax' has references to all the four axes
plt.suptitle("Understanding the distribution of various factors", fontsize=20)
sns.distplot(df1['Salary'], ax = ax[0,0])  # Plot on 1st axes
ax[0][0].set_title('Distribution of Age')
sns.distplot(df1['EmpID'], ax = ax[0,1])  # Plot on IInd axes
ax[0][1].set_title('Distribution of Total no of employee')
sns.distplot(df1['EmpStatusID'], ax = ax[1,0])  # Plot on IIIrd axes
ax[1][0].set_title('Distribution of Employeement')
sns.distplot(df1['PositionID'], ax = ax[1,1])  # Plot on IV the axes
ax[1][1].set_title('Distribution of Years in Current Role')
plt.show()                                  


# In[ ]:




