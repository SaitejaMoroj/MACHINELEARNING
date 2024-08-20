#!/usr/bin/env python
# coding: utf-8

# # Credict card fraud detection

# The dataset contains transactions made by credit cards in September 2013 by European cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

#  ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.

# In[37]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#loading the dataset
credit_card = pd.read_csv('Downloads/archive (3)/creditcard.csv')


# In[5]:


credit_card


# In[7]:


credit_card.head()


# In[8]:


credit_card.info()


# In[9]:


credit_card.isna().sum()


# In[10]:


credit_card.describe()


# In[11]:


#checking for the fault and normaltranstation
credit_card['Class'].value_counts()


# In[13]:


legit=credit_card[credit_card.Class==0]


# In[14]:


fraud=credit_card[credit_card.Class==1]


# In[15]:


len(legit)


# In[16]:


len(fraud)


# In[17]:


legit.head()


# In[18]:


fraud.head()


# In[19]:


#Descibe the each
legit.describe()


# In[20]:


fraud.describe()


# In[21]:


#comparing the both classes
credit_card.groupby('Class').mean()


# In[22]:


#making the databalanced
legit_sample=legit.sample(n=492)


# In[25]:


#concatenate two datasamples
new_data=pd.concat([legit_sample,fraud],axis=0)


# In[26]:


new_data


# In[27]:


len(new_data)


# In[28]:


new_data['Class'].value_counts()


# In[31]:


new_data.groupby('Class').mean()


# In[32]:


#plotting using the values
new_data['Amount'].value_counts()


# In[33]:


pd.crosstab(new_data['Amount'],new_data['Class'])


# # scatterplot

# In[34]:


plt.scatter(new_data['Amount'],new_data['Class'])


# In[38]:


benign_transactions = credit_card[credit_card['Class'] == 0]
malignant_transactions = credit_card[credit_card['Class'] == 1]

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=benign_transactions, x='Amount', y='Class', label='No fraud')
sns.scatterplot(data=malignant_transactions, x='Amount', y='Class', label='fraud')

# Set labels and title
plt.xlabel('Amount')
plt.ylabel('Transaction Type')
plt.title('Scatter Plot of Amount vs Transaction Type')

# Show legend
plt.legend()

# Show the plot
plt.show()


# In[39]:


#using the ml algo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[75]:


X = credit_card.drop('Class', axis=1)
y = credit_card['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[76]:


X


# In[77]:


y


# In[78]:


len(X)


# In[79]:


len(y)


# In[ ]:





# In[81]:


len(X_train)


# In[82]:


len(X_test)


# In[83]:


y_train.shape,X_train.shape


# In[ ]:






# In[85]:


model=LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)


# In[86]:


model.score(X_test,y_test)


# In[87]:


#model evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# In[88]:


y_preds=model.predict(X_test)


# In[89]:


y_preds


# In[90]:


len(y_preds)


# In[92]:


accuracy=accuracy_score(y_preds,y_test)
accuracy


# In[93]:


mean_absolute_error(y_preds,y_test)


# In[94]:


r2_score(y_preds,y_test)


# In[95]:


mean_squared_error(y_preds,y_test)


# # importing the model
# 

# In[98]:


from joblib import dump, load

# Assuming you have a trained model named 'model'
# Save the model to a file
dump(model, 'credit_card_model.joblib')

# Load the model from the file
loaded_model = load('credit_card_model.joblib')


# In[ ]:




