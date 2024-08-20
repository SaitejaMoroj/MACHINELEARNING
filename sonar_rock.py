#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#view of data
sonar_rock=pd.read_csv("Copy of sonar data.csv")


# In[3]:


sonar_rock


# In[4]:


#classifying the x and y
X=sonar_rock.drop("R",axis=1)

y=sonar_rock['R']


# In[5]:


len(X)


# In[6]:


X


# #Dataexploration (exploratory data analysis  or EDA)
#  
#  comparint the every column of the data and becoming the expert on the data
# 
# questions are trying to solve
# 
# kind of the data we are being dealing with like categorical or numerical
# 
# what is missing and we are dealing with 
# 
# what the outliers and why should we care about them
# 
# Add change or remove of the features to get more out of our data

# In[7]:


# comparing the Eda
sonar_rock.shape

sonar_rock['0.0200'].value_counts()


# In[8]:


#evaluating using the crosstab
pd.crosstab(sonar_rock['0.0200'],sonar_rock['R'])


# In[9]:


#plotting
plt.scatter(sonar_rock['0.0200'],sonar_rock.R=='R')
plt.scatter(sonar_rock['0.0200'],sonar_rock.R=='M')
plt.xlabel('sonar_rock')
plt.ylabel('Accuracy')


# In[10]:


#Describe
sonar_rock.describe()


# In[11]:


#information
sonar_rock.info()


# model evaluation
# 1)logisticRegression
# 
# 2)RandomForestClassifier
# 

# In[12]:


#modelling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# In[13]:


#evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score,f1_score,roc_curve


# In[14]:


#evaluating the Hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[15]:


#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


np.random.seed(42)
model=LogisticRegression()
model.fit(X_train,y_train)


# In[17]:


model.score(X_test,y_test)


# In[18]:


np.random.seed(42)
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# Evaluating  using hyperparameters
# 
# we will test the data and increase its accuracy
# 
# *Hyperparameter
# 
# *cross_validation
# 
# *f1_score
# 
# *Roc_curve
# 
# *precision
# 
# *recall
# 
# *confusion_matrix
# 
# *feature importance
# 
# *classification_report
# 
# *Area under the curve

# In[19]:


#hyperparameters for logisticRegression
log_reg={'C':np.logspace(0,20),
         'solver':['liblinear']}
#hyperparameter for RandomForestClassifier
rand_reg={'n_estimators':[10,50,100,150,200],
         'max_depth':[None,2,3,4,5],
         'min_samples_split':[2,5,10],
         'min_samples_leaf':[1,2,4],
         'max_features':['sqrt', 'log2']}


# ## for logisticRegresion by using the RandomSearchCv
# new_reg=RandomSearchCv(LogisticRegression(), param_distributions=log_reg,n_iter=20,verbose=True)

# In[20]:


np.random.seed(42)
new_reg=RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg,n_iter=20,verbose=True)
new_reg.fit(X_train,y_train)


# In[21]:


new_reg.best_params_


# In[22]:


new_reg.score(X_test,y_test)


# In[23]:


#evaluating for RandomForestClassifier
np.random.seed(42)
rand_reg1=RandomizedSearchCV(RandomForestClassifier(), param_distributions=rand_reg,cv=5,n_iter=20,verbose=True)
rand_reg1.fit(X_train,y_train)


# In[24]:


rand_reg1.score(X_test,y_test)


# In[25]:


#gridsearchcv
#hyperparameters for logisticRegression
log_grid={'C':np.logspace(0,20),
         'solver':['liblinear']}
#hyperparameter for RandomForestClassifier
rand_reg={'n_estimators':[10,50,100,150,200],
         'max_depth':[None,2,3,4,5],
         'min_samples_split':[2,5,10],
         'min_samples_leaf':[1,2,4],
         'max_features':['sqrt', 'log2']}



# In[26]:


#evaluating the grid
log_grid1=GridSearchCV(LogisticRegression(), param_grid=log_grid,cv=5,verbose=True)


# In[27]:


log_grid1.fit(X_train,y_train)



# In[28]:


log_grid1.score(X_test,y_test)



# In[29]:


#evaluating the grid for RandomForest
rand_grid1=GridSearchCV(RandomForestClassifier(), param_grid=rand_reg,cv=5,verbose=True)
rand_grid1.fit(X_train,y_train)


# In[30]:


rand_grid1.score(X_test,y_test)


# Evaluating our machinelearning classifier,beyond the accuracy
# *ROC and AUC curve
# 
# *Confusion matrix
# 
# *classification report
# 
# *Precision
# 
# *recall
# 
# *f1_score tocomparision and evaluate and predict the model we need to make the predictions

# In[44]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming X_test and y_test are defined properly

# Encode labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Get predicted probabilities for the positive class
y_pred_proba = log_grid1.predict_proba(X_test)[:, 1]

# Calculate false positive rate (fpr), true positive rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_pred_proba)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[33]:


y_preds=rand_grid1.predict(X_test)


# In[34]:


print(confusion_matrix(y_preds,y_test))


# In[35]:


import seaborn as sns

# Set font size for seaborn
sns.set_context("paper", font_scale=1.5)
# Assuming you have already calculated the confusion matrix
conf_matrix = confusion_matrix(y_preds, y_test)

# Plot confusion matrix using seaborn heatmap
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[46]:


classification_report(y_preds,y_test)


# In[47]:


precision = precision_score(y_test, y_preds, pos_label='M')
print(f'Precision: {precision}')


# In[49]:


from sklearn.metrics import make_scorer
recall_scorer = make_scorer(recall_score, pos_label='M')
recall_scorer


# In[39]:


#Exporting the model
import pickle

# Assuming you have a trained model named 'model'
# Train your model here...

# Save the trained model to a file using pickle
with open('sonar_rock.pkl', 'wb') as file:
    pickle.dump(sonar_rock, file)


# In[ ]:




