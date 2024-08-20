#!/usr/bin/env python
# coding: utf-8

# #predcing the heart disease using the machine learning model
# 
# In this notebook we are going to use various ml libraries
# 

# we are going to follow this 
# 
# 1)problem statement
# 
# 2)Data    given data contains utc attributes 
# 
# And it also has a version of kaggle https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# 
# 3)Evaluation   if we are being working with the 95%accuracy our model predicted was correct
# 
# 4)Features
# 
# 5)Modelling
# 
# 6)Experimentation
# 
# 

# #problem Defination
# we are given an heart disease dataset and we are trying tho find out wheather the person has heartdisease or not
# 
# age
# sex
# chest pain type (4 values)
# resting blood pressure
# serum cholestoral in mg/dl
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results (values 0,1,2)
# maximum heart rate achieved
# exercise induced angina
# oldpeak = ST depression induced by exercise relative to rest
# the slope of the peak exercise ST segment
# number of major vessels (0-3) colored by flourosopy
# thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

# #preparing the libraries
# now we are using pandas,numpy and matplotlib for the datascience visualization

# In[3]:


#preparing the model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#evaluate the model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[214]:


#model evaluation
from sklearn.model_selection import train_test_split,cross_val_score
#Hyperparameter
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#classification report
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve



# In[25]:


#load_data
heart_disease=pd.read_csv("heart-disease.csv")
#view the data
heart_disease


# In[27]:


heart_disease.shape #we are having 303 columns and 14 rows


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

# In[34]:


#checking the top data
heart_disease.head()


# In[30]:


#checking down data
heart_disease.tail()


# In[36]:


v=heart_disease['target'].value_counts()
v


# In[39]:


#plotting the v means the person having heart_disease or not in the bar graph
v.plot(kind='bar',color=['salmon','blue'])


# In[40]:


#gathering the information about the data
heart_disease.info()


# In[42]:


heart_disease.isna().sum()


# In[43]:


#describing about the data
heart_disease.describe()


# Heart_disease frequaency according to the sex
# 

# In[45]:


heart_disease.sex.value_counts()


# In[46]:


#compare the sex with the target
pd.crosstab(heart_disease.target,heart_disease.sex)


# In[56]:


#creating crosstab plot

pd.crosstab(heart_disease.target, heart_disease.sex).plot(kind='bar', figsize=(10,6), color=['red','green'])
plt.title("Heart Disease in the male and female")
plt.xlabel('0=No Disease,1=Disease')
plt.ylabel("Amount")
plt.legend(["Female","male"])
plt.xticks()


# In[57]:


heart_disease.head(3)


# In[59]:


heart_disease['thalach'].value_counts()


# In[61]:


pd.crosstab(heart_disease.thalach,heart_disease.target)


# Age v/s Maximum heartrate

# In[78]:


plt.figure(figsize=(10,6))
#scatter with positive rate 
plt.scatter(heart_disease.age[heart_disease.target==1],heart_disease.thalach[heart_disease.target==1])
plt.scatter(heart_disease.age[heart_disease.target==0],heart_disease.thalach[heart_disease.target==0],c=['red'])
plt.title("hear disease present or not")
plt.xlabel("age")
plt.ylabel('Max heart_rate')
plt.legend(['Disease','No_Disease'])


# In[80]:


#check distribution[->its is another distribution of the data] of age
heart_disease.age.plot.hist()


# In[81]:


heart_disease.head(2)


# In[82]:


heart_disease['cp'].value_counts()


# Heart disease frequency per chestpain

# In[84]:


pd.crosstab(heart_disease.cp,heart_disease.target)


# In[89]:


#visualze the Heartdisease frequency
pd.crosstab(heart_disease.cp,heart_disease.target).plot(kind='bar',figsize=(10,6),color=['blue','black'])
plt.title("chestpain wise heart_disease prediction")
plt.xlabel("chestpain")
plt.ylabel("amount")
plt.legend(['no_pain','pain'])
plt.xticks(rotation=0)


# using corelation matrix->it tells us independent variable is related to eachother

# In[90]:


heart_disease.corr()


# In[94]:


#makking the corelation box more pretty
corr_matrix=heart_disease.corr()
fig,ax=plt.subplots(figsize=(10,6))
ax=sns.heatmap(corr_matrix,
                annot=True,
               linewidths=0.5,
               fmt='.2f',
               cmap='YlGnBu')
                                             
                                              


# 5.0 machinelearning modelling

# In[99]:


#splitting the data in to x and y
X=heart_disease.drop('target',axis=1)
y=heart_disease['target']
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[100]:


X_train


# In[101]:


y_train


# In[102]:


X_test


# In[104]:


y_test


# In[105]:


len(X_test)


# In[106]:


len(X_train)


# In[107]:


len(y_test),len(y_train)


# #now we are going to trin the model
# 
# classify the model based on tthe classification
# 
# predict the model
# 
# we are going to use
# 
# logistic regression
# 
# k-neighbours
# 
# Randomforestclassifier

# In[146]:


#using dictionary
models={'logistic':LogisticRegression(),
        'kneighbor':KNeighborsClassifier(),
        'RandomForestclassifier':RandomForestClassifier()
       }
def train_test_score(models, X_train, y_train, X_test, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():  #name->key and model->value like regression models
        # Fitting the model
        model.fit(X_train, y_train)
        # Evaluating the model score and storing it in model_scores dictionary
        model_scores[name] = model.score(X_test, y_test) #model[name]->its is the individual key
    return model_scores




# In[147]:


model_scores=train_test_score(models=models,X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test)
model_scores


# In[148]:


#model_comparison
model_compare=pd.DataFrame(model_scores,index=['accuracy'])
model_compare.T.plot.bar();


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

# In[152]:


#Hyperparameter turning
#lets tune for the knn
train_model=[]
test_model=[]
#chance for turing the neighbours
neighbours=range(1,21)
knn= KNeighborsClassifier()
#lets loop thriugh the neighbours
for i in  neighbours:
    knn.set_params(n_neighbors=i)
    #fitting the model
    knn.fit(X_train,y_train)
    #appending to the train_model
    train_model.append(knn.score(X_train,y_train))
    #updating the score of the test_model
    test_model.append(knn.score(X_test,y_test))


# In[154]:


train_model


# In[155]:


test_model


# In[204]:


plt.plot(neighbours,train_model,label="Train_model")
plt.plot(neighbours,test_model,label="Test_model")
plt.xticks(np.arange(1,21,1))
plt.xlabel('number of neighbours')
plt.ylabel("amount")
plt.legend()
print(f"maximum testscore{max(test_model)*100:.2f}%")


# #hyperparameter turning using randomizedsearchcv
# we are going to tune the following
# 
# *logisticRegressionModel()
# 
# *RandomForestClassifier()
# 

# In[192]:


#creating logistic_regression turning
logis_turning = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}
#Hyperparameter for RandomForestClassifier
random_class = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}


# turning the logisticregression using randomizedserchcv

# In[193]:


np.random.seed(42)
#setup hyperparameter for logisticregression
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=logis_turning, n_iter=20, verbose=True)
rs_log_reg.fit(X_train, y_train)


# In[194]:


rs_log_reg.best_params_


# In[195]:


rs_log_reg.score(X_test,y_test)


# Turning for the RandomForestClassifier

# In[196]:


np.random.seed(42)
rn_rand_class=RandomizedSearchCV(RandomForestClassifier(), param_distributions=random_class,cv=5, n_iter=20, verbose=True)


# In[197]:


rn_rand_class


# In[203]:


rn_rand_class.fit(X_train,y_train)


# In[201]:


rn_rand_class.best_params_


# In[202]:


rn_rand_class.score(X_test,y_test)


# Hyperturning using Gridsearchcv
# 
# since our logisticregression provides the best model when compared to other model we are using
# GridSearchCv

# In[205]:


logis_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}
log_grid=GridSearchCV(LogisticRegression(),param_grid=logis_grid,cv=5,verbose=True)


# In[207]:


log_grid.fit(X_train,y_train)


# In[208]:


log_grid.score(X_test,y_test)


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

# In[209]:


#make predictions 
y_preds=log_grid.predict(X_test)


# In[210]:


y_preds


# In[211]:


y_test


# In[227]:


# Get predicted probabilities for the positive class
y_pred_proba = log_grid.predict_proba(X_test)[:, 1]

# Calculate false positive rate (fpr), true positive rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

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


# In[220]:


print(confusion_matrix(y_preds,y_test))


# In[225]:


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


# In[ ]:





# In[228]:


classification_report(y_test,y_preds)


# calculate the metrics using cross validation
# we are going through calculate the precision_score,recall,f1_score

# In[231]:


log_grid.best_params_


# In[234]:


clf=LogisticRegression(C= 0.23357214690901212,solver='liblinear')


# In[236]:


#cross_validated accuracy
cv_acc=cross_val_score(clf,X,y,cv=5,scoring='accuracy')
cv_acc


# In[237]:


cv_acc=np.mean(cv_acc)
cv_acc


# In[239]:


#cross_validated precission
cv_precision=cross_val_score(clf,X,y,cv=5,scoring='precision')
cv_precision


# In[243]:


cv_precision=np.mean(cv_precision)
cv_precision


# In[244]:


#cross_validated recall

cv_recall=cross_val_score(clf,X,y,cv=5,scoring='recall')
cv_recall=np.mean(cv_recall)
cv_recall


# In[246]:


#cross_validated f1_score
#cross_validated accuracy
cv_f1=cross_val_score(clf,X,y,cv=5,scoring='f1')
cv_f1=np.mean(cv_f1)
cv_f1


# In[249]:


#visualise the cross_validated metrics
cv_metrics=pd.DataFrame({"accuracy":cv_acc,
                        "precision":cv_precision,
                        'recall':cv_recall,
                        'f1':cv_f1},index=[0])
cv_metrics.T.plot.bar(title='cross validation_metrics')


# ###feature importance

# #features importance
# contribute that whivh feature gives the most of the outcome

# In[250]:


#fitting an instance of logistic_regression
log_grid.best_params_


# In[253]:


clf=LogisticRegression(C= 0.23357214690901212, solver= 'liblinear')
clf.fit(X_train,y_train)


# In[254]:


clf.coef_


# In[256]:


#matching the coef_
feature_dict=dict(zip(heart_disease.columns,list(clf.coef_[0])))
feature_dict


# In[258]:


#visualise the features
feature_visualise=pd.DataFrame(feature_dict,index=[0])
feature_visualise.T.plot.bar(title="feature importance",legend=False)


# In[259]:


pd.crosstab(heart_disease.slope,heart_disease.target)


# #Experimentation

# In[262]:


#Exporting the model
import pickle

# Assuming you have a trained model named 'model'
# Train your model here...

# Save the trained model to a file using pickle
with open('heart_disease_project.pkl', 'wb') as file:
    pickle.dump(heart_disease, file)


# In[ ]:




