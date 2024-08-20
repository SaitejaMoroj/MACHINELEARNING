#!/usr/bin/env python
# coding: utf-8

# creating the model model to predict the wheather the person consistist of diabets or not

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# File Names and format:
# (1) Date in MM-DD-YYYY format
# (2) Time in XX:YY format
# (3) Code
# (4) Value
# 
# The Code field is deciphered as follows:
# 
# 33 = Regular insulin dose
# 34 = NPH insulin dose
# 35 = UltraLente insulin dose
# 48 = Unspecified blood glucose measurement
# 57 = Unspecified blood glucose measurement
# 58 = Pre-breakfast blood glucose measurement
# 59 = Post-breakfast blood glucose measurement
# 60 = Pre-lunch blood glucose measurement
# 61 = Post-lunch blood glucose measurement
# 62 = Pre-supper blood glucose measurement
# 63 = Post-supper blood glucose measurement
# 64 = Pre-snack blood glucose measurement
# 65 = Hypoglycemic symptoms
# 66 = Typical meal ingestion
# 67 = More-than-usual meal ingestion
# 68 = Less-than-usual meal ingestion
# 69 = Typical exercise activity
# 70 = More-than-usual exercise activity
# 71 = Less-than-usual exercise activity
# 72 = Unspecified special event 

# In[2]:


#loading the dataset
diabets_data=pd.read_csv("Downloads/diabetes.csv")


# In[3]:


diabets_data


# In[4]:


len(diabets_data)


# In[5]:


diabets_data['Outcome']


# In[6]:


# Assuming diabets_data is your DataFrame
crosstab_result = pd.crosstab(diabets_data['Outcome'] == 1, diabets_data['Outcome'] == 0).astype(int)
crosstab_result


# In[7]:


len(diabets_data.Outcome)


# In[8]:


crosstab_result.plot(kind='bar')
plt.xlabel("diabetic_count")
plt.ylabel('amount')
plt.legend(['Diabetic','Non_Diabetic'])
plt.title('Diabetic Counts')
plt.show()


# In[9]:


len(diabets_data[diabets_data['Outcome'] == 0])


# In[10]:


diabets_data.Age.value_counts()


# In[11]:


x=pd.crosstab(diabets_data['Age'],diabets_data['Outcome'])


# In[12]:


len(x)


# In[13]:


plt.scatter(x=diabets_data['Age'],y=diabets_data['Outcome'])


# In[14]:


diabets_data.head()


# In[15]:


diabets_data.tail()


# In[16]:


#evaluating on the pregnency
diabets_data.Pregnancies.value_counts()


# In[17]:


len(diabets_data.Pregnancies)


# In[18]:


new_cross=pd.crosstab(diabets_data.Pregnancies,diabets_data.Outcome)


# In[19]:


new_cross.plot(kind='bar')


# In[20]:


#checking for the regular insulin level
diabets_data.Insulin.value_counts()


# In[25]:


pd.crosstab(diabets_data.Insulin,diabets_data.Outcome)


# In[21]:


plt.scatter(x=diabets_data['Insulin'],y=diabets_data['Outcome'])


# In[22]:


#plotting the insulin and outcome using histgram
plt.hist(diabets_data['Insulin'], bins=10) 
plt.hist(diabets_data['Outcome'], bins=10) 


# In[23]:


diabets_data.describe()


# In[24]:


diabets_data.info()


# In[25]:


#checking for the null values
diabets_data.isnull().sum()


# In[26]:


#no null values are present
#comparing with  each column to the target outcome
diabets_data.head()


# In[27]:


#working on the glucose column
diabets_data.Glucose.value_counts()


# In[28]:


b=pd.crosstab(diabets_data.Glucose,diabets_data.Outcome)


# In[29]:


b.plot(kind='bar')
plt.legend(['Non_diabetic','Diabetic'])
plt.title("Glucose v/s Outcome")
plt.show()


# #comparing the bp with the outcome

# In[30]:


diabets_data.BloodPressure.value_counts()


# In[31]:


bp=pd.crosstab(diabets_data.BloodPressure,diabets_data.Outcome)
bp


# In[33]:


plt.scatter(x=diabets_data['BloodPressure'],y=diabets_data['Outcome'])
plt.xlabel('Representation based on bp')
plt.ylabel('amount')


# In[34]:


#representation on bar
bp.plot(kind='bar')
plt.xlabel('Bp representation')
plt.ylabel('amount')
plt.legend(['Non_diabetic','Diabetiic'])
plt.title('Representation of Diabetics')
plt.show()


# In[35]:


#checking the diabetics and non_diabetics in all
diabets_data.groupby('Outcome').mean()


# In[36]:


#classifying the data in to x and y
X=diabets_data.drop('Outcome',axis=1)
y=diabets_data['Outcome']


# In[37]:


X


# In[38]:


y


# Applying the ml algorithm

# In[39]:


from sklearn .svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[41]:


X_train,len(X_train)


# In[42]:


y_train,len(y_train)


# In[43]:


X_test


# In[44]:


y_test


# In[49]:


len(y_test)


# Data Standardization

# In[45]:


scaler=StandardScaler()
standard_scaler=scaler.fit_transform(X)


# In[46]:


print(standard_scaler)


# In[47]:


#view using the pandas
pd.DataFrame(standard_scaler)


# In[48]:


X=standard_scaler
y=diabets_data['Outcome']  


# again splitting the standard_data in to X_train,y_train

# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[50]:


X


# In[51]:


y


# In[52]:


X.shape,y.shape,X_train.shape,y_train.shape


# In[53]:


#applying the machinelearning algorithm first we create an python function

model={'svm':SVC(),
       'KNeighborsClassifier':KNeighborsClassifier(),
       'RandomForestClassifier':RandomForestClassifier()}
def modeling(models, X_train, y_train, X_test, y_test):
    np.random.seed(42)
    modelss = {}
    for model_name, model_instance in models.items():  # Use .items() instead of .item()
        model_instance.fit(X_train, y_train)
        modelss[model_name] = model_instance.score(X_test, y_test)
        # Append as a tuple
    return modelss


# In[54]:


model=modeling(models=model,X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test)
model


# In[55]:


#model_comparison
model_compare=pd.DataFrame(model,index=['accuracy'])
model_compare.T.plot.bar(color=['green'])


# when we are using the logistic_regression

# In[56]:


from sklearn.linear_model import LogisticRegression
np.random.seed(42)
models= LogisticRegression()
models.fit(X_train,y_train)
models.score(X_test,y_test)


# In[89]:


#turning the model by importing the Hyperparameters
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
#importing the evaluation_metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve


# Evaluating  Hyperparameters on the KNN

# In[58]:


model_trained=[]
model_test=[]
knn=KNeighborsClassifier()

neighbours=range(1,21)
for i in neighbours: 
    
    knn.set_params(n_neighbors=i)
    knn.fit(X_train,y_train)
    model_trained.append(knn.score(X_train,y_train))
    model_test.append(knn.score(X_test,y_test))
    


# In[59]:


model_trained


# In[60]:


model_test


# In[61]:


plt.plot(neighbours,model_trained,label="Train_model")
plt.plot(neighbours,model_test,label="Test_model")
plt.xticks(np.arange(1,21,1))
plt.xlabel('number of neighbours')
plt.ylabel("amount")
plt.legend()
print(f"maximum testscore{max(model_test)*100:.2f}%")


# Applying hyperparameter RandomSearchcv to RandomForest,logistic and Svm
# 
# creating hyperparameter for logisticregression

# In[62]:


logistic_turning={'C':np.logspace(-4, 4, 20),
                  'solver':['liblinear']}


# In[63]:


#checking for the randomforestclassifier parameters
RandomForestClassifier().get_params()


# In[64]:


#RandomForestClassifier
# Define the parameter grid for RandomForestClassifier
rf_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}


# In[65]:


#using RandomSearchCV
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=logistic_turning, n_iter=20, verbose=True)
rs_log_reg.fit(X_train, y_train)


# In[66]:


rs_log_reg.best_params_


# In[67]:


rs_log_reg.score(X_test,y_test)


# RandomForestTurning

# In[68]:


rs_rf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rf_params, n_iter=100, cv=3, random_state=42, n_jobs=-1)


# In[69]:


rs_rf


# In[70]:


rs_rf.fit(X_train,y_train)


# In[72]:


rs_rf.score(X_test,y_test)


# Hyperparameter turning for Svm

# In[73]:


param_grid1 = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 10, 100]
}
rs_rf1= RandomizedSearchCV(estimator=SVC(), param_distributions=param_grid1, n_iter=100, cv=3, random_state=42, n_jobs=-1)


# In[74]:


rs_rf1


# In[75]:


rs_rf1.fit(X_train,y_train)


# In[76]:


rs_rf1.score(X_test,y_test)


# Using gridsearchcv  for the machinelearning model

# In[77]:


logis_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}
log_grid=GridSearchCV(LogisticRegression(),param_grid=logis_grid,cv=5,verbose=True)


# In[78]:


log_grid


# In[79]:


log_grid.fit(X_train,y_train)


# In[80]:


log_grid.score(X_test,y_test)


# Gridsearch for the RandomSearchcv

# In[81]:


rf_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
rand_grid=GridSearchCV(RandomForestClassifier(),param_grid=rf_params,cv=5,verbose=True)


# In[82]:


rand_grid.fit(X_train,y_train)


# In[ ]:


rand_grid


# In[83]:


rand_grid.score(X_test,y_test)


# In[87]:


#Hypertext paramter for the SVC
param_grid2 = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 10, 100]
}

rand_griding = GridSearchCV(SVC(), param_grid=param_grid2, cv=5, verbose=True)


# In[88]:


rand_griding.fit(X_test,y_test)
rand_griding.score(X_train,y_train)


# In[91]:


#dealing with the permormancemetrics
#we predict the X_test with Y_test
rand_griding.fit(X_train, y_train)
y_preds = rand_griding.predict(X_test)



# In[92]:


y_preds


# In[95]:


#Dealing with the classification_report
classify=confusion_matrix(y_preds,y_test)
classify


# In[98]:


import seaborn as sns
sns.heatmap(classify, annot=True, fmt="d", cmap="Blues")

# Add labels and title
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[100]:


#preparing the classification_report
report=classification_report(y_preds,y_test)
report


# In[102]:


#Dealing with the precsionscore
prec=precision_score(y_preds,y_test)
prec


# In[103]:


#Dealing with the recall

recall=recall_score(y_preds,y_test)
recall


# In[104]:


#Dealing with the f1_score
f1=f1_score(y_preds,y_test)
f1


# In[113]:


from sklearn.metrics import make_scorer

# Define the scoring metrics you want to use
scoring_metrics = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score, average='weighted'),
    'Recall': make_scorer(recall_score, average='weighted'),
    'F1': make_scorer(f1_score, average='weighted')
}

# Perform cross-validation and calculate each scoring metric
for metric_name, scorer in scoring_metrics.items():
    scores = cross_val_score(rand_griding, X_train, y_train, cv=5, scoring=scorer)
    print(f"{metric_name}: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")


# In[ ]:





# In[122]:


# Define the Random Forest Classifier
random_forest_classifier = RandomForestClassifier()

# Fit the classifier
random_forest_classifier.fit(X_train, y_train)

# Calculate predicted probabilities for the positive class
y_probs = random_forest_classifier.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


# In[123]:


#Dumping the code
import pickle

# Train your Random Forest Classifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, y_train)

# Save the trained classifier to a file
with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(random_forest_classifier, f)


# In[ ]:




