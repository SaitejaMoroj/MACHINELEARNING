#!/usr/bin/env python
# coding: utf-8

# Dealing with the breastcancer dataset

# # This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. This is one of three domains provided by the Oncology Institute that has repeatedly appeared in the machine learning literature. (See also lymphography and primary-tumor.)

# Attribute Information:
# 
# 1) ID number
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
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
# n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server:

# # importing libraries

# In[108]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns


# # loading the dataset

# In[109]:


breast_cancer=load_breast_cancer()
breast_cancer


# # data set loading
# 

# In[110]:


breast_cancer = load_breast_cancer()


# In[111]:


data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)


# In[112]:


data['target'] = breast_cancer.target


# In[113]:


data.head()


# In[114]:


data.tail()


# In[115]:


len(data)


# In[116]:


data.info()


# In[117]:


data.describe()


# In[118]:


#dealing with the everycolum
data.keys()


# In[119]:


#radius (mean of distances from center to points on the perimeter)
data['mean radius'].value_counts()


# In[120]:


pd.crosstab(data['mean radius'],data['target'])


# In[121]:


#display using the scatter plot
data_target_0 = data[data['target'] == 0]
data_target_1 = data[data['target'] == 1]
plt.scatter(data_target_0['mean radius'], [0]*len(data_target_0), color='blue', label='Target 0')
plt.scatter(data_target_1['mean radius'], [1]*len(data_target_1), color='orange', label='Target 1')
# Add labels and title
plt.xlabel('Mean Radius')
plt.ylabel('Target')
plt.title('Scatter Plot of Mean Radius vs Target')
plt.legend()
# Show plot
plt.show()


# In[122]:


data.head()


# In[123]:


data['mean concavity'].value_counts()


# In[66]:


#plotting using the scatter
pd.crosstab(data['mean concavity'],data['target'])


# In[124]:


#plotting using the scatter
data_target_0=data[data['target'] == 0]
data_target_0=data[data['target'] == 1]

plt.scatter(data_target_0['mean concavity'], [0]*len(data_target_0), color='blue', label='Target 0')
plt.scatter(data_target_1['mean concavity'], [1]*len(data_target_1), color='orange', label='Target 1')
plt.xlabel('mean concavity')
plt.ylabel('target')
plt.title('Representation between the concavity and target')
plt.show()


# In[125]:


#Dealing with the worst texture
pd.crosstab(data['worst texture'],data['target'])


# In[126]:


#plotting for the worst texture
data_target_0 = data[data['target'] == 0]
data_target_1 = data[data['target'] == 1]

# Create scatter plot for target == 0
plt.scatter(data_target_0['worst texture'], [0]*len(data_target_0), color='blue', label='Target 0')

# Create scatter plot for target == 1
plt.scatter(data_target_1['worst texture'], [1]*len(data_target_1), color='orange', label='Target 1')

# Add labels and title
plt.xlabel('Worst Texture')
plt.ylabel('Target')
plt.title('Scatter Plot of Worst Texture vs Target')
plt.legend()

# Show plot
plt.show()


# In[ ]:


#checking for the symmentry


# In[127]:


#using the ml model since we are being predcting the numerical values use are being using the Regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # splitting the data in to x and y

# In[128]:


x=data.drop('target',axis=1)
y=data['target']


# In[129]:


scaled=StandardScaler()
x_scaled = scaled.fit_transform(x)


# In[130]:


X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=42)


# In[131]:


clf= RandomForestRegressor()
clf.fit(X_train,y_train)


# In[132]:


clf.score(X_test,y_test)

classifying using the LogisticRegression

# In[134]:


reg=LogisticRegression()
np.random.seed(42)
reg.fit(X_train,y_train)
reg.score(X_test,y_test)


# # using  SGDRegressor

# In[136]:


reg1= SGDRegressor()
np.random.seed(42)
reg1.fit(X_train,y_train)
reg1.score(X_test,y_test)


# # using the linear regression

# In[137]:


reg_linear=LinearRegression()
np.random.seed(42)
reg_linear.fit(X_train,y_train)
reg_linear.score(X_test,y_test)


# # evaluate the same models using the functions
# 

# In[138]:


model={'RandomForestRegressor':RandomForestRegressor(),
     'LogisticRegression':LogisticRegression(),
     'SGDRegressor':SGDRegressor(),
      'LinearRegression':LinearRegression()}
def eval_model(model,X_train,X_test,y_test,y_train):
    np.random.seed(42)
    model_score={}
    for models,model_items in model.items():
        model_items.fit(X_train,y_train)
        model_score[models]=model_items.score(X_test,y_test)
    return model_score


# In[139]:


new_model=eval_model(model=model,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
new_model


# In[140]:


plt.figure(figsize=(10, 6))
plt.bar(new_model.keys(), new_model.values(), color=['blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Evaluation Scores')
plt.show()


# # checking for the corealtion

# In[141]:


data.corr


# In[142]:


correlation_matrix = data.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# # hyperparameter turning for the the models RandomforestRegressor
# 

# importing the hyperparameter

# In[143]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# In[148]:


param1 = {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [None, 2, 4, 6],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt']
}
param_model=RandomizedSearchCV(RandomForestRegressor(),param_distributions=param1,n_iter=10,verbose=True)


# In[149]:


param_model.fit(X_train,y_train)


# In[150]:


param_model.score(X_test,y_test)


# In[154]:


param_model.best_params_


# In[157]:


#Hyperparamneter for LogisticRegression
param2 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'], 
    'class_weight': [None, 'balanced']  
}
param_model2= RandomizedSearchCV (LogisticRegression(),param_distributions=param2,n_iter=10,verbose=True)


# In[158]:


param_model2.fit(X_train,y_train)


# In[159]:


param_model2.score(X_test,y_test)


# In[160]:


param_model.best_params_


# In[164]:


#Hyperparameter for SGD
from scipy.stats import uniform, randint
param3 = {
    'alpha': uniform(0.0001, 0.01), 
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 
    'penalty': ['l1', 'l2', 'elasticnet'], 
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],  
    'eta0': uniform(0.01, 1), 
    'max_iter': randint(100, 1000),  
    'tol': uniform(1e-5, 1e-3), 
    'epsilon': uniform(1e-5, 1e-3),  
    'shuffle': [True, False], 
    'early_stopping': [True, False]  
}

# Define RandomizedSearchCV
param_model3 = RandomizedSearchCV(SGDRegressor(), param_distributions=param3, n_iter=10, verbose=True)


# In[165]:


param_model3.fit(X_train,y_train)
param_model3.score(X_test,y_test)


# In[177]:


#linear regression
param4=  {
    'fit_intercept': [True, False],
    'copy_X': [True, False]
}

param_model4 = RandomizedSearchCV(LinearRegression(), param_distributions=param4, n_iter=10, verbose=True)


# In[178]:


param_model4.fit(X_train,y_train)


# In[180]:


param_model4.score(X_test,y_test)


# In[181]:


param_model4.best_params_


# # gridsearchcv hyperparameter

# In[182]:


param_grid= {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [None, 2, 4, 6],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt']
}
param_modelgrid=GridSearchCV(RandomForestRegressor(), param_grid=param_grid,verbose=True)


# In[183]:


param_modelgrid.fit(X_train,y_train)


# In[184]:


param_modelgrid.score(X_test,y_test)


# In[186]:


#gridparam for the logisticregression
param2_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'], 
    'class_weight': [None, 'balanced']  
}
param_modelgrid1=GridSearchCV(LogisticRegression(), param_grid=param2_grid,verbose=True)


# In[188]:


param_modelgrid1.fit(X_train,y_train)


# In[189]:


param_modelgrid1.score(X_train,y_train)


# In[190]:


param_modelgrid1.score(X_test,y_test)


# In[198]:


#Dealing with the SGD
param_grid = {
    'alpha': [uniform(0.0001, 0.01).rvs()],
    'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive', 'huber', 'squared_error'],
    'penalty': ['l1', 'l2'],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [uniform(0.01, 1).rvs()],
    'max_iter': [randint(100, 1000).rvs()],
    'tol': [uniform(1e-5, 1e-3).rvs()],
    'epsilon': [uniform(1e-5, 1e-3).rvs()],
    'shuffle': [True, False],
    'early_stopping': [True, False]
}

param_modelgrid2=GridSearchCV(SGDRegressor(), param_grid=param_grid,verbose=True)


# In[199]:


param_modelgrid2.fit(X_train,y_train)


# In[200]:


param_modelgrid2.score(X_test,y_test)


# In[201]:


#linear regression
param4_grid=  {
    'fit_intercept': [True, False],
    'copy_X': [True, False]
}
param_modelgrid3=GridSearchCV(LinearRegression(),param_grid=param4_grid,verbose=True)


# In[202]:


param_modelgrid3.fit(X_train,y_train)


# In[203]:


param_modelgrid3.score(X_test,y_test)


# # Evaluation metrics

# In[204]:


y_preds=param_modelgrid1.predict(X_test)


# In[205]:


y_preds


# In[208]:


from sklearn.metrics import accuracy_score
x=accuracy_score(y_preds,y_test)
print(f"Accuracy: {x:.2f}%")


# In[218]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[224]:


x=mean_squared_error(y_preds,y_test)
x


# In[225]:


y=mean_absolute_error(y_preds,y_test)
y


# In[226]:


z=r2_score(y_preds,y_test)
z


# In[227]:


scoring=[x,y,z]


# In[228]:


scoring


# In[231]:


metrics = ['mean_squared_error', 'mean_absolute_error', 'r2_score']

plt.figure(figsize=(10, 6))
plt.bar(metrics, scoring, color=['red', 'blue', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Evaluation Scores')
plt.show()


# # importing the model

# In[233]:


import joblib
joblib.dump(model, 'breast_cancer.pkl')

# Load model
loaded_model = joblib.load('breast_cancer.pkl')


# In[245]:


print(loaded_model)


