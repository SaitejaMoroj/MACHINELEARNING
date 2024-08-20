#!/usr/bin/env python
# coding: utf-8

# # # problem statement
# This dataset is taken from eBay using web API. It consists of not only new, factory sealed product posts, it also includes refurbished or second hand products. It has 3981 different rows and 10 distinctive columns.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


prices=pd.read_csv('Downloads/archive (4)/raw_ebay.csv')
prices


# In[3]:


prices.info()


# In[4]:


prices.shape


# In[5]:


prices.duplicated().sum()


# In[6]:


#plotting with laptop brand and price
fig,ax=plt.subplots()
ax.scatter(prices['Brand'][:100],prices['Price'][:100])


# In[7]:


prices.info()


# In[8]:


prices.head().T


# In[9]:


prices.info()


# In[13]:


#converting the object in to the categories
for label, content in prices.items():
    if not pd.api.types.is_numeric_dtype(content):
        prices[label]=content.astype('category')
        print(label)


# In[14]:


prices[prices['Processor']=='Intel Core i5 7th Gen']['Processor'].value_counts().plot(kind='hist')


# In[15]:


prices[prices['Condition']=='Very Good - Refurbished']['Condition'].value_counts().plot(kind='bar')


# In[16]:


prices[prices['Condition'] == 'New']['Condition'].value_counts().plot(kind='bar')


# In[17]:


#plotting for the openbox
prices[prices['Condition']=='Excellent - Refurbished']['Condition'].value_counts().plot(kind='hist')


# In[18]:


prices


# In[19]:


prices.info()


# In[20]:


sns.barplot(x=prices['Brand'],y=prices['Price'])
plt.xticks()


# In[21]:


prices


# In[22]:


prices['GPU'].value_counts()


# In[23]:


#cjecking their is any null value in the GPU
prices['GPU'].isna().sum()


# In[24]:


prices['Intel HD Graphics']=prices['GPU'].apply(lambda x:1 if 'Intel HD Graphics' in x else 0)
prices['Intel HD Graphics']


# In[25]:


prices.head(10)


# In[26]:


prices['Intel HD Graphics'].value_counts().plot(kind='bar')


# In[27]:


#plotting using seaborn
sns.barplot(x=prices['Intel HD Graphics'],y=prices['Price'])


# In[28]:


prices['Intel Iris Xe Graphics']=prices['GPU'].apply(lambda  x:1 if 'Intel Iris Xe Graphics' in x else 0)


# In[29]:


prices


# In[30]:


#plotting using seaborn
sns.barplot(x=prices['Intel Iris Xe Graphics'],y=prices['Price'])


# In[35]:


prices['8gb RAm']=prices["RAM"].apply(lambda x:1 if '8' in x else 0)
prices['16gb RAm']=prices["RAM"].apply(lambda x:1 if '16' in x else 0)
prices['32gb RAm']=prices["RAM"].apply(lambda x:1 if '32' in x else 0)


# In[36]:


prices


# In[38]:


#plotting based upon the ram value
prices['8gb RAm'].value_counts().plot(kind='bar')


# In[40]:


prices['16gb RAm'].value_counts().plot(kind='bar')


# In[42]:


prices['32gb RAm'].value_counts().plot(kind='bar')


# In[43]:


#plotting using the seaborn
sns.scatterplot(x=prices['8gb RAm'],y=prices['Price'])


# In[45]:


sns.scatterplot(x=prices['16gb RAm'],y=prices['Price'])


# In[46]:


sns.scatterplot(x=prices['32gb RAm'],y=prices['Price'])


# # # feature enginnering on X and y for the resolution
# 

# In[47]:


new_resolution=prices['Resolution'].str.split('x',n=1,expand=True)


# In[48]:


prices['x_res']=new_resolution[0]
prices['y_res']=new_resolution[1]


# In[49]:


prices


# In[58]:


prices['x_res']=prices['x_res'].astype('category')
prices['y_res']=prices['y_res'].astype('category')


# In[59]:


prices.info()


# In[60]:


prices['x_res'].dtype


# In[61]:


prices


# In[79]:


#classify the processor
def processor(text):
    if text == 'Intel Core i5' or text == 'Intel Core i5 7th Gen' or text == 'i7' or text == 'Intel Core i5':
        return text
    else:
        if text.split()[0] != 'Intel':
            return 'Other Intel processor'  # Changed print to return
        else:
            return 'Other processor'  # Indentation fixed




# In[80]:


#applying to an newcolumn that processor
prices['CPU']=prices['Processor'].apply(processor)


# In[81]:


prices


# In[82]:


prices['CPU'].value_counts().plot(kind='bar')


# In[87]:


sns.barplot(x=prices['CPU'],y=prices["Price"])
plt.xticks(rotation='vertical')
plt.show()


# In[88]:


prices.drop('Processor',axis=1,inplace=True)


# In[89]:


prices


# In[90]:


prices['RAM'].value_counts().plot(kind='bar')


# In[97]:


#Removing extra created RAM
prices.drop('8gb RAm',axis=1,inplace=True)



# In[99]:


prices.drop('16gb RAm',axis=1,inplace=True)


# In[100]:


prices.drop('32gb RAm',axis=1,inplace=True)


# In[101]:


prices


# In[103]:


sns.barplot(x=prices['RAM'],y=prices['Price'])
plt.xticks(rotation='vertical')


# In[104]:


prices['Condition'].value_counts()


# In[105]:


prices['Condition'].value_counts().plot(kind='bar')


# In[108]:


sns.barplot(x=prices['Condition'],y=prices['Price'])
plt.xticks(rotation='vertical')


# # classfication of data bases on the condition

# In[111]:


prices['New']=prices['Condition'].apply(lambda x:1 if 'New' in x else 0)
prices['Open box']=prices['Condition'].apply(lambda x:1 if 'Open box' in x else 0)
prices['Excellent - Refurbished']=prices['Condition'].apply(lambda x:1 if 'Excellent - Refurbished' in x else 0)
prices['Very Good - Refurbished']=prices['Condition'].apply(lambda x:1 if 'Very Good - Refurbished' in x else 0)
prices['Good - Refurbished']=prices['Condition'].apply(lambda x:1 if 'Good - Refurbished' in x else 0)
prices['unkmown']=prices['Condition'].apply(lambda x:1 if '__' in x else 0)
prices['Used']=prices['Condition'].apply(lambda x:1 if 'Used' in x else 0)


# In[113]:


prices.drop('Condition',axis=1,inplace=True)


# In[115]:


prices.drop('Resolution',axis=1,inplace=True)


# In[116]:


prices


# In[118]:


prices['GPU'].value_counts()


# In[120]:


prices['GPU'].value_counts().plot(kind='bar')
plt.xticks(rotation='vertical')


# In[122]:


#Classify the data based up on the brand
prices['GPU BRAND']=prices['GPU'].apply(lambda x:x.split()[0])


# In[124]:


prices['GPU BRAND'].value_counts()


# In[125]:


#checking the laptop brand bases up on the laptop
prices[prices['GPU BRAND']=="AMD"]


# In[131]:


sns.barplot(x=prices['GPU BRAND'],y=prices['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[135]:


prices.drop('GPU',axis=1,inplace=True)


# In[136]:


prices


# In[137]:


#working on the product discription
prices['Product_Description'].value_counts()


# In[138]:


len(prices['Product_Description'])


# In[139]:


#classfying the data on the first index value
prices['product']=prices['Product_Description'].apply(lambda x:x.split()[0])


# In[142]:


prices['product'].value_counts()


# In[143]:


prices['product'].value_counts().plot(kind='bar')


# In[146]:


sns.barplot(x=prices['product'][:30],y=prices['Price'])


# In[147]:


prices.drop('Product_Description',axis=1,inplace=True)


# In[148]:


prices


# # Dealing with the GPU_type

# In[149]:


prices['GPU_Type'].value_counts()


# In[150]:


prices['GPU']=prices['GPU_Type'].apply(lambda x:x.split()[0])


# In[151]:


prices['GPU'].value_counts()


# In[153]:


prices['GPU'].value_counts().plot(kind='bar')


# In[157]:


#plotting using seaborn
sns.barplot(x=prices['GPU'],y=prices['Price'])
plt.xticks(rotation='vertical')


# In[158]:


prices.drop('GPU',axis=1,inplace=True)


# In[159]:


prices


# In[160]:


prices['GPU']=prices['GPU_Type'].apply(lambda x:x.split()[0])


# In[161]:


prices['GPU'].value_counts()


# In[162]:


prices.drop('GPU_Type',axis=1,inplace=True)


# In[163]:


prices


# In[164]:


#working on the screen time
prices['Screen_Size'].value_counts()


# In[166]:


prices['Screen_Size'].value_counts().plot(kind='bar')
plt.xticks(rotation='vertical')


# In[170]:


sns.barplot(x=prices['Screen_Size'][:100],y=prices['Price'])
plt.xticks(rotation='vertical')


# In[173]:


#convert the string values in the data to float values
import re

# Define a function to extract numeric values
def extract_numeric(value):
    match = re.search(r'\d+\.*\d*', value)
    if match:
        return float(match.group())
    else:
        return None

# Apply the function to the 'screentime' column
prices['Screen_Size'] = prices['Screen_Size'].apply(extract_numeric)


# In[175]:


prices['Screen_Size'].dtype


# In[187]:


sns.distplot(prices['Price'])


# In[179]:


numeric_prices = prices.select_dtypes(include=['float64', 'int64']) 
sns.heatmap(numeric_prices.corr())


# In[190]:


#splitting the data in to x and y
x=prices.drop('Price',axis=1)
y=prices['Price']


# In[191]:


x


# In[192]:


y


# In[194]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[195]:


X_train


# In[196]:


X_test


# In[197]:


y_train


# In[198]:


y_test


# # converting the categoricaldata in to numerical

# In[227]:


from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder


# # importing ml models 
# This model is based on the regression so we are importing the all regression models and checkig for the best for the value

# In[200]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor


# In[203]:


prices.drop('Intel HD Graphics',axis=1,inplace=True)


# In[205]:


prices.drop('Intel Iris Xe Graphics',axis=1,inplace=True)


# In[206]:


prices


# In[207]:


prices.info()


# # Data cleaning

# In[231]:


categorical_features=['Brand', 'RAM', 'x_res', 'y_res']
one_hot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

transformer=ColumnTransformer([('one_hot',one_hot,categorical_features)],remainder='passthrough')
transformed_x=transformer.fit_transform(prices)
transformed_x


# In[232]:


transformed_df.info()


# In[233]:


X_train, X_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2)


# In[234]:


X_train


# In[235]:


y_train


# # filling the missing values

# In[239]:


transformed_df.isna().sum()


# In[240]:


prices.isna().sum()


# In[241]:


prices.info()


# In[245]:


#filling the missing categorical variables
for label,content in prices.items():
    if not pd.api.types.is_numeric_dtype(content):
        prices[label]=content.astype.is_numeric.fillnana(content.mode())
        print(label)


# In[246]:


prices.isna().sum()


# In[248]:


#filling the integer missing value
for label,content in prices.items():
    if  pd.api.types.is_numeric_dtype(content):
        prices[label]=content.fillna(content.median())
        print(label)


# In[249]:


prices.info()


# In[253]:


#splitting now the data in to X and y
x=prices.drop('Price',axis=1)
y=prices['Price']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[254]:


X_train


# In[255]:


X_test


# In[256]:


y_train


# In[257]:


len(X_test)


# In[258]:


len(X_train)


# In[259]:


len(y_test)


# In[260]:


len(y_train)


# # LinearRegression

# In[261]:


#classfying the dta bases up on the model
model=LinearRegression()
model.fit(X_train,y_train)


# In[268]:


v=model.score(X_test,y_test)
print(f"model Performance:{v:.2f}%")


# # Ridge

# In[263]:


model1=Ridge()
model1.fit(X_train,y_train)


# In[266]:


p=model1.score(X_test,y_test)
print(f"Model performance: {p:.2f}%")


# #  Lasso

# In[269]:


model2= Lasso()
model2.fit(X_train,y_train)


# In[271]:


print(f"model's score {model2.score(X_test,y_test):.2f}%")


# In[456]:


def models(X_train, y_train, X_test, y_test):
    regression_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'MLPRegressor': MLPRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'XGBRegressor': XGBRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor()
    }
    
    for label, model in regression_models.items():
        model.fit(X_train, y_train)  # Get the score
        print(f"{label} performance: {score:.2f}")


# In[457]:


model=models(X_train,y_train,X_train,y_test)


# In[465]:


scores = models(X_train, y_train, X_test, y_test)


# In[467]:





# # Hyperparameterturning

# In[ ]:





# In[367]:


import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.model_selection import GridSearchCV
def scores(X_test,y_test,X_train,y_train):
    regression_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'MLPRegressor': MLPRegressor(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'XGBRegressor': XGBRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor()
        }

    # Define the parameter grids for each model
    param_grid = {
        'LinearRegression': {},
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.1, 1.0, 10.0]},
        'SVR': {'C': [0.1, 1.0, 10.0], 'gamma': [0.1, 1.0, 10.0]},
        'KNeighborsRegressor': {'n_neighbors': [3, 5, 7]},
        'DecisionTreeRegressor': {'max_depth': [None, 10, 20]},
        'RandomForestRegressor': {'n_estimators': [50, 100, 200]},
        'GradientBoostingRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,), (50, 50)]},
        'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'ExtraTreesRegressor': {'n_estimators': [50, 100, 200]}
    }

    best_models = {}


    for model_name, model in regression_models.items():
        grid_search = GridSearchCV(model, param_grid=param_grid[model_name], cv=5, scoring='r2', verbose=True)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_

    for model_name, best_model in best_models.items():
        print(f"Best hyperparameters for {model_name}: {best_model.get_params()}")


# # Gridsearchcv

# In[449]:


import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.model_selection import GridSearchCV

def scores(X_test, y_test, X_train, y_train):
    regression_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'MLPRegressor': MLPRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'XGBRegressor': XGBRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor()
    }

    # Define the parameter grids for each model
    param_grid = {
        'LinearRegression': {},
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.1, 1.0, 10.0]},
        'SVR': {'C': [0.1, 1.0, 10.0], 'gamma': [0.1, 1.0, 10.0]},
        'KNeighborsRegressor': {'n_neighbors': [3, 5, 7]},
        'DecisionTreeRegressor': {'max_depth': [None, 10, 20]},
        'RandomForestRegressor': {'n_estimators': [50, 100, 200]},
        'GradientBoostingRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,), (50, 50)]},
        'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'ExtraTreesRegressor': {'n_estimators': [50, 100, 200]}
    }
    np.random.seed(42)
    for model_name, model in regression_models.items():
        print(f"Training {model_name}...")
        grid_search = GridSearchCV(model, param_grid=param_grid[model_name], cv=5, verbose=True)
        grid_search.fit(X_train, y_train)
        new_score = grid_search.score(X_test, y_test)
        print(grid_search.best_params_)
        print(f"Test Score for {model_name}: {new_score:.2f}%")

   


# In[450]:


new_score=scores(X_train,y_train,X_test,y_test)
print(new_score)


# In[ ]:





# In[451]:


import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
def model_scores(X_train, y_train, X_test, y_test):
    regression_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'MLPRegressor': MLPRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'XGBRegressor': XGBRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor()
    }

    param_grid = {
        'LinearRegression': {},
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.1, 1.0, 10.0]},
        'SVR': {'C': [0.1, 1.0, 10.0], 'gamma': [0.1, 1.0, 10.0]},
        'KNeighborsRegressor': {'n_neighbors': [3, 5, 7]},
        'DecisionTreeRegressor': {'max_depth': [None, 10, 20]},
        'RandomForestRegressor': {'n_estimators': [50, 100, 200]},
        'GradientBoostingRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,), (50, 50)]},
        'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        'ExtraTreesRegressor': {'n_estimators': [50, 100, 200]}
    }
    np.random.seed(42)
   
    for model_name, model_instance in regression_models.items():
        print(f"Training {model_name}...")
        param_search = RandomizedSearchCV(model_instance, param_distributions=param_grid[model_name], n_iter=10, verbose=1, n_jobs=-1, cv=5)
        param_search.fit(X_train, y_train)
        test_score = param_search.score(X_test, y_test)
        print(f'{test_score:.2f}%')


# In[452]:


scores=model_scores(X_train,y_train,X_test,y_test)
print(scores)


# # As compared to the above models GradientBoostingRegressor makes more percentage of predction

# In[468]:


new_param={'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
param_grid=RandomizedSearchCV(GradientBoostingRegressor(),param_distributions=new_param,n_iter=10,cv=5,verbose=True)
           


# In[469]:


param_grid.fit(X_train,y_train)


# In[470]:


param_grid.score(X_test,y_test)


# In[471]:


param_grid.best_params_


# # Evaluation Metrics

# In[472]:


#turning X_test to y_preds
y_preds=param_grid.predict(X_test)


# In[474]:


y_preds


# In[485]:


x=r2_score(y_preds,y_test)
print(x)


# In[486]:


y=mean_squared_error(y_preds,y_test)
print(y)


# In[487]:


z=mean_absolute_error(y_preds,y_test)
print(z)


# In[488]:


scoring=[x,y,z]


# In[490]:


metrics = ['mean_squared_error', 'mean_absolute_error', 'r2_score']

plt.figure(figsize=(8, 6))
plt.bar(metrics, scoring, color=['red', 'blue', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Evaluation Scores')
plt.show()


# # Exploring the model

# In[495]:


import joblib
joblib.dump(model, 'laptop_price_predction.pkl')

# Load model
loaded_model = joblib.load('laptop_price_predction.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:








# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




