#!/usr/bin/env python
# coding: utf-8

# In[1]:


#just load and import our need library
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# # DATA LOAD  - BIGMART INTO DATAFRAME

# In[2]:


# Load / Read the data
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
train.head(10)


# In[13]:


train['Item_Type'] = train['Item_Type'].astype('category')
train['Outlet_Size'] = train['Outlet_Size'].astype('category')
train['Otlet_Location_Type'] = train['Outlet_Location_Type'].astype('category')
train['Outlet_Type'] =train['Outlet_Type'].astype('category')
train['Item_Type'].dtype


# In[12]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(train.Item_Outlet_Sales,train.Item_Type,rwidth=0.8)
plt.xlabel("Price distribution")
plt.ylabel("Count") 


# In[15]:


category_counts = train['Item_Type'].value_counts()


# In[16]:


category_counts


# In[ ]:





# In[ ]:





# In[14]:


import matplotlib.pyplot as plt

# Assuming train.Item_Outlet_Sales is your numeric data
# and train.Item_Type is your categorical data

# Count the occurrences of each category
category_counts = train['Item_Type'].value_counts()

# Create a bar chart
plt.bar(category_counts.index, category_counts)

# Add labels and title
plt.xlabel('Item Type')
plt.ylabel('Count')
plt.title('Distribution of Item Types')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1437]:


#understaing the our dataset, remove unwanted columns 
train1 = train.drop(['Item_Identifier','Outlet_Identifier'], axis = 'columns')
test1 = test.drop(['Item_Identifier','Outlet_Identifier'], axis = 'columns')
train1.shape,test1.shape


# # Data Cleaning - Handle Null values

# In[1438]:


#find missing values
train1.isnull().sum()


# In[1439]:


test1.isnull().sum()


# In[1440]:


#Item weight and Outlet_size have huge missing values
#Try mean  on Item_Weight 
mean_value = train1['Item_Weight'].mean()
mean_value_test = test1['Item_Weight'].mean()
train1['Item_Weight'].fillna(mean_value, inplace = True)
test1['Item_Weight'].fillna(mean_value_test, inplace = True)


# In[1441]:


#the weight feature now null 
df = train1.copy()
df.isnull().sum()


# In[1442]:


dt = test1.copy()
dt.isnull().sum()


# In[1443]:


# Calculate mode of 'Outlet_Size'
mode_value = df['Outlet_Size'].mode()[0] 
df['Outlet_Size'].fillna(mode_value, inplace=True)
mode_value = dt['Outlet_Size'].mode()[0] 
dt['Outlet_Size'].fillna(mode_value, inplace=True)


# In[1444]:


df.isnull().sum()


# In[1445]:


dt.isnull().sum()


# In[1446]:


df.shape,dt.shape


# # Detect and Removen Outliners

# In[1447]:


train1['Item_Weight'].describe()


# In[1448]:


plt.figure(figsize=(6, 4))
plt.boxplot(df['Item_Weight'])
plt.title('Weight')
plt.show()


# In[1449]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt['Item_Weight'])
plt.title('Weight')
plt.show()


# In[1450]:


Q1 = df['Item_Weight'].quantile(0.25)
Q3 = df['Item_Weight'].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
Upper = Q3 + 1.5* IQR
df1 = df[(df.Item_Weight>=lower)&(df.Item_Weight<=Upper)]
df1.shape


# In[1451]:


Q1 = dt['Item_Weight'].quantile(0.25)
Q3 = dt['Item_Weight'].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
Upper = Q3 + 1.5* IQR
dt1 = dt[(dt.Item_Weight>=lower)&(dt.Item_Weight<=Upper)]
dt1.shape


# In[1452]:


# Now in MRP features to detect the outliners
df1['Item_MRP'].describe(), dt1['Item_MRP'].describe()


# In[1453]:


plt.figure(figsize=(6, 4))
plt.boxplot(df1['Item_MRP'])
plt.title('Boxplot of Feature')
plt.show()


# In[1454]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt1['Item_MRP'])
plt.title('Boxplot of Feature')
plt.show()


# In[1455]:


q1 = df1['Item_MRP'].quantile(0.25)
q3 = df1['Item_MRP'].quantile(0.75)
IQR = q3-q1
lower_limit = q1 - 1.5*IQR
Upper_limit = q3 + 1.5* IQR
df2 = df1[(df1.Item_Weight >lower_limit)&(df1.Item_Weight<Upper_limit)]
df2.shape


# In[1456]:


q1 = dt1['Item_MRP'].quantile(0.20)
q3 = dt1['Item_MRP'].quantile(0.80)
IQR = q3-q1
lower_limit = q1 - 1.5*IQR
Upper_limit = q3 + 1.5* IQR
dt2 = dt1[(dt1.Item_Weight >lower_limit)&(dt1.Item_Weight<Upper_limit)]
dt2.shape


# # Feature Engineering

# In[1457]:


df2['Item_Visibility'].describe()


# In[1458]:


dt2['Item_Visibility'].describe()


# In[1459]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df2['Item_Visibility'])
plt.show()


# In[1460]:


df_2.shape


# In[1461]:


plt.figure(figsize=(6, 4))
plt.boxplot(df_2['Item_Visibility'])
plt.title('Visibility')
plt.show()


# In[1462]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt_2['Item_Visibility'])
plt.title('Visibility')
plt.show()


# In[1463]:


A1 = df_2['Item_Visibility'].quantile(0.30)
A3 = df_2['Item_Visibility'].quantile(0.70)
A1,A3


# In[1464]:


IQR = A3-A1
lower_limits = A1 - 1.5*IQR
Upper_limits = A3 + 1.5* IQR


# In[1465]:


df3 = df_2[(df_2.Item_Visibility>=lower_limits)& (df_2.Item_Visibility<=Upper_limits)]
df3.shape


# In[1466]:


A1 = dt_2['Item_Visibility'].quantile(0.30)
A3 = dt_2['Item_Visibility'].quantile(0.70)
A1,A3


# In[1467]:


IQR = A3-A1
lower_limits = A1 - 1.5*IQR
Upper_limits = A3 + 1.5* IQR
dt3 = dt_2[(dt_2.Item_Visibility>lower_limits)& (dt_2.Item_Visibility<Upper_limits)]
dt3.shape


# In[1468]:


plt.figure(figsize=(6, 4))
plt.boxplot(df3['Item_Visibility'])
plt.title('Visibility')
plt.show()


# In[1469]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt3['Item_Visibility'])
plt.title('Visibility')
plt.show()


# In[1470]:


def find_unique(df):
       for column in df3:
           if df3[column].dtype=='object':
               print(f'{column} : {df3[column].unique()}')


# In[1471]:


find_unique(df3)


# In[1472]:


find_unique(dt3)


# In[1473]:


# In item_fat content - low fat and regular fat are only two section.
df3['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)
dt3['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)


# In[1474]:


find_unique(df3)


# In[1475]:


frequency = df3['Item_Fat_Content'].value_counts()
print(frequency)


# In[1476]:


cross_tab = pd.crosstab(df3['Item_Fat_Content'], df3['Outlet_Size'])
print(cross_tab)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1477]:


find_unique(dt3)


# In[1478]:


df3.head()


# In[1479]:


df3.head()


# In[1480]:


#change into cateogrical into numerical Use label encoding and dummies
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df3['Fat'] = label_encoder.fit_transform(df3['Item_Fat_Content'])
df3['Size '] = label_encoder.fit_transform(df3['Outlet_Size'])
df3['type_tier'] = label_encoder.fit_transform(df3['Outlet_Location_Type'])
df3['type_super'] = label_encoder.fit_transform(df3['Outlet_Type'])
dt3['Fat'] = label_encoder.fit_transform(dt3['Item_Fat_Content'])
dt3['Size '] = label_encoder.fit_transform(dt3['Outlet_Size'])
dt3['type_tier'] = label_encoder.fit_transform(dt3['Outlet_Location_Type'])
dt3['type_super'] = label_encoder.fit_transform(dt3['Outlet_Type'])


# In[1481]:


df3.head()


# In[1482]:


df4 = df3.drop(['Item_Fat_Content','Outlet_Size','Outlet_Type','Outlet_Location_Type'], axis='columns')
dt4 = dt3.drop(['Item_Fat_Content','Outlet_Size','Outlet_Type','Outlet_Location_Type'], axis='columns')
df4.head()


# In[ ]:





# In[1483]:


df4.head()


# In[1484]:


df4['Item_Type'].value_counts(ascending=False), dt4['Item_Type'].value_counts(ascending=False)


# In[1485]:


dummies = pd.get_dummies(df4['Item_Type'])
dummies_t = pd.get_dummies(dt4['Item_Type'])


# In[1486]:


df_4 = pd.concat([dummies,df4], axis = 'columns')


# In[1487]:


dt_4 = pd.concat([dummies_t,dt4], axis = 'columns')


# In[1488]:


df_4.head()


# In[1489]:


dt_4.head()


# In[1490]:


df5 = df_4.drop(['Item_Type'], axis = 'columns')
dt5 = dt_4.drop(['Item_Type'], axis = 'columns')
df5.head()


# In[1491]:


dt5['Item_MRP'].unique()


# In[ ]:





# In[1492]:


x = df6.drop(['Item_Outlet_Sales','Seafood'], axis = 'columns')
x.shape


# In[1493]:


x.head()


# In[1494]:


y = df5['Item_Outlet_Sales']
y.head()


# In[1495]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scale = scaler.fit_transform(x)
z_scale = scaler.fit_transform(x)


# In[1496]:


z_scale


# In[1497]:


x_train, x_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.2,random_state=15)


# In[1498]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators= 1000)
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[1499]:


x_train, x_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.2,random_state=15)


# In[1500]:


lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)


# In[1501]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.2,random_state=1000)


# In[1502]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[1503]:


y_predict = lr_clf.predict(x_test)


# In[1512]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Example usage for regression evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[1511]:


mse,mae,r2


# In[ ]:





# In[1506]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators= 1000)
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1507]:


y_predicted = lr_clf.predict(x_test)


# In[1508]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1000)

cross_val_score(LinearRegression(), x, y, cv=cv)


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





# In[1509]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        },
        'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse'],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


# In[ ]:





# In[ ]:




