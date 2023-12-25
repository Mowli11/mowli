#!/usr/bin/env python
# coding: utf-8

# In[3]:


#just load and import our need library
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# # DATA LOAD  - BIGMART INTO DATAFRAME

# In[4]:


# Load / Read the data
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
train.head(10)


# In[5]:


#understaing the our dataset, remove unwanted columns 
train1 = train.drop(['Item_Identifier','Outlet_Identifier'], axis = 'columns')
test1 = test.drop(['Item_Identifier','Outlet_Identifier'], axis = 'columns')
train1.shape,test1.shape


# # Data Cleaning - Handle Null values

# In[6]:


#find missing values
train1.isnull().sum()


# In[7]:


test1.isnull().sum()


# In[8]:


#Item weight and Outlet_size have huge missing values
#Try mean  on Item_Weight 
mean_value = train1['Item_Weight'].mean()
mean_value_test = test1['Item_Weight'].mean()
train1['Item_Weight'].fillna(mean_value, inplace = True)
test1['Item_Weight'].fillna(mean_value_test, inplace = True)


# In[9]:


#the weight feature now null 
df = train1.copy()
df.isnull().sum()


# In[10]:


dt = test1.copy()
dt.isnull().sum()


# In[11]:


# Calculate mode of 'Outlet_Size'
mode_value = df['Outlet_Size'].mode()[0] 
df['Outlet_Size'].fillna(mode_value, inplace=True)
mode_value = dt['Outlet_Size'].mode()[0] 
dt['Outlet_Size'].fillna(mode_value, inplace=True)


# In[12]:


df.isnull().sum()


# In[13]:


dt.isnull().sum()


#  # Detect and Removen Outliners

# In[14]:


df['Item_Weight'].describe()


# In[15]:


plt.figure(figsize=(6, 4))
plt.boxplot(df['Item_Weight'])
plt.title('Weight')
plt.show()


# In[16]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt['Item_Weight'])
plt.title('Weight')
plt.show()


# In[17]:


Q1 = df['Item_Weight'].quantile(0.20)
Q3 = df['Item_Weight'].quantile(0.80)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
Upper = Q3 + 1.5* IQR
df1 = df[(df.Item_Weight>lower)&(df.Item_Weight<Upper)]
df1.shape


# In[18]:


Q1 = dt['Item_Weight'].quantile(0.20)
Q3 = dt['Item_Weight'].quantile(0.80)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
Upper = Q3 + 1.5* IQR
dt1 = dt[(dt.Item_Weight>lower)&(dt.Item_Weight<Upper)]
dt1.shape


# In[19]:


# Now in MRP features to detect the outliners
df1['Item_MRP'].describe(), dt1['Item_MRP'].describe()


# In[20]:


plt.figure(figsize=(6, 4))
plt.boxplot(df1['Item_MRP'])
plt.title('Boxplot of Feature')
plt.show()


# In[21]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt1['Item_MRP'])
plt.title('Boxplot of Feature')
plt.show()


# In[22]:


q1 = df1['Item_MRP'].quantile(0.20)
q3 = df1['Item_MRP'].quantile(0.80)
IQR = q3-q1
lower_limit = q1 - 1.5*IQR
Upper_limit = q3 + 1.5* IQR
df2 = df1[(df1.Item_Weight >lower_limit)&(df1.Item_Weight<Upper_limit)]
df2.shape


# In[23]:


q1 = dt1['Item_MRP'].quantile(0.20)
q3 = dt1['Item_MRP'].quantile(0.80)
IQR = q3-q1
lower_limit = q1 - 1.5*IQR
Upper_limit = q3 + 1.5* IQR
dt2 = dt1[(dt1.Item_Weight >lower_limit)&(dt1.Item_Weight<Upper_limit)]
dt2.shape


# # Feature Engineering

# In[24]:


df2['Item_Visibility'].describe()


# In[25]:


dt2['Item_Visibility'].describe()


# In[26]:


df_2 = df2[df2['Item_Visibility'] != 0]
df_2.shape


# In[27]:


dt_2 = dt[dt2['Item_Visibility'] != 0]
dt_2.shape


# In[28]:


#Make sure the visibility will be an better numerical value
df_2['Visibility']=df_2['Item_Visibility'] * 100
df_2['Visibility'].head()


# In[29]:


dt_2['Visibility']=dt_2['Item_Visibility'] * 100
dt_2['Visibility'].head()


# In[30]:


df_2.shape


# In[31]:


plt.figure(figsize=(6, 4))
plt.boxplot(df_2['Visibility'])
plt.title('Visibility')
plt.show()


# In[32]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt_2['Visibility'])
plt.title('Visibility')
plt.show()


# In[33]:


A1 = df_2['Visibility'].quantile(0.30)
A3 = df_2['Visibility'].quantile(0.70)
A1,A3


# In[34]:


IQR = A3-A1
lower_limits = A1 - 1.5*IQR
Upper_limits = A3 + 1.5* IQR


# In[35]:


df3 = df_2[(df_2.Visibility>lower_limits)& (df_2.Visibility<Upper_limits)]
df3.shape


# In[36]:


A1 = dt_2['Visibility'].quantile(0.30)
A3 = dt_2['Visibility'].quantile(0.70)
A1,A3


# In[37]:


IQR = A3-A1
lower_limits = A1 - 1.5*IQR
Upper_limits = A3 + 1.5* IQR
dt3 = dt_2[(dt_2.Visibility>lower_limits)& (dt_2.Visibility<Upper_limits)]
dt3.shape


# In[38]:


plt.figure(figsize=(6, 4))
plt.boxplot(df3['Visibility'])
plt.title('Visibility')
plt.show()


# In[39]:


plt.figure(figsize=(6, 4))
plt.boxplot(dt3['Visibility'])
plt.title('Visibility')
plt.show()


# In[40]:


def find_unique(df):
        for column in df3:
            if df3[column].dtype=='object':
                print(f'{column} : {df3[column].unique()}')


# In[41]:


find_unique(df3)


# In[42]:


find_unique(dt3)


# In[43]:


# In item_fat content - low fat and regular fat are only two section.
df3['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)
dt3['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)


# In[44]:


find_unique(df3)


# In[45]:


find_unique(dt3)


# In[46]:


df3.head()


# In[47]:


df3.head()


# In[48]:


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


# In[49]:


df3.head()


# In[50]:


df4 = df3.drop(['Item_Fat_Content','Item_Visibility','Outlet_Size','Outlet_Type','Outlet_Location_Type'], axis='columns')
dt4 = dt3.drop(['Item_Fat_Content','Item_Visibility','Outlet_Size','Outlet_Type','Outlet_Location_Type'], axis='columns')
df4.head()


# In[51]:


df4['Item_Type'].value_counts(ascending=False), dt4['Item_Type'].value_counts(ascending=False)


# In[52]:


dummies = pd.get_dummies(df4['Item_Type'])
dummies_t = pd.get_dummies(dt4['Item_Type'])


# In[53]:


df_4 = pd.concat([dummies,df4], axis = 'columns')


# In[54]:


dt_4 = pd.concat([dummies_t,dt4], axis = 'columns')


# In[55]:


df_4.head()


# In[56]:


df5 = df_4.drop(['Item_Type'], axis = 'columns')
dt5 = dt_4.drop(['Item_Type'], axis = 'columns')
df5.head()


# In[57]:


def custom_round(value):
    return int(value + 0.5) if value - int(value) >= 0.5 else int(value)

# Apply the custom rounding function to the Series
rounded_series = df5['Item_MRP'].apply(lambda x: custom_round(x))
rounded_value = df5['Visibility'].apply(lambda x : custom_round(x))
rounded_ser = dt5['Item_MRP'].apply(lambda x: custom_round(x))
rounded_va = dt5['Visibility'].apply(lambda x : custom_round(x))

print(rounded_series, rounded_value)


# In[58]:


df5.head()


# In[59]:


d = df5.drop(['Item_MRP'], axis = 'columns')
d.head()


# In[60]:


t = dt5.drop(['Item_MRP'], axis = 'columns')
t.head()


# In[61]:


df6 = pd.concat([d,rounded_series], axis = 'columns')
dt6= pd.concat([t,rounded_ser], axis = 'columns')


# In[62]:


df6.head()


# In[63]:


dt6.head()


# # Model to train

# In[64]:


x = df6.drop(['Item_Outlet_Sales','Seafood'], axis = 'columns')
x.shape


# In[65]:


y = df5['Item_Outlet_Sales']
y.head()


# In[66]:


z = dt6.drop(['Seafood'], axis = 'columns')
z.head()


# # Make an Standardize 

# In[67]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scale = scaler.fit_transform(x)


# In[68]:


z_scale = scaler.fit_transform(z)


# In[ ]:





# In[69]:


x_scale


# In[70]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.2,random_state=1000)


# In[71]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators= 1000)
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[86]:


pre = model.predict(z_scale)
pre


# In[72]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)


# # Get an Best model using Grid  

# In[74]:


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


# # My best model with score is random_forest with 0.578

# # Evaluate the model

# In[206]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[207]:


mse,rmse,mae,r2


# # Visualization of our data

# In[102]:


df3['Item_Type'] = df3['Item_Type'].astype('category')
df3['Item_Fat_Content'] = df3['Item_Fat_Content'].astype('category')


# In[104]:


plt.figure(figsize=(14, 8))
sns.countplot(x='Item_Type', hue='Item_Fat_Content', data=df3)
plt.xticks(rotation=45, ha='right')
plt.title('Item Fat Content Distribution by Item Type')
plt.show()


# In[114]:


df3['Item_Type'] = df3['Item_Type'].astype('category')
df3['Outlet_Size'] = df3['Outlet_Size'].astype('category')
df3['Otlet_Location_Type'] = df3['Outlet_Location_Type'].astype('category')
df3['Outlet_Type'] = df3['Outlet_Type'].astype('category')


# In[117]:


df3['Item_Outlet_Sales'] = df3['Item_Outlet_Sales'].astype(float)


# In[176]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df3.Item_Outlet_Sales, rwidth = 0.8)
plt.xlabel("Price distribution")
plt.ylabel("Count") #Given plot says our sales high in below 4k


# In[177]:


plt.figure(figsize=(20, 10))
for outlet_type, data in df3.groupby('Outlet_Type'):
    plt.hist(data['Item_Outlet_Sales'], label=outlet_type, alpha=0.6, bins=30)
    
plt.xlabel("Price distribution")
plt.ylabel("Count")
plt.title("Distribution of Item Outlet Sales by Outlet Type")
plt.legend()
plt.show() #Our supermarket type1 as huge impact on our sales 


# In[180]:


fig, axs = plt.subplots(figsize=(20, 10), ncols=len(df3['Outlet_Location_Type'].unique()), sharey=True)


for idx, (outlet_type, data) in enumerate(df3.groupby('Outlet_Location_Type')):
    ax = axs[idx] if len(df3['Outlet_Location_Type'].unique()) > 1 else axs  
    ax.hist(data['Item_Outlet_Sales'], alpha=0.6, bins=30)
    ax.set_title(outlet_type)
    ax.set_xlabel("Price distribution")
    ax.set_ylabel("Count")
    
plt.suptitle("Distribution of Item Outlet Sales by Outlet Location Type")
plt.tight_layout()
plt.show()


# In[181]:


sales_by_item_type = df3.groupby('Item_Type')['Item_Outlet_Sales'].sum()


plt.figure(figsize=(10, 8))
plt.pie(sales_by_item_type, labels=sales_by_item_type.index, autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Item Type')
plt.axis('equal')  
plt.show() #snacks and Fruits are high share for our sales


# In[190]:


fig, axs = plt.subplots(figsize=(20, 10), ncols=len(df3['Outlet_Size'].unique()), sharey=True)


for idx, (outlet_type, data) in enumerate(df3.groupby('Outlet_Size')):
    ax = axs[idx] if len(df3['Outlet_Location_Type'].unique()) > 1 else axs  
    ax.hist(data['Item_Outlet_Sales'], alpha=0.6, bins=30)
    ax.set_title(outlet_type)
    ax.set_xlabel("Price distribution")
    ax.set_ylabel("Count")
    
plt.suptitle("Distribution of Item Outlet Sales by Outlet Location Type")
plt.tight_layout()
plt.show() #our medium size products are so high margin


# In[191]:


fig, axs = plt.subplots(figsize=(20, 10), ncols=len(df3['Item_Fat_Content'].unique()), sharey=True)

for idx, (outlet_type, data) in enumerate(df3.groupby('Item_Fat_Content')):
    ax = axs[idx] if len(df3['Outlet_Location_Type'].unique()) > 1 else axs  
    ax.hist(data['Item_Outlet_Sales'], alpha=0.6, bins=30)
    ax.set_title(outlet_type)
    ax.set_xlabel("Price distribution")
    ax.set_ylabel("Count")
    
plt.suptitle("Distribution of Item Outlet Sales by Outlet Location Type")
plt.tight_layout()
plt.show() 


# In[203]:


df3['Item_Type'] = df3['Item_Type'].astype('category')
df3['Item_Fat_Content'] = df3['Item_Fat_Content'].astype('category')

plt.figure(figsize=(14, 8))
ax = sns.countplot(x='Item_Type', hue='Item_Fat_Content', data=df3)
plt.xticks(rotation=45, ha='right')
plt.title('Item Fat Content Distribution by Item Type')

# Add count annotations on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()


# In[200]:


plt.figure(figsize=(14, 8))
sns.countplot(x='Item_Type', hue='Outlet_Size', data=df3)
plt.xticks(rotation=45, ha='right')
plt.title('Types of size ')
plt.show()


# In[208]:


import pickle
file_pathes = 'Saved_data.pikl'
with open(file_pathes, 'wb') as file:
    pickle.dump(df4, file)
    print("Data saved")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




