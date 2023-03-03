#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns

#relevant ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#ML models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#default theme
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

#warning hadle
warnings.filterwarnings("ignore")

print("Libraries imported")


# In[4]:


df = pd.read_csv("C:/Users/ASUS/Downloads/train_u6lujuX_CVtuZ9i.csv")


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df['Credit_History'] = df['Credit_History'].astype('O')
df.describe(include='O')


# In[23]:


df.duplicated().any()


# In[24]:


# Let's look at the target percentage

plt.figure(figsize=(8,6))
sns.countplot(df['Loan_Status']);

print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))

# We can consider it as imbalanced data, but for now i will not


# In[25]:


df.columns


# In[27]:


df.head(1)


# In[28]:


grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History');


# In[29]:


print(pd.crosstab(df['Credit_History'],df['Loan_Status']))

df_history_Y = df[df['Credit_History'] == 1]
df_history_N = df[df['Credit_History'] == 0]

perc_df_self_Y = df_history_Y['Loan_Status'].value_counts()['Y']/len(df_history_Y)
perc_df_self_N = df_history_N['Loan_Status'].value_counts()['Y']/len(df_history_N)

print('\n')

print('Percentage loans with Credit_History Y: %.3f' %perc_df_self_Y)
print('Percentage loans with Credit_History N: %.3f' %perc_df_self_N)


# In[30]:


# Gender

grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender');


# In[31]:


# Married
plt.figure(figsize=(15,5))
sns.countplot(x='Married', hue='Loan_Status', data=df);


# In[32]:


# Dependents

plt.figure(figsize=(15,5))
sns.countplot(x='Dependents', hue='Loan_Status', data=df);


# In[33]:


# Education

grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Education');


# In[34]:


df_graduated = df[df['Education'] == 'Graduate']
df_not_graduated = df[df['Education'] != 'Graduate']

print('Graduate')
df_graduated.head()


# In[35]:


print('Not graduate')
df_not_graduated.head()


# In[36]:


n_loans_gr = len(df_graduated[df_graduated['Loan_Status'] == 'Y'])
#n_loans_gr = df_graduated['Loan_Status'].value_counts()[0]
length_gr =len(df_graduated)
perc_df_graduated_Y = n_loans_gr/length_gr

n_loans_not_gr = len(df_not_graduated[df_not_graduated['Loan_Status'] == 'Y'])
#n_loans_gr = df_graduated['Loan_Status'].value_counts()[0]
length_not_gr =len(df_not_graduated)
perc_df_not_graduated_Y = n_loans_not_gr/length_not_gr

print('Percentage loans for NOT graduated: %.2f' % perc_df_not_graduated_Y)

print('Percentage loans for graduated: %.2f' % perc_df_graduated_Y)


# In[37]:


# Self_Employed

grid = sns.FacetGrid(df,col='Self_Employed', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Loan_Status');

# No pattern (same as Education)


# In[38]:


df_self_employed_Y = df[df['Self_Employed'] == 'Yes']
df_self_employed_N = df[df['Self_Employed'] == 'No']

n_loans_self_y = df_self_employed_Y['Loan_Status'].value_counts()[0]
length_self_y =len(df_self_employed_Y)
perc_df_self_Y = n_loans_self_y/length_self_y

n_loans_self_n = df_self_employed_N['Loan_Status'].value_counts()[0]
length_self_n =len(df_self_employed_N)
perc_df_self_N = n_loans_self_n/length_self_n

print('Percentage loans for self employed: %.2f' % perc_df_self_Y)

print('Percentage loans for not self employed: %.2f' % perc_df_self_N)


# In[39]:


# Property_Area

plt.figure(figsize=(15,5))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df);


# In[40]:


# ApplicantIncome

plt.scatter(df['ApplicantIncome'], df['Loan_Status']);

# No pattern


# In[41]:


# The numerical data

df.groupby('Loan_Status').median()


# In[42]:


df.isnull().sum().sort_values(ascending=False)


# In[43]:


# We will separate the numerical columns from the categorical

cat_data = []
num_data = []

for i,c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else :
        num_data.append(df.iloc[:, i])
        

cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()


# In[44]:


cat_data.head()


# In[45]:


num_data.head()


# In[49]:


cat_data.isnull().sum().any() #cat_data missing values?


# In[50]:


cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
cat_data.isnull().sum().any() # No more missing data


# In[51]:


num_data.isnull().sum().any() # num_data missing data?


# In[52]:


# Numerical data
# Fill every missing value with their previous value in the same column.

num_data.fillna(method='bfill', inplace=True)
num_data.isnull().sum().any() # no more missing data


# In[53]:


from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()
cat_data.head()


# In[54]:


# Transform the target column

target_values = {'Y': 0 , 'N' : 1}

# Save 'Loan_Status' column in 'target'
target = cat_data['Loan_Status']

#Remove 'Loan_Status' column from cat_data
cat_data.drop('Loan_Status', axis=1, inplace=True)

# Map 'target' according to 'target_values' 
target = target.map(target_values)


# In[55]:


# Transform the remaining columns of cat_data

for i in cat_data:
    cat_data[i] = le.fit_transform(cat_data[i])
target.head()


# In[56]:


cat_data.head()


# In[57]:


# Create new Pandas object 
df = pd.concat([cat_data, num_data, target], axis=1)
df.head()


# In[58]:


# Create (X, y) Pandas objects for data training 
X = pd.concat([cat_data, num_data], axis=1)
y = target


# In[59]:


from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    
print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

# Almost same ratio
print('\nratio of target in y_train :',y_train.value_counts().values/ len(y_train))
print('ratio of target in y_test :',y_test.value_counts().values/ len(y_test))
print('ratio of target in original_data :',df['Loan_Status'].value_counts().values/ len(df))


# In[60]:


# We will use 4 different models for training

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}


# In[61]:


# Loss

from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score

def loss(y_true, y_pred, retu=False):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
    else:
        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' 
              % (pre, rec, f1, loss, acc))


# In[62]:


# Train data

def train_eval_train(models, X, y):
    for name, model in models.items():
        print(name,':')
        model.fit(X, y)
        loss(y, model.predict(X))
        print('-'*30)

train_eval_train(models, X_train, y_train)

# We can see that best model is LogisticRegression at least for now, SVC is just 
# memorizing the data so it is overfitting.


# In[63]:


# Cross validation

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

def train_eval_cross(models, X, y, folds):
    # We will change X & y to dataframe because we will use iloc (iloc don't work on numpy array)
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    idx = [' pre', ' rec', ' f1', ' loss', ' acc']
    for name, model in models.items():
        ls = []
        print(name,':')

        for train, test in folds.split(X, y):
            model.fit(X.iloc[train], y.iloc[train]) 
            y_pred = model.predict(X.iloc[test]) 
            ls.append(loss(y.iloc[test], y_pred, retu=True))
        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0])  #[0] because we don't want to show the name of the column
        print('-'*30)
        
train_eval_cross(models, X_train, y_train, skf)

# SVC is just memorizing the data, and you can see 
# that here DecisionTreeClassifier is better than LogisticRegression


# In[64]:


# Some explanation about Logistic Regression

x = []
idx = [' pre', ' rec', ' f1', ' loss', ' acc']

# We will use one model
log = LogisticRegression()

for train, test in skf.split(X_train, y_train):
    log.fit(X_train.iloc[train], y_train.iloc[train])
    ls = loss(y_train.iloc[test], log.predict(X_train.iloc[test]), retu=True)
    x.append(ls)
    
# Thats what we get
pd.DataFrame(x, columns=idx)

# (column 0 represent the precision_score of the 10 folds)
# (row 0 represent the (pre, rec, f1, loss, acc) for the first fold)
# then we should find the mean of every column
# pd.DataFrame(x, columns=idx).mean(axis=0)


# In[65]:


# Credit_History is the best.

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);

# Here we got 58% similarity between LoanAmount & ApplicantIncome 
# and that may be bad for our model so we will see what we can do


# In[66]:


# I will try to make some operations on some features, here I just tried diffrent operations on different features,
# having experience in the field, and having knowledge about the data will also help

X_train['new_col'] = X_train['CoapplicantIncome'] / X_train['ApplicantIncome']  
X_train['new_col_2'] = X_train['LoanAmount'] * X_train['Loan_Amount_Term']


# In[67]:


data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);

# new_col 0.03 , new_col_2, 0.047
# Not that much , but that will help us reduce the number of features


# In[68]:


X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)


# In[69]:


train_eval_cross(models, X_train, y_train, skf)


# In[70]:


#print(X_train.shape)
#print(X_train.shape[0]) # Rows
#print(X_train.shape[1]) # Columns

print('************************\n')

for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')


# In[71]:


# new_col_2

# We can see we got right_skewed
# We can solve this problem with very simple statistical technique , by taking the logarithm of all the values
# because when data is normally distributed that will help improving our model

from scipy.stats import norm

fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(X_train['new_col_2'], ax=ax[0], fit=norm)
ax[0].set_title('new_col_2 before log')

X_train['new_col_2'] = np.log(X_train['new_col_2'])  # logarithm of all the values

sns.distplot(X_train['new_col_2'], ax=ax[1], fit=norm)
ax[1].set_title('new_col_2 after log');


# In[72]:


# Now we will evaluate our models, and i will do that continuously ,so i don't need to 
# mention that every time

train_eval_cross(models, X_train, y_train, skf)


# In[73]:


# new_col

# Most of our data is 0 , so we will try to change other values to 1

print('before:')
print(X_train['new_col'].value_counts())

X_train['new_col'] = [x if x==0 else 1 for x in X_train['new_col']]
print('-'*50)
print('\nafter:')
print(X_train['new_col'].value_counts())


# In[74]:


train_eval_cross(models, X_train, y_train, skf)

# We are improving our models as we go


# In[75]:


for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')
    


# In[76]:


# Outliers: we will use boxplot to detect them 

sns.boxplot(X_train['new_col_2']);
plt.title('new_col_2 outliers', fontsize=15);
plt.xlabel('');


# In[77]:


threshold = 0.1  # This number is a hyper parameter, by reducing it, more points are removed.
                 # You can just try different values, the deafult value is (1.5) it works good for most cases.
                 # Be careful, you don't want to try a small number because you may loss some important information from data.
                 # That's why I was surprised when 0.1 gived me the best result
            
new_col_2_out = X_train['new_col_2']

q25, q75 = np.percentile(new_col_2_out, 25), np.percentile(new_col_2_out, 75) # Q25, Q75

print('Quartile 25: {} , Quartile 75: {}'.format(q25, q75))

iqr = q75 - q25
print('iqr: {}'.format(iqr))

cut = iqr * threshold
lower, upper = q25 - cut, q75 + cut
print('Cut Off: {}'.format(cut))
print('Lower: {}'.format(lower))
print('Upper: {}'.format(upper))

outliers = [x for x in new_col_2_out if x < lower or x > upper]
print('Nubers of Outliers: {}'.format(len(outliers)))
print('outliers:{}'.format(outliers))

data_outliers = pd.concat([X_train, y_train], axis=1)
print('\nlen X_train before dropping the outliers', len(data_outliers))
data_outliers = data_outliers.drop(data_outliers[(data_outliers['new_col_2'] > upper) | (data_outliers['new_col_2'] < lower)].index)

print('len X_train before dropping the outliers', len(data_outliers))


# In[78]:


X_train = data_outliers.drop('Loan_Status', axis=1)
y_train = data_outliers['Loan_Status']


# In[79]:


sns.boxplot(X_train['new_col_2']);
plt.title('new_col_2 without outliers', fontsize=15);
plt.xlabel('');

# good :)


# In[80]:


train_eval_cross(models, X_train, y_train, skf)

# Now we get 94.1 ??? for precision & 53.5 for recall


# In[81]:


# Self_Employed got really bad corr (-0.00061) , let's try to remove it and see what will happen

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);


# In[82]:


#X_train.drop(['Self_Employed'], axis=1, inplace=True)

train_eval_cross(models, X_train, y_train, skf)

# looks like Self_Employed is not important
# KNeighborsClassifier improved

# droping all the features Except for Credit_History actually improved KNeighborsClassifier and didn't change anything in other models
# so you can try it by you self
# but don't forget to do that on testing data too

#X_train.drop(['Self_Employed','Dependents', 'new_col_2', 'Education', 'Gender', 'Property_Area','Married', 'new_col'], axis=1, inplace=True)


# In[83]:


data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);


# In[84]:


X_test.head()


# In[85]:


X_test_new = X_test.copy()


# In[86]:


x = []

X_test_new['new_col'] = X_test_new['CoapplicantIncome'] / X_test_new['ApplicantIncome']  
X_test_new['new_col_2'] = X_test_new['LoanAmount'] * X_test_new['Loan_Amount_Term']
X_test_new.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)

X_test_new['new_col_2'] = np.log(X_test_new['new_col_2'])

X_test_new['new_col'] = [x if x==0 else 1 for x in X_test_new['new_col']]


# In[87]:


X_test_new.head()


# In[88]:


X_train.head()


# In[89]:


for name,model in models.items():
    print(name, end=':\n')
    loss(y_test, model.predict(X_test_new))
    print('-'*40)


# In[ ]:




