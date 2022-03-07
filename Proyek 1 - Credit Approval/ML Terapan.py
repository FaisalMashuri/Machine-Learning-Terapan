#!/usr/bin/env python
# coding: utf-8

# # Import library yang dibutuhkan

# In[137]:


import numpy as np
import pandas as pd
from scipy import stats
from zipfile import ZipFile
import seaborn as sns
# !pip install imbalanced_learn
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import plot_confusion_matrix, accuracy_score, mean_squared_error, f1_score,  precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import set_config


# # Load dataset

# In[3]:


zip_dir = "archive.zip"
zip_ref = ZipFile(zip_dir, 'r')
zip_ref
zip_ref.extractall()
zip_ref.close()


# In[5]:


app_df = pd.read_csv('./application_record.csv')
credit_df = pd.read_csv('./credit_record.csv')


# In[6]:


app_df.head()


# # Exploratory Data Analysis

# In[7]:


# Memuat informasi dari dataset
app_df.info()


# In[8]:


# Melihat Statistik dataset
app_df.describe()


# In[9]:


# melihat jumlah nilai null pada dataset
app_df.isnull().sum()


# In[10]:


# Mengecek colom apa saja yang tidak mengandung nilai numerik
cat_columns = app_df.columns[(app_df.dtypes =='object').values].tolist()
cat_columns


# In[11]:


# Mengecek kolom apa saya yang mengandung nilai numerik
app_df.columns[(app_df.dtypes !='object').values].tolist()


# In[12]:


# mengecek nilai unik pada kolom non numerik

for i in app_df.columns[(app_df.dtypes =='object').values].tolist():
    print(i,'\n')
    print(app_df[i].value_counts())
    print('-----------------------------------------------')


# # Data cleaning

# In[18]:


app_df.drop('OCCUPATION_TYPE', axis=1, inplace=True)
app_df.drop('FLAG_MOBIL', axis=1, inplace=True)
app_df.drop('FLAG_WORK_PHONE', axis=1, inplace=True)
app_df.drop('FLAG_PHONE', axis=1, inplace=True)
app_df.drop('FLAG_EMAIL', axis=1, inplace=True)


# ## handling outliers

# In[19]:


# Mengecek kolom yang berisi nilai numerik
app_df.columns[(app_df.dtypes !='object').values].tolist()


# In[22]:


num_cols = ['CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AGE_YEARS',
 'DAYS_EMPLOYED',
 'CNT_FAM_MEMBERS']

plt.figure(figsize=(19,9))
app_df[num_cols].boxplot()
plt.title("Numerical variables in the data", fontsize=20)
plt.show()


# In[23]:


# Fungsi Untuk Mendeteksi Outliers
def detect_outlier(data_1):
    outliers=[]
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# In[26]:


AMT_outliers = detect_outlier(app_df['AMT_INCOME_TOTAL'])
CNTCh_outliers = detect_outlier(app_df['CNT_CHILDREN'])
DE_outliers = detect_outlier(app_df['DAYS_EMPLOYED'])
CFM_outliers = detect_outlier(app_df['CNT_FAM_MEMBERS'])
print("outliers")
print(f"AMT_INCOME_TOTAL : {len(AMT_outliers)}")
print(f"CNT_CHILDREN: {len(CNTCh_outliers)}")
print(f"YEARS_EMPLOYED : {len(DE_outliers)}")
print(f"CNT_FAM_MEMBERS : {len(CFM_outliers)}")


# In[27]:


# Fungsi untuk menghapus outliers
def remove_outlier(data):
    z = np.abs(stats.zscore(data))
    threshold = 3
    Q1 = np.percentile(data, 25,
                   interpolation = 'midpoint')
    Q3 = np.percentile(data, 75,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = data >= (Q3+1.5*IQR)
     # Below Lower bound
    lower = data <= (Q1-1.5*IQR)
    return data.index[upper]


# In[28]:


application_df = app_df.copy()
application_df.head()


# In[30]:


application_df.drop(remove_outlier(application_df["CNT_CHILDREN"]), inplace=True)
application_df.drop(remove_outlier(application_df["AMT_INCOME_TOTAL"]), inplace=True)
application_df.drop(remove_outlier(application_df["DAYS_EMPLOYED"]), inplace=True)
application_df.drop(remove_outlier(application_df["CNT_FAM_MEMBERS"]), inplace=True)


# In[32]:


num_cols = ['CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AGE_YEARS',
 'DAYS_EMPLOYED',
 'CNT_FAM_MEMBERS']

plt.figure(figsize=(19,9))
application_df[num_cols].boxplot()
plt.title("Numerical variables in the data", fontsize=20)
plt.show()


# In[33]:


application_df.head()


# In[34]:


AMT_outliers = detect_outlier(app_df['AMT_INCOME_TOTAL'])
CNTCh_outliers = detect_outlier(app_df['CNT_CHILDREN'])
DE_outliers = detect_outlier(app_df['DAYS_EMPLOYED'])
CFM_outliers = detect_outlier(app_df['CNT_FAM_MEMBERS'])
print("outliers")
print(f"AMT_INCOME_TOTAL : {len(AMT_outliers)}")
print(f"CNT_CHILDREN: {len(CNTCh_outliers)}")
print(f"YEARS_EMPLOYED : {len(DE_outliers)}")
print(f"CNT_FAM_MEMBERS : {len(CFM_outliers)}")


# In[36]:


# Mengkategorikan kolom 'STATUS' ke klasifikasi biner 0 : Klien Baik dan 1 : klien buruk
credit_df['STATUS'].replace(['C', 'X'],0, inplace=True)
credit_df['STATUS'].replace(['2','3','4','5'],1, inplace=True)
credit_df['STATUS'] = credit_df['STATUS'].astype('int')


# In[37]:


credit_df.info()


# In[38]:


credit_df_trans = credit_df.groupby('ID').agg(max).reset_index()


# In[39]:


credit_df_trans.drop('MONTHS_BALANCE', axis=1, inplace=True)
credit_df_trans.head()


# In[40]:


# menggabungkan dua set data berdasarkan 'ID'
final_df = pd.merge(application_df, credit_df_trans, on='ID', how='inner')
final_df.head()


# # Data visualisasi

# In[41]:


# Grafik ini menunjukkan bahwa, tidak ada kolom (Fitur) yang sangat berkorelasi dengan 'Status'
plt.figure(figsize = (8,8))
sns.heatmap(final_df.corr(), annot=True)
plt.show()


# In[42]:


# Grafik ini menunjukkan bahwa, sebagian besar aplikasi diajukan oleh Female's
plt.pie(final_df['CODE_GENDER'].value_counts(), labels=['Female', 'Male'], autopct='%1.2f%%')
plt.title('% of Applications submitted based on Gender')
plt.show()


# In[43]:


# Grafik ini menunjukkan bahwa, sebagian besar aplikasi disetujui untuk Wanita
plt.pie(final_df[final_df['STATUS']==0]['CODE_GENDER'].value_counts(), labels=['Female', 'Male'], autopct='%1.2f%%')
plt.title('% of Applications Approved based on Gender')
plt.show()


# In[44]:


# Grafik ini menunjukkan bahwa, mayoritas pemohon tidak memiliki mobil
plt.pie(final_df['FLAG_OWN_CAR'].value_counts(), labels=['No', 'Yes'], autopct='%1.2f%%')
plt.title('% of Applications submitted based on owning a Car')
plt.show()


# In[45]:


# Grafik ini menunjukkan bahwa, sebagian besar pemohon memiliki properti / Rumah Real Estate
plt.pie(final_df['FLAG_OWN_REALTY'].value_counts(), labels=['Yes','No'], autopct='%1.2f%%')
plt.title('% of Applications submitted based on owning a Real estate property')
plt.show()


# In[46]:


# Grafik ini menunjukkan bahwa, sebagian besar pelamar tidak memiliki anak
plt.figure(figsize = (8,8))
plt.pie(final_df['CNT_CHILDREN'].value_counts(), labels=final_df['CNT_CHILDREN'].value_counts().index, autopct='%1.2f%%')
plt.title('% of Applications submitted based on Children count')
plt.legend()
plt.show()


# In[47]:


# Grafik ini menunjukkan bahwa, sebagian besar pendapatan pemohon berkisar antara 100k hingga 300k
plt.hist(final_df['AMT_INCOME_TOTAL'], bins=20)
plt.xlabel('Total Annual Income')
plt.title('Histogram')
plt.show()


# In[48]:


# Grafik ini menunjukkan bahwa, sebagian besar pelamar bekerja secara profesional
plt.figure(figsize = (8,8))
plt.pie(final_df['NAME_INCOME_TYPE'].value_counts(), labels=final_df['NAME_INCOME_TYPE'].value_counts().index, autopct='%1.2f%%')
plt.title('% of Applications submitted based on Income Type')
plt.legend()
plt.show()


# In[49]:


# Grafik ini menunjukkan bahwa sebagian besar pelamar sudah menikah
plt.figure(figsize=(8,8))
sns.barplot(final_df['NAME_FAMILY_STATUS'].value_counts().index, final_df['NAME_FAMILY_STATUS'].value_counts().values)
plt.title('% of Applications submitted based on Family Status')
plt.show()


# In[50]:


# Grafik ini menunjukkan bahwa, sebagian besar pemohon tinggal di Rumah/Apartemen
plt.figure(figsize=(12,5))
sns.barplot(final_df['NAME_HOUSING_TYPE'].value_counts().index, final_df['NAME_HOUSING_TYPE'].value_counts().values)
plt.title('% of Applications submitted based on Housing Type')
plt.show()


# In[51]:


# Grafik ini menunjukkan bahwa, mayoritas pelamar berusia 25 hingga 65 tahun
plt.hist(final_df['AGE_YEARS'], bins=20)
plt.xlabel('Age')
plt.title('Histogram')
plt.show()


# # Data transforming

# In[53]:


final_df.head()


# In[54]:


features = final_df.drop(['STATUS'], axis=1)
label = final_df['STATUS']


# In[56]:


numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")), 
    ("scaler", MinMaxScaler()), 
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", OneHotEncoder()), 
])


# In[58]:


X_train,X_test,Y_train,Y_test = train_test_split(features, label, test_size = 0.2, random_state = 42)


# In[60]:


X_train.info()


# In[69]:


preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline,["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AGE_YEARS", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS"]),
    ("categoric", categorical_pipeline, ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE" ])
])


# In[70]:


pipeline_svm = Pipeline([
    ("prep", preprocessor),
    ("algo", svm.SVC(kernel='linear'))
])


# In[110]:


set_config(display="diagram")
pipeline_svm


# In[75]:


pipeline_xgb = Pipeline([
    ("prep", preprocessor),
    ("algo", XGBClassifier())
])


# In[111]:


set_config(display="diagram")
pipeline_xgb


# In[71]:


pipeline_svm.fit(X_train, Y_train)


# In[91]:


pipeline_svm.score(X_test,Y_test)


# In[76]:


pipeline_xgb.fit(X_train, Y_train)


# In[92]:


pipeline_xgb.score(X_test,Y_test)


# In[117]:


y_pred_svm = pipeline_svm.predict(X_test)
y_pred_xgb = pipeline_xgb.predict(X_test)


# In[127]:


eval_model = pd.DataFrame(columns=['train_mse', 'test_mse', 'accuracy'], index=['SVM','XGBoost'])


# In[128]:


eval_model.loc["SVM", 'train_mse'] = mean_squared_error(y_true=Y_train, y_pred=pipeline_svm.predict(X_train))/1e3
eval_model.loc["SVM", 'test_mse'] = mean_squared_error(y_true=Y_test, y_pred=pipeline_svm.predict(X_test))/1e3
eval_model.loc["XGBoost", 'train_mse'] = mean_squared_error(y_true=Y_train, y_pred=pipeline_xgb.predict(X_train))/1e3 
eval_model.loc["XGBoost", 'test_mse'] = mean_squared_error(y_true=Y_test, y_pred=pipeline_xgb.predict(X_test))/1e3


# In[129]:


eval_model.loc["SVM", 'accuracy'] = accuracy_score(y_pred_svm, Y_test)
eval_model.loc["XGBoost", 'accuracy'] = accuracy_score(y_pred_xgb, Y_test)


# In[132]:


eval_model


# In[139]:


plot_confusion_matrix(pipeline_svm, X_test, Y_test)


# In[140]:


plot_confusion_matrix(pipeline_xgb, X_test, Y_test)


# In[ ]:




