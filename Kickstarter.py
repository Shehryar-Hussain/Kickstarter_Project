# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 03:11:43 2021

@author: shehr
"""

import pandas as pd
import numpy

# Import the TRAINING dataset here
kick_data = pd.read_excel('C:/Users/shehr/Desktop/INSY 662/Project/Kickstarter.xlsx')

# Import TEST dataset here
kick_data_test = pd.read_excel('C:/Users/shehr/Desktop/INSY 662/Project/Kickstarter-Grading-Sample.xlsx')

kick_data.isnull().sum()
kick_data1 = kick_data.drop(['launch_to_state_change_days'], axis=1)

kick_data1 = kick_data1.dropna()
kick_data1.isnull().sum()

kick_data1 = kick_data1[(kick_data1['state'] == 'successful') | (kick_data1['state'] == 'failed')]

dummy_category = pd.get_dummies(kick_data1['category'], prefix = 'category')
kick_data1 = kick_data1.join(dummy_category)

dummy_country = pd.get_dummies(kick_data1['country'], prefix = 'country')
kick_data1 = kick_data1.join(dummy_country)

dummy_currency = pd.get_dummies(kick_data1['currency'], prefix = 'currency')
kick_data1 = kick_data1.join(dummy_currency)

dummy_state = pd.get_dummies(kick_data1['state'], prefix = 'state')
kick_data1 = kick_data1.join(dummy_state)

# Finding Correlations between predictors

import seaborn as sn
import matplotlib.pyplot as plt

df = pd.DataFrame(kick_data1, columns=['goal', 'pledged', 'backers_count', 'usd_pledged', 'static_usd_rate'])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot = True)
plt.show()

df = pd.DataFrame(kick_data1, columns=['name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean'])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot = True)
plt.show()

# Dropping Irrelevant Predictors

kick_data2 = kick_data1.drop(['state', 'currency', 'country', 'deadline', 'category',
                              'state_changed_at', 'created_at', 'launched_at',
                              'deadline_weekday', 'state_changed_at_weekday',
                              'created_at_weekday', 'launched_at_weekday', 'spotlight', 'pledged',
                              'backers_count', 'blurb_len', 'name_len', 'usd_pledged', 'pledged'], axis=1)

X = kick_data2.iloc[:, 2: 83]
y = kick_data2['state_successful']

# Finding Relevant Predictors

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(random_state = 0))
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]
selected_feat

# Update X 

X1 = kick_data2[selected_feat]

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size = 0.33, random_state = 5)

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
parameters = {'n_estimators' : [100, 500, 1000], 'min_samples_split' : [10, 100, 500, 1000, 1500]}
grid = GridSearchCV(estimator = model, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(X1_train, y1_train)

# Results from GridSearchCV

print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")    

print("\n The best estimator across ALL searched params:\n",
      grid.best_estimator_)
print("\n The best score across ALL searched params:\n",
      grid.best_score_)
print("\n The best parameters across ALL searched params:\n",
      grid.best_params_)
print("\n ========================================================")

# Run Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 100000)

model = lr.fit(X1_train, y1_train)

y_test_pred = model.predict(X1_test)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y1_test, y_test_pred))

# Calculate the F1 score
from sklearn import metrics
print('F1 Score:', metrics.f1_score(y1_test, y_test_pred))

# Run a Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
model1 = randomforest.fit(X1_train, y1_train)
y_test_pred1 = model1.predict(X1_test)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y1_test, y_test_pred1))

# Calculate the F1 score
from sklearn import metrics
print('F1 Score:', metrics.f1_score(y1_test, y_test_pred1))

# Run GBT Regression with Optimum Values

gbt = GradientBoostingClassifier(random_state = 0, min_samples_split = 1000, n_estimators = 100)
model_gbt_final = gbt.fit(X1_train, y1_train)
y_test_pred2 = model_gbt_final.predict(X1_test)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y1_test, y_test_pred2))

# Calculate the F1 score
from sklearn import metrics
print('F1 Score:', metrics.f1_score(y1_test, y_test_pred2))

## Implementing Clustering

X3 = kick_data1[['goal', 'usd_pledged', 'name_len_clean',
                 'blurb_len_clean', 'backers_count', 'static_usd_rate']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X3_std = scaler.fit_transform(X3)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator

kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeans.fit(X3_std)
    sse.append(kmeans.inertia_)
    
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(1, 11), sse, curve= "convex", direction= "decreasing")
kl.elbow

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X3_std)
    score = silhouette_score(X3_std, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
  
kmeans = KMeans(n_clusters = 3)
model_kmeans_final = kmeans.fit(X3_std)
labels = model_kmeans_final.predict(X3_std)
print(labels)

col_names = X3.columns
for i in range(3):
    print('Cluster:', i)
    for j in range(len(col_names)):
        print(col_names[j], ':', kmeans.cluster_centers_[i][j])
        

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X3_std, labels)

silhouette_score(X3_std, labels)

# Running model on TESTING SET

# Data Preprocessing on TESTING dataset
kick_data_test.isnull().sum()
kick_data_test1 = kick_data_test.drop(['launch_to_state_change_days'], axis=1)

kick_data_test1 = kick_data_test1.dropna()
kick_data_test1.isnull().sum()

# Selecting Rows with only success and failure outcomes
kick_data_test1 = kick_data_test1[(kick_data_test1['state'] == 'successful') | (kick_data_test1['state'] == 'failed')]

# Dummifying categorical data
dummy_category = pd.get_dummies(kick_data_test1['category'], prefix = 'category')
kick_data_test1 = kick_data_test1.join(dummy_category)

dummy_country = pd.get_dummies(kick_data_test1['country'], prefix = 'country')
kick_data_test1 = kick_data_test1.join(dummy_country)

dummy_currency = pd.get_dummies(kick_data_test1['currency'], prefix = 'currency')
kick_data_test1 = kick_data_test1.join(dummy_currency)

dummy_state = pd.get_dummies(kick_data_test1['state'], prefix = 'state')
kick_data_test1 = kick_data_test1.join(dummy_state)

X1_train = kick_data1[selected_feat]
X1_test = kick_data_test1[selected_feat]

y_train = kick_data1['state_successful']
y_test = kick_data_test1['state_successful']

y_test_pred2 = model_gbt_final.predict(X1_test)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_test_pred2))

# Calculate the F1 score
from sklearn import metrics
print('F1 Score:', metrics.f1_score(y_test, y_test_pred2))

# Running clustering model on TESTING data

X3 = kick_data_test1[['goal', 'static_usd_rate']]
labels = model_kmeans_final.predict(X3_std)
print(labels)

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X3_std, labels)

silhouette_score(X3_std, labels)