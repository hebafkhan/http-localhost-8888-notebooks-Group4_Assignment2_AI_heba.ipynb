#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Assignment2 Predict the outliers
# Import necessary libraries for data manipulation and machine learning

import pandas as pd  # Importing pandas for data handling and analysis

import numpy as np  # Importing NumPy for numerical operations

from sklearn.model_selection import train_test_split  # Importing function to split data into training and testing sets

from sklearn.preprocessing import StandardScaler  # Importing for feature scaling

from sklearn.ensemble import IsolationForest  # Importing Isolation Forest algorithm for outlier detection

from sklearn.cluster import DBSCAN  # Importing DBSCAN clustering algorithm for density-based outlier detection

from sklearn.svm import OneClassSVM  # Importing One-Class SVM for semi-supervised outlier detection

from sklearn.neighbors import LocalOutlierFactor  # Importing Local Outlier Factor algorithm for outlier detection

from sklearn.metrics import f1_score, confusion_matrix  # Importing metrics for evaluating model performance

import time  # Importing time module for measuring execution time

import matplotlib.pyplot as plt  # Importing matplotlib for plotting graphs

import seaborn as sns  # Importing seaborn for enhanced data visualization

import os  # Importing os module for operating system functionalities (like file path handling)

import zipfile


# In[6]:


# Summary Table
summary_table = pd.DataFrame({
    'Model': ['Isolation Forest', 'One Class SVM', 'Local Outlier Factor', 'DBSCAN'],
    'Task': ['F1 Score Calculated', 'F1 Score Calculated', 'F1 Score Calculated', 'F1 Score Calculated'],
    'Confusion Matrix Built': ['Yes', 'Yes', 'Yes', 'Yes'],
    'Issues': ['', '', '', ''],
    'Train Time': ['', '', '', ''],
    'Training F1': ['', '', '', ''],
    'Test F1': ['', '', '', '']
})

print(summary_table)


# In[7]:


# Load the data

# Define the directory where your CSV file is located
extract_dir = r'C:\Users\heebs\Downloads\NetworkIntrusionDetection'
csv_file_name = 'NetworkIntrusionDetection.csv'

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\heebs\Downloads\NetworkIntrusionDetection\NetworkIntrusionDetection.csv')

# Load the dataset
csv_file_path = os.path.join(extract_dir, csv_file_name)


# In[8]:


# We are randomly sample data since the dataset is too large
# We are sampling to save execution time and memory.

sample_size = 10000  # Setting the desired sample size
data_sample = data.sample(n=sample_size, random_state=42)

# Convert categorical variables to numeric
if 'label' in data_sample.columns:
    data_sample['label'] = data_sample['label'].map({'normal': 0, 'malicious': 1})

# Separate features and target
X = data_sample.drop('label', axis=1)
y = data_sample['label']

# Convert labels to binary (0 for benign and 1 for malicious)
y = (y == 1).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[9]:


# Function to calculate F1 score and confusion matrix
def evaluate_model(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return f1, cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Initializing results table
results = {
    'Model': [],
    'F1 Score Calculated': [],
    'Confusion Matrix Built': [],
    'Issues': [],
    'Train Time': [],
    'Training F1': [],
    'Test F1': []
}


# In[10]:


# Isolation Forest
print("Running Isolation Forest")
for i in range(3):
    contamination = 0.1 + i * 0.05  # Adjust contamination
    iforest = IsolationForest(contamination=contamination, random_state=42)
    
    start_time = time.time()
    iforest.fit(X_train_scaled)
    train_time = time.time() - start_time
    
    y_pred_train = iforest.predict(X_train_scaled)
    y_pred_test = iforest.predict(X_test_scaled)
    
    # Converting predictions to binary (1 for inliers, 0 for outliers)
    y_pred_train = (y_pred_train == 1).astype(int)
    y_pred_test = (y_pred_test == 1).astype(int)
    
    train_f1, train_cm = evaluate_model(y_train, y_pred_train)
    test_f1, test_cm = evaluate_model(y_test, y_pred_test)
    
    results['Model'].append(f'Isolation Forest (iter {i+1})')
    results['F1 Score Calculated'].append('Yes')
    results['Confusion Matrix Built'].append('Yes')
    results['Issues'].append('')
    results['Train Time'].append(train_time)
    results['Training F1'].append(train_f1)
    results['Test F1'].append(test_f1)
    
    plot_confusion_matrix(test_cm, f'Isolation Forest Confusion Matrix (iter {i+1})')


# In[11]:


# DBSCAN
print("Running DBSCAN")
for i in range(3):
    eps = 0.5 + i * 0.25  # Adjust eps
    dbscan = DBSCAN(eps=eps, min_samples=5)
    
    start_time = time.time()
    dbscan.fit(X_train_scaled)
    train_time = time.time() - start_time
    
    y_pred_train = dbscan.labels_
    y_pred_test = dbscan.fit_predict(X_test_scaled)
    
    # Converting predictions to binary (1 for inliers, 0 for outliers)
    y_pred_train = (y_pred_train != -1).astype(int)
    y_pred_test = (y_pred_test != -1).astype(int)
    
    train_f1, train_cm = evaluate_model(y_train, y_pred_train)
    test_f1, test_cm = evaluate_model(y_test, y_pred_test)
    
    results['Model'].append(f'DBSCAN (iter {i+1})')
    results['F1 Score Calculated'].append('Yes')
    results['Confusion Matrix Built'].append('Yes')
    results['Issues'].append('')
    results['Train Time'].append(train_time)
    results['Training F1'].append(train_f1)
    results['Test F1'].append(test_f1)
    
    plot_confusion_matrix(test_cm, f'DBSCAN Confusion Matrix (iter {i+1})')


# In[12]:


# One-Class SVM
print("Running One-Class SVM")
for i in range(3):
    nu = 0.1 + i * 0.05  # Adjust nu
    ocsvm = OneClassSVM(kernel='rbf', nu=nu)
    
    start_time = time.time()
    ocsvm.fit(X_train_scaled)
    train_time = time.time() - start_time
    
    y_pred_train = ocsvm.predict(X_train_scaled)
    y_pred_test = ocsvm.predict(X_test_scaled)
    
    # Converting predictions to binary (1 for inliers, 0 for outliers)
    y_pred_train = (y_pred_train == 1).astype(int)
    y_pred_test = (y_pred_test == 1).astype(int)
    
    train_f1, train_cm = evaluate_model(y_train, y_pred_train)
    test_f1, test_cm = evaluate_model(y_test, y_pred_test)
    
    results['Model'].append(f'One-Class SVM (iter {i+1})')
    results['F1 Score Calculated'].append('Yes')
    results['Confusion Matrix Built'].append('Yes')
    results['Issues'].append('')
    results['Train Time'].append(train_time)
    results['Training F1'].append(train_f1)
    results['Test F1'].append(test_f1)
    
    plot_confusion_matrix(test_cm, f'One-Class SVM Confusion Matrix (iter {i+1})')


# In[13]:


# Local Outlier Factor
print("Running Local Outlier Factor")
for i in range(3):
    n_neighbors = 20 + i * 10  # Adjust number of neighbors
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    
    start_time = time.time()
    y_pred_train = lof.fit_predict(X_train_scaled)
    train_time = time.time() - start_time
    
    y_pred_test = lof.fit_predict(X_test_scaled)
    
    # Converting predictions to binary (1 for inliers, 0 for outliers)
    y_pred_train = (y_pred_train != -1).astype(int)
    y_pred_test = (y_pred_test != -1).astype(int)
    
    train_f1, train_cm = evaluate_model(y_train, y_pred_train)
    test_f1, test_cm = evaluate_model(y_test, y_pred_test)
    
    results['Model'].append(f'Local Outlier Factor (iter {i+1})')
    results['F1 Score Calculated'].append('Yes')
    results['Confusion Matrix Built'].append('Yes')
    results['Issues'].append('')
    results['Train Time'].append(train_time)
    results['Training F1'].append(train_f1)
    results['Test F1'].append(test_f1)
    
    plot_confusion_matrix(test_cm, f'Local Outlier Factor Confusion Matrix (iter {i+1})')


# In[14]:


# Converting results into a DataFrame
results_df = pd.DataFrame(results)

# Displaying the final summary table
print(results_df)

# Saveing the results to a CSV file
results_df.to_csv('outlier_detection_results.csv', index=False)


# In[15]:


# Comparing supervised and unsupervised approaches
supervised_f1 = results_df[results_df['Model'].str.contains('Isolation Forest|DBSCAN')]['Test F1'].mean()
unsupervised_f1 = results_df[results_df['Model'].str.contains('One-Class SVM|Local Outlier Factor')]['Test F1'].mean()

print(f"Average F1 Score for Supervised Approaches: {supervised_f1:.4f}")
print(f"Average F1 Score for Unsupervised Approaches: {unsupervised_f1:.4f}")

if supervised_f1 > unsupervised_f1:
    print("Supervised approaches performed better on average.")
else:
    print("Unsupervised approaches performed better on average.")


# In[ ]:




