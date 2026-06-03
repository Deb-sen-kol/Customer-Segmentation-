#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 16:47:13 2025

@author: debasmitasen
"""
import pandas as pd

df = pd.read_csv('/Users/debasmitasen/Downloads/customer_segmentation.csv')
df.head()
df.describe()
df.info()
df.isnull().sum()

product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]
product_data = df[product_cols]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
product_scaled = scaler.fit_transform(product_data)

from sklearn.cluster import KMeans

# Use the elbow method to decide optimal k
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(product_scaled)
    inertia.append(km.inertia_)

# Plot elbow
import matplotlib.pyplot as plt
import seaborn as sns


plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df['ProductSegment'] = kmeans.fit_predict(product_scaled)

# Show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Now print your DataFrame again
cluster_means = df.groupby('ProductSegment')[[
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]].mean()
print(df.groupby('ProductSegment')[[
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]].mean())

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Average Spend per Product Type by Customer Segment')
plt.ylabel('Customer Segment')
plt.xlabel('Product Category')
plt.show()

df[['ID', 'ProductSegment'] + product_cols].head()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load your dataset
# Replace with actual data load, e.g., pd.read_csv("yourfile.csv")
# df = pd.read_csv("your_data.csv")

# Sample structure - assuming you already have the dataset as `df`
# We'll now compute the derived features

# STEP 2: Feature Engineering
df['MarketingCampaignAccepted'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)

# STEP 3: Select features for segmentation
features = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
            'NumStorePurchases', 'NumWebVisitsMonth', 'MarketingCampaignAccepted']

X = df[features]

# STEP 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5: Choose number of clusters using Elbow Method (optional)
# Uncomment if you want to determine optimal K
# sse = []
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(X_scaled)
#     sse.append(kmeans.inertia_)
# plt.plot(range(1, 10), sse, marker='o')
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.title("Elbow Method For Optimal k")
# plt.show()

# STEP 6: Apply KMeans with chosen k
k = 3  # you can change based on elbow plot
kmeans = KMeans(n_clusters=k, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# STEP 7: Analyze clusters
cluster_summary = df.groupby('Cluster')[features].mean().round(2)
print("Cluster Summary:")
print(cluster_summary)

# Apply KMeans clustering
k_optimal = 3  # use your chosen value from elbow curve
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Count number of customers in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()

# Convert to percentages
cluster_percentages = (cluster_counts / len(df)) * 100

# Display nicely
cluster_summary_df = pd.DataFrame({
    'Cluster': cluster_counts.index,
    'Count': cluster_counts.values,
    'Percentage': cluster_percentages.round(2)
})

print(cluster_summary_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.pie(cluster_percentages, labels=cluster_percentages.index, autopct='%1.1f%%', 
        colors=sns.color_palette("Set2"), startangle=140)
plt.title("Cluster Size Distribution (%)")
plt.show()

