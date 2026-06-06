import pandas as pd
customer_data = pd.read_csv('customer_shopping_data.csv')
# print(customer_data)

customer_encode = customer_data[['gender','category','payment_method','invoice_date','shopping_mall']]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in customer_encode.columns:
    customer_encode[c] = le.fit_transform(customer_encode[c])
# print(customer_encode)

c = customer_data.drop(columns=['invoice_no','customer_id','gender','category','payment_method','invoice_date','shopping_mall'])
# print(c)
customer_dataset = pd.concat([c,customer_encode],axis=1)
# print(customer_dataset)

# Applying K-Means clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(customer_dataset)

# Extracting cluster labels and centroids
labels = km.labels_
df_labels = pd.DataFrame(labels,columns=['cluster_assignment'])
centroids = km.cluster_centers_

# Adding cluster labels to dataset
customer_dataset_final = pd.concat([customer_dataset,df_labels],axis=1)
# print(customer_dataset_final)

# Calculating SSE for each cluster
import numpy as np
sse = []
for i in range(3):
    l = labels==i
    data_points = customer_dataset[l]
    centroid = centroids[i]
    s_err = np.sum((data_points - centroid)**2)
    sse.append(s_err)
print("SSE per cluster :",sse)

# Visualizing customer clusters
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))

plt.scatter(customer_dataset['quantity'],customer_dataset['price'],c=labels)
plt.scatter(centroids[:,1],centroids[:,2],marker='X',s=100,color='red',label='Centroids')
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.title("Customer Segmentation based on Purchase History")
plt.legend()
plt.show()