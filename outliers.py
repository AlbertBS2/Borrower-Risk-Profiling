import pandas as pd
from data_preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


loan_data = "data/accepted_2007_to_2018Q4.csv.gz"
unemployment_rate_data = ["data/unemployment_rate_0.csv", "data/unemployment_rate_1.csv", "data/unemployment_rate_2.csv", "data/unemployment_rate_3.csv", "data/unemployment_rate_4.csv"]

data = preprocess_data(loan_data, unemployment_rate_data)
X = data.copy()
y = X.pop('default')

X = data.drop(columns=['default']).copy()

# cap anything beyond |z|>4
z = (X - X.mean()) / X.std(ddof=0)
X_clipped = X.mask(z > 4, X.mean()).mask(z < -4, X.mean())

# then impute (if needed) and scale
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X_clipped)
X_scaled = StandardScaler().fit_transform(X_imp)





# Number of components PCA
n = 20
pca = PCA(n_components=n)

# Fit and transform
X_pca = pca.fit_transform(X_scaled)

import numpy as np
loading_strength = np.abs(pca.components_[0])  # first PC
top_features = np.argsort(loading_strength)[::-1][:10]  # top 10 features
print("Top 10 features contributing to the first principal component:")
print(data.columns[top_features])



#K-means
from sklearn.cluster import KMeans

km = KMeans(n_clusters=5, n_init='auto', random_state=0)
labels = km.fit_predict(X_pca)


# Add cluster labels to the original data
data['cluster'] = labels

data['cluster'].value_counts()

