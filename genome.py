#%%
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.GCToo import GCToo

from cmapPy.pandasGEXpress.parse import parse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt
#%%
gct = parse('PAAD.gct')
data = gct.data_df
meta_col = gct.col_metadata_df
meta_row = gct.row_metadata_df
#%%
data.shape, meta_col.shape, meta_row.shape
#%%
# unique columns : sample types
set([col.split('-')[1] for col in data.columns])
#%%
cols = [print(col) for col in meta_col.columns]
print(len(cols))

#%%
for hist, meta in meta_col.groupby('histological_type'):
    print(hist, ':')
    for hist_other, other_meta in meta.groupby('histological_type_other'):
        print('\t', hist_other)
#%%
# neuroendocrine ids 
nids = []
for hist_other, other_meta in meta_col.groupby('histological_type_other'):
    if 'neuroendocrine' in hist_other and not 'moderately' in hist_other:
        # print(len(other_meta.index))
        for id_ in other_meta.index:
            nids.append(id_)
print('#neuroendocrine : ', len(nids))
print(nids)
#%%
Y = data.columns.tolist()
for i, id_ in enumerate(Y):
    Y[i] = 0
    if id_ in nids:
       Y[i] = 1
Y = np.array(Y)
#%%
# missing gene values per sample
missing = data.shape[0] -  data.count()
print(missing.describe())
sns.distplot(missing)
#%%
# transpose data
data = data.T
#%%
# look for missing nan values for each gene
missing = data.shape[0] -  data.count()
print(missing.describe())
sns.distplot(missing)
#%%
# Using KNN based imputer for missing values ref: https://academic.oup.com/bib/article/12/5/498/268546
# X = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X)
X = data.to_numpy()
X = KNNImputer(n_neighbors=5).fit_transform(X)
X = StandardScaler().fit_transform(X)
#%%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(y_test)
#%%
# # of neoendocrine samples in test data 
print('# of neoendocrine samples in test data :', y_test.sum())
#%%
lr = LogisticRegression(penalty='l2')
lr.fit(x_train, y_train)
sns.lineplot(data=lr.decision_function(x_test))
#%%
print(lr.score(x_test, y_test))
# %%
sns.lineplot(data=list(lr.coef_))
#%%
lr = LogisticRegression(penalty='l1', solver='liblinear')
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
sns.lineplot(data=list(lr.coef_))
#%%
pca = PCA(n_components=20, random_state=42)
componenets = pd.DataFrame(pca.fit_transform(X))
#%%
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
# %%
componenets.columns = list(map(str, componenets.columns))
componenets['labels'] = Y
componenets
# %%
sns.scatterplot(x='0', y='1', hue='labels', data=componenets)
# %%
inertias = []
for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(componenets.iloc[:,:3])
    inertias.append(kmeans.inertia_)
plt.plot(range(1,10), inertias, '-o')
plt.show()
#%%

# %%
