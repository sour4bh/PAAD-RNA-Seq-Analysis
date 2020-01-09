#%%
import pandas as pd
import numpy as np
import scipy

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
gct = parse('data/PAAD.gct')
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
pca = PCA(n_components=20, random_state=42)
componenets = pd.DataFrame(pca.fit_transform(X))
#%%
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
# %%
componenets['labels'] = Y
# %%
componenets.columns = list(map(str, componenets.columns))
sns.scatterplot(x='0', y='1', hue='labels', data=componenets)
# %%
inertias = []
for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(componenets.iloc[:,:3])
    inertias.append(kmeans.inertia_)
plt.plot(range(1,10), inertias, '-o')
plt.show()
# elbows at k=3
components = componenets.iloc[:,:3]
#%%
# K-Means using k=3 clusters on first 3 principal components
kmeans = KMeans(n_clusters=3, random_state=42).fit(componenets.iloc[:,:3])
componenets['predict'] = kmeans.predict(componenets.iloc[:,:3])
sns.scatterplot(x='0', y='1', style='predict', hue='labels', data=componenets)
# %%
# missclassifications 
componenets['histological_type'] = meta_col['histological_type'].values
componenets['histological_type_other'] = meta_col['histological_type_other'].values
componenets[['labels', 'predict', 'histological_type', 'histological_type_other']][componenets.predict == 1]
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
# remove neuroendocrine 
delete = componenets[['labels', 'predict', 'histological_type', 'histological_type_other']][componenets.predict == 1].index
data2 = data.drop(index=[data.index[i] for i in delete])
meta_col2 = meta_col.drop(index=[data.index[i] for i in delete])
data2.shape, meta_col2.shape
# %%
# Load IFNS 
ifns = []
with open('data/type1_IFN.txt', 'r') as f:
    ifns = f.read().split('\n')
print('ifns :', len(ifns))
check = True
for ifn in ifns:
    check = check and ifn in data.columns
print('all present :', check)

# %%    
plt.figure(figsize=(15,10))
sns.boxplot(data=data2[ifns])

# %%
import argparse, sys, os
import pandas as pd 
# from tempfile import mkdtemp, gettempdir
from subprocess import Popen, PIPE
import subprocess

# R GSVA function call params:
method='gsva'
kcdf='Gaussian'
abs_ranking=False
min_sz=1
max_sz=None
parallel_sz=0
parallel_type="SOCK"
mx_diff_options=[True, False] # two gsva calculations as on Hänzelmann, S., Castelo, R. & Guinney, J. GSVA
tau=None
ssgsea_norm=True
verbose=False
tempdir= None

df = data2.T
genset = pd.DataFrame(data={'name': ['type1ifn' for i in range(len(ifns))], 'member': ifns})

try:
    os.mkdir('temp')
except FileExistsError:
    pass
df.to_csv('temp/expr.csv')

members = genset['member'].unique()
genset = genset.groupby(['name']).apply(lambda x: "\t".join(sorted(list(x['member'])))).reset_index().rename(columns={0:'members'})
# Write our gene sets
with open('temp/gs.gmt','w') as f:
    for row in genset.itertuples():
        name = row.name
        description = 'description'
        fields = row.members
        f.write(name+"\t"+description+"\t"+fields+"\n")
cur = os.path.dirname(os.path.realpath(__file__))
temp = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp')
rscript = os.path.join(cur, 'gsva.r')
for mx_diff in mx_diff_options:
    cmd = ["Rscript", rscript]
    cmd += [str(x) for x in [method,kcdf,abs_ranking,min_sz,max_sz,parallel_sz,parallel_type,mx_diff,tau,ssgsea_norm,verbose,temp]]
    print('Rscript command to execute : ')
    print(cmd)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    print(result.stdout.decode('utf-8'))
    try:
        os.remove(f'temp/gsva_scores_mx_diff_{"t" if mx_diff else "f"}.csv')
    except FileNotFoundError:
        pass
    finally:
        os.rename('temp/gsva_scores.csv', f'temp/gsva_scores_mx_diff_{"t" if mx_diff else "f"}.csv')
# %%
gsva_scores = pd.read_csv('temp/gsva_scores_mx_diff_t.csv').iloc[:,1:]
gsva_scores = gsva_scores.append(pd.read_csv('temp/gsva_scores_mx_diff_f.csv').iloc[:,1:])
gsva_scores.index = ['mx_diff_true', 'mx_diff_false']
gsva_scores
# %%
plt.figure(figsize=(10,5))
plt.title('Difference between largest positive and negative deviations')
sns.distplot(gsva_scores.loc['mx_diff_true'], axlabel='GSAV Score')
# %%
plt.figure(figsize=(10,5))
plt.title('Maximum deviation from Zero')
sns.distplot(gsva_scores.loc['mx_diff_false'], axlabel='GSAV Score')
#%%


# %%
