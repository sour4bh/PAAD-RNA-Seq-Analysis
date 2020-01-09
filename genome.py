#%%
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.GCToo import GCToo

from cmapPy.pandasGEXpress.parse import parse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
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
for yi, id_ in enumerate(Y):
    Y[yi] = 0
    if id_ in nids:
       Y[yi] = 1
np.array(Y)
# %%
X = data.to_numpy().transpose()
X = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X)
X = StandardScaler().fit(X).transform(X)
# %%
X, x, Y, y = train_test_split(X, Y, test_size=0.2, random_state=42)
print(x, y)
lr = LogisticRegression(penalty='l1')
lr.fit(X, Y)
Xd = lr.decision_function(X)
import seaborn as sns
sns.lineplot(data=Xd)
#%%
sns.lineplot(data=list(lr.coef_))
print(lr.score(x,y))
# %%
for col in data.columns:
    if col in nids:
        print(data[col].isna().value_counts())
