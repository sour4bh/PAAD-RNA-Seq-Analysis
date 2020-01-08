#%%
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.GCToo import GCToo
from cmapPy.pandasGEXpress.parse import parse
#%%
gct = parse('PAAD.gct')
#%%

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

