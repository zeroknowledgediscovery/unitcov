#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import *
from glob import glob


# In[2]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


# In[3]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


# In[4]:


import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# ## Get data

# In[5]:


with open('pipeline_data/steps.dat', 'r') as handle:
    steps = handle.readline().strip().split()


# In[6]:


csv_fname = glob('pipeline_data/data_????-??-??_glm.csv')[0]
df = pd.read_csv(csv_fname, dtype={'fips': str}).set_index('fips')
df.head()


# ## Functions

# In[7]:


class tfRegr:
    
    def __init__(self, 
                 epoch=100, 
                 verbose=False, 
                 validation_split=0.2, 
                 learning_rate=.1):
        
        self.normalizer = preprocessing.Normalization()
        self.ep = epoch
        self.vb = verbose
        self.vs = validation_split
        self.lr = learning_rate
    
    def fit(self, X, y):
        self.normalizer.adapt(X)

        self.model = tf.keras.Sequential([
            self.normalizer,
            layers.Dense(units=1)
        ])

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.lr),
            loss='mean_absolute_error'
        )

        self.history = self.model.fit(
            X, y,
            epochs=self.ep,
            verbose=self.vb, # logging
            validation_split=self.vs # Calculate validation results on a fraction of the training data
        )
        return self.model


# In[8]:


def predict(df, cur_idx):

    regrs = {
        'RF': RandomForestRegressor(min_samples_split=2), 
        'ET': ExtraTreesRegressor(min_samples_split=2), 
        'TF': tfRegr(),
    }
    
    cur = f'case{cur_idx}'

    glm = [f'case{cur_idx - 1}_glm', f'case{cur_idx}_glm'] 
    cols = [f'case{cur_idx - 1}'] + glm
    X, y = df[cols].values, df[cur].values 
    
    
    for name, regr in regrs.items():
        regr = regr.fit(X, y)

        #================ Evaluation ===============START
        y_pred = regr.predict(X).flatten()
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(((y - y_pred) ** 2).mean())
        print(f'\t{name}: r2 = {r2:.3f}, rmse = {rmse:.3f}')
        #================ Evaluation ===============START
        
        cols = [cur] + glm
        df[cur + f'_{name}_1'] = regr.predict(df[cols].values).flatten()
        
        cols = [cur + f'_{name}_1'] + glm
        df[cur + f'_{name}_2'] = regr.predict(df[cols].values).flatten()
    
    prd_cols_1 = [col for col in df.columns if col.endswith('_1')]
    prd_cols_2 = [col for col in df.columns if col.endswith('_2')]
    prd_1 = df[prd_cols_1].mean(axis=1).values
    prd_2 = df[prd_cols_2].mean(axis=1).values
    
    df.drop(prd_cols_1 + prd_cols_2, axis=1, inplace=True)
    
    return prd_1, prd_2


# ## Forecast Runs

# In[9]:


cur_idx = len(steps) - 1
print(steps[cur_idx])

num_runs = 100
prds_1, prds_2 = [], []

for r in range(num_runs):
    print(f'run = {r}')
    prd_1, prd_2 = predict(df, cur_idx)
    prds_1.append(prd_1)
    prds_2.append(prd_2)


# ## Prepare for submission

# In[10]:


# target end dates
cur_dt = datetime.strptime(steps[cur_idx], '%Y-%m-%d')
fc_dt = (cur_dt + timedelta(days=1)).date()
nxt_dt_1 = (cur_dt + timedelta(days=7)).date()
nxt_dt_2 = (cur_dt + timedelta(days=14)).date()
nxt_dts = [nxt_dt_1, nxt_dt_2]

# quantiles
qs = [.025, .1, .25, .5, .75, .9, .975]


# In[11]:


df_county_prd_1 = pd.DataFrame(data=np.array(prds_1).T, index=df.index, columns=range(num_runs))
df_county_prd_2 = pd.DataFrame(data=np.array(prds_2).T, index=df.index, columns=range(num_runs))
county_dfs = [df_county_prd_1, df_county_prd_2]


# ### County

# In[12]:


forecast_dfs = []

for i, county_df in enumerate(county_dfs):
    w = i + 1
    csv_fname = f'results/{steps[cur_idx]}_county_case_{w}-wk_{num_runs}.csv'
    county_df.to_csv(csv_fname, float_format='%.2f')
    
    # ========================= Point esitmate =========================START
    df_point = county_df.mean(axis=1).reset_index()        .rename(columns={0: 'value', 'fips': 'location'})
    df_point['type'] = 'point'
    df_point['quantile'] = 'NA'
    # ========================= Point esitmate =========================END

    # ========================= Quantile esitmates =========================START
    df_quantile = county_df.quantile(qs, axis=1).T.reset_index()        .melt(id_vars=['fips']).sort_values('fips')        .rename(columns={'fips': 'location', 'variable': 'quantile'})
    df_quantile['type'] = 'quantile'
    # ========================= Quantile esitmates =========================END

    df_forecast = pd.concat([df_point, df_quantile])
    df_forecast['target'] = f'{w} wk ahead inc case'
    df_forecast['target_end_date'] = nxt_dts[i]
    
    forecast_dfs.append(df_forecast)

df_forecast_county = pd.concat(forecast_dfs, axis=0).reset_index(drop=True)
print(df_forecast_county.shape)
df_forecast_county.head()


# ### States

# In[13]:


forecast_dfs = []

for i, county_df in enumerate(county_dfs):
    w = i + 1
    
    df_tmp = county_df.reset_index()
    df_tmp['state'] = df_tmp['fips'].apply(lambda x: x[:2])
    state_df = df_tmp.groupby(by='state').sum()
    
    # ========================= Point esitmate =========================START
    df_point = state_df.mean(axis=1).reset_index()        .rename(columns={0: 'value', 'state': 'location'})
    df_point['type'] = 'point'
    df_point['quantile'] = 'NA'
    # ========================= Point esitmate =========================END

    # ========================= Quantile esitmates =========================START
    df_quantile = state_df.quantile(qs, axis=1).T.reset_index()        .melt(id_vars=['state']).sort_values('state')        .rename(columns={'state': 'location', 'variable': 'quantile'})
    df_quantile['type'] = 'quantile'
    # ========================= Quantile esitmates =========================END

    df_forecast = pd.concat([df_point, df_quantile])
    df_forecast['target'] = f'{w} wk ahead inc case'
    df_forecast['target_end_date'] = nxt_dts[i]
    
    forecast_dfs.append(df_forecast)

df_forecast_state = pd.concat(forecast_dfs, axis=0).reset_index(drop=True)
df_forecast_state.head()


# ### Nation

# In[14]:


forecast_dfs = []

for i, county_df in enumerate(county_dfs):
    w = i + 1
    
    nation_df = pd.DataFrame(data=county_df.sum(axis=0), columns=['US']).T
    nation_df.index.name = 'nation'
    
    # ========================= Point esitmate =========================START
    df_point = nation_df.mean(axis=1).reset_index()        .rename(columns={0: 'value', 'nation': 'location'})
    df_point['type'] = 'point'
    df_point['quantile'] = 'NA'
    # ========================= Point esitmate =========================END

    # ========================= Quantile esitmates =========================START
    df_quantile = nation_df.quantile(qs, axis=1).T.reset_index()        .melt(id_vars=['nation']).sort_values('nation')        .rename(columns={'nation': 'location', 'variable': 'quantile'})
    df_quantile['type'] = 'quantile'
    # ========================= Quantile esitmates =========================END

    df_forecast = pd.concat([df_point, df_quantile])
    df_forecast['target'] = f'{w} wk ahead inc case'
    df_forecast['target_end_date'] = nxt_dts[i]
    
    forecast_dfs.append(df_forecast)

df_forecast_nation = pd.concat(forecast_dfs, axis=0).reset_index(drop=True)
df_forecast_nation


# In[15]:


df_forecast = pd.concat([df_forecast_county, df_forecast_state, df_forecast_nation])
df_forecast.reset_index(drop=True, inplace=True)
df_forecast['forecast_date'] = fc_dt
cols = ['forecast_date', 'target', 'target_end_date', 'location', 'type', 'quantile', 'value']
df_forecast = df_forecast[cols]

# =============== make nagative value zero ==============
value_dezero = df_forecast['value']
value_dezero[value_dezero < 0]  = 0
df_forecast['value'] = value_dezero


# In[16]:


team = 'UChicagoCHATTOPADHYAY'
model = 'UnIT'
df_forecast.to_csv(f'results/{fc_dt}-{team}-{model}_case.csv', float_format='%.2f', index=False)


# In[17]:


validation = '/home/yihuang/Documents/Data/covid19-forecast-hub/code/validation/validate_single_forecast_file.py'

get_ipython().system(' python3 {validation} results/2020-11-08-UChicagoCHATTOPADHYAY-UnIT_case.csv')


# In[ ]:




