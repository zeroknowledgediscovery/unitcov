import numpy as np
import pandas as pd
from glob import glob
from subprocess import check_output

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import r2_score

import pylab as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


folder = '../pipeline_data'


with open(f'{folder}/steps.dat', 'r') as handle:
    steps = handle.readline().strip().split()
    
ns = len(steps)

steps_c = [f'case{i}' for i in range(ns)]
steps_d = [f'death{i}' for i in range(ns)]
steps_ = steps_c + steps_d


data_fname = glob(f'{folder}/data_????-??-??.csv')[0]

last_date = data_fname.split('/')[-1][:-4].split('_')[-1]
print(f'last date = {last_date}')
steps_c = steps_c + [f'total_case']
steps_d = steps_d + [f'total_death']
steps_ = steps_c + steps_d

df = pd.read_csv(data_fname, dtype={'fips': str}).set_index('fips')


df['urban_risk'] = df['perc_urban_pop'] * (df['risk_flu'] - df['risk_flu'].min())
covariates = [
    'population', 
    'perc_65yrs', 
    'perc_minority', 
    'perc_black', 
    'perc_hispanic', 
    'perc_poverty', 
    'perc_urban_pop',
    'income',
    'risk_flu',
    'urban_risk'
]

df_z = df[steps_].copy()
for c in covariates:
    mean, std = df[c].mean(), df[c].std()
    df_z[c] = (df[c] - mean) / std
df_z.head()



cov_prefix = 'population+perc_65yrs+perc_minority+perc_black+perc_hispanic+perc_poverty+income+'
cov = cov_prefix + 'perc_urban_pop+risk_flu+urban_risk'


dfs_case = []
cors = []
for i in range(len(steps_c)):
    cur = steps_c[i]
    formula = f'{cur}~{cov}'

    model = smf.glm(
        formula=formula,
        data=df_z,
        family=sm.families.Poisson(sm.families.links.log())
    ).fit()
    print(f'{model.summary()}\n\n')
    
    prd = model.predict()
    cor = pd.DataFrame(
        data={'prd': prd, 'grd': df_z[cur].values}, 
        index=df.index).corr().loc['prd', 'grd']
    cors.append(cor)
    
    df_z[cur + '_glm'] = model.predict()
    
    dfs_case.append(pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0])


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(cors, marker='o')



dfs_death = []
cors = []
for i in range(len(steps_d)):
    cur = steps_d[i]
    formula = f'{cur}~{cov}'

    model = smf.glm(
        formula=formula,
        data=df_z,
        family=sm.families.Poisson(sm.families.links.log())
    ).fit()
    print(f'{model.summary()}\n\n')
    
    prd = model.predict()
    cor = pd.DataFrame(
        data={'prd': prd, 'grd': df_z[cur].values}, 
        index=df.index).corr().loc['prd', 'grd']
    cors.append(cor)
    
    df_z[cur + '_glm'] = model.predict()
    
    dfs_death.append(pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0])


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(cors, marker='o')


existing_csv = glob(f'{folder}/data_????-??-??_glm.csv')

if len(existing_csv) > 0:
    for csv in existing_csv:
        check_output(f'rm {csv}', shell=True)
        
df_z.to_csv(data_fname[:-4] + '_glm.csv', float_format='%.2f')


# cols = [col for col in df_z.columns if (col.startswith('case')) or (col.startswith('death'))]
# print((df_z[cols] < 0).sum().sum())
#
#
# # ## Get latex tables
# # Let us save them to the urisk folder directly for now
# 
# 
# save_folder = '/home/yihuang/Documents/TEX/pub_ds2_/urisk/Figures/data/'
# 
# 
# 
# index_name_map = {
#     'population': '\\texttt{pop}',
#     'perc 65yrs': '\\texttt{$\%65+$}',     
#     'perc minority': '\\texttt{$\%$minority}',
#     'perc black': '\\texttt{$\%$black}',
#     'perc hispanic': '\\texttt{$\%$hispanic}',
#     'perc poverty': '\\texttt{$\%$poverty}',
#     'income': '\\texttt{income}',
#     'perc urban pop': '\\texttt{$\%$urban}',
#     'risk flu': '\\texttt{\\unit}',
#     'urban risk': '\\texttt{urban \\unit}',
# }
# column_name_map = {
#     'z-value': '$z$-value',
#     '0.025': '$.025$',
#     '0.975': '$.975$',
# }
# 
# 
# 
# cols = ['coef', 'z', '[0.025', '0.975]']
# cols_modified = ['coef.', 'z-value', '0.025', '0.975']
# df_total_case = dfs_case[-1][cols]
# df_total_case.rename(columns={c: c_prime for c, c_prime in zip(cols, cols_modified)}, inplace=True)
# df_total_case.rename(index={x: x.replace('_', ' ') for x in df_total_case.index}, inplace=True)
# 
# df_total_case.drop('Intercept', inplace=True)
# df_total_case.rename(index=index_name_map, inplace=True)
# df_total_case.rename(columns=column_name_map, inplace=True)
# 
# tab_case = df_total_case.to_latex(formatters={col: lambda x: f'${x:.3f}$' for col in cols_modified}, escape=False)
# 
# print(tab_case)
# 
# save_fname_case = f'{save_folder}/total_case.tab'
# with open(save_fname_case, 'w') as handle:
#     handle.write(tab_case)
# 
# 
# 
# cols = ['coef', 'z', '[0.025', '0.975]']
# cols_modified = ['coef.', 'z-value', '0.025', '0.975']
# df_total_death = dfs_death[-1][cols]
# df_total_death.rename(columns={c: c_prime for c, c_prime in zip(cols, cols_modified)}, inplace=True)
# df_total_death.rename(index={x: x.replace('_', ' ') for x in df_total_death.index}, inplace=True)
# 
# df_total_death.drop('Intercept', inplace=True)
# df_total_death.rename(index=index_name_map, inplace=True)
# df_total_death.rename(columns=column_name_map, inplace=True)
# 
# tab_death = df_total_death.to_latex(formatters={col: lambda x: f'${x:.3f}$' for col in cols_modified}, escape=False)
# 
# print(tab_death)
# 
# save_fname_death = f'{save_folder}/total_death.tab'
# with open(save_fname_death, 'w') as handle:
#     handle.write(tab_death)
# 
# 
# 
# cols = ['coef', 'z', 'P>|z|', '[0.025', '0.975]']
# cols_modified = ['coef.', 'z-value', 'p-value', '0.025', '0.975']
# 
# inds = [
#     'Intercept', 'population', 
#     'perc_65yrs', 'perc_minority', 
#     'perc_black', 'perc_hispanic', 
#     'perc_poverty', 'income', 
#     'perc_urban_pop', 
#     'risk_flu', 'urban_risk'
# ]
# inds_modified = [
#     'Intercept', 'pop', 
#     '$\%65+$', '$\%$minority', 
#     '$\%$black', '$\%$hispanic', 
#     '$\%$poverty', 'income', 
#     '$\%$urban', 
#     'pre-UnIT', 'UnIT'
# ]
# 
# 
# 
# for i, s in enumerate(steps):
#     print(f'{i}: {s}')
# 
# 
# 
# def foo(x):
#     y = abs(x)
#     if y < 1:
#         return f'${x:.3f}$'
#     elif (y >= 1) and (y < 10):
#         return f'${x:.2f}$'
#     else:
#         return f'${x:.1f}$'
# 
# 
# 
# start = 19
# mydict = {'case': dfs_case, 'death': dfs_death}
# for target, dfs in mydict.items():
#     data = {}
#     abnormal_p_values = []
#     for i in range(start, len(steps)):
# 
#         tmp_df = dfs[i][cols].copy()
#         tmp_df.drop('Intercept', axis=0, inplace=True)
#         tmp_df = tmp_df.rename(columns={c: c_prime for c, c_prime in zip(cols, cols_modified)}).T
#         tmp_df = tmp_df.rename(columns={c: c_prime for c, c_prime in zip(inds, inds_modified)}).T
#         
#         for col in ['z-value', '0.025', '0.975']:
#             key = (steps[i][5:], column_name_map[col])
#             data[key] = [foo(x) for x in tmp_df[col]]
#         
#         key = (steps[i][5:], 'coef.')
#         coefs = []
#         for idx in tmp_df.index:
#             value = foo(tmp_df.loc[idx, 'coef.'])
#             if tmp_df.loc[idx, 'p-value'] < 0.01:
#                 coefs.append("{}".format(value))
#             elif tmp_df.loc[idx, 'p-value'] >= 0.01 and tmp_df.loc[idx, 'p-value'] < 0.05:
#                 coefs.append("\\textcolor{{{}}}{{{}}}".format('blue', value))
#             else:
#                 coefs.append("\\textcolor{{{}}}{{{}}}".format('red', value))
#         data[key] = coefs
# 
#     index = [f'\\texttt{{{idx}}}' for idx in tmp_df.index]
#     df = pd.DataFrame(data=data, index=index).T
#     print(df)
#     
#     tab_weekly = df.to_latex( 
#         escape=False, 
#         multirow=True)
# 
#     save_fname = f'{save_folder}/weekly_{target}_{steps[start]}.tab'
#     with open(save_fname, 'w') as handle:
#         handle.write(tab_weekly)

