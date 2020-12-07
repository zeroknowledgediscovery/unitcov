import pandas as pd
from glob import glob
from subprocess import check_output
from datetime import date



folder = '../pipeline_data'


def modify_date(date):
    m, d, y = list(map(int, date.split('/')))
    date_vec = [y, m, d]
    return '20' + '-'.join([str(x).zfill(2) for x in date_vec])


url_case = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
df_case = pd.read_csv(url_case).dropna()
df_death = pd.read_csv(url_death).dropna()

print(f'JHU case data shape = {df_case.shape}')
print(f'JHU death data shape = {df_death.shape}')


dates_case = sorted([modify_date(col) for col in df_case.columns if col.endswith('/20') or col.endswith('/21')])
dates_death = sorted([modify_date(col) for col in df_death.columns if col.endswith('/20') or col.endswith('/21')])

date_check = (dates_case[0] == dates_death[0]) & (dates_case[-1] == dates_death[-1])
print(f'Date check passed? {date_check}')

csv_case_fname = url_case.split('/')[-1][:-4] + '_' + dates_case[-1] + '.csv'
csv_death_fname = url_death.split('/')[-1][:-4] + '_' + dates_case[-1] + '.csv'

existing_case_csv = glob(f'{folder}/time_series_covid19_confirmed_US*csv')
existing_death_csv = glob(f'{folder}/time_series_covid19_deaths_US*csv')
if len(existing_case_csv) > 0:
    for csv in existing_case_csv:
        check_output(f'rm {csv}', shell=True)
if len(existing_death_csv) > 0:
    for csv in existing_death_csv:
        check_output(f'rm {csv}', shell=True)    

df_case.to_csv(f'{folder}/{csv_case_fname}', index=False)
df_death.to_csv(f'{folder}/{csv_death_fname}', index=False)


glob(f'{folder}/time_series_covid19_*csv')


start = '2020-04-04'
end = date.today()
dates_ = pd.date_range(start, end, freq='7D')
# print(dates_)


df_case.rename(
    columns={col: modify_date(col) for col in df_case.columns if col.endswith('/20') or col.endswith('/21')}, 
    inplace=True)
df_death.rename(
    columns={col: modify_date(col) for col in df_death.columns if col.endswith('/20') or col.endswith('/21')}, 
    inplace=True)


dates = sorted(list(set(df_case.columns.values).intersection([str(d.date()) for d in dates_])))


df_case['FIPS'] = df_case['FIPS'].apply(lambda x: str(int(x)).zfill(5))
df_case = df_case.rename(columns={'FIPS': 'fips'}).set_index('fips')
df_case = df_case.drop([
    'UID', 'iso2', 'iso3', 'code3',
    'Admin2', 'Province_State', 'Country_Region', 
    'Lat', 'Long_', 'Combined_Key'], axis=1)

df_case['2020-01-21'] = 0


df_death['FIPS'] = df_death['FIPS'].apply(lambda x: str(int(x)).zfill(5))
df_death = df_death.rename(columns={'FIPS': 'fips'}).set_index('fips')
df_death = df_death.drop([
    'UID', 'iso2', 'iso3', 'code3',
    'Admin2', 'Province_State', 'Country_Region', 
    'Lat', 'Long_', 'Combined_Key'], axis=1)

df_death['2020-01-21'] = 0


cols = ['2020-01-21'] + dates

df_case_step = df_case[cols]    .rename(columns={col: f'case{i - 1}' for i, col in enumerate(cols)})    .diff(axis=1).dropna(axis=1)
df_death_step = df_death[cols]    .rename(columns={col: f'death{i - 1}' for i, col in enumerate(cols)})    .diff(axis=1).dropna(axis=1)

df_covid = pd.concat([df_case_step, df_death_step], axis=1)
df_covid[df_covid < 0] = 0
df_covid['total_case'] = df_case[dates_case[-1]]
df_covid['total_death'] = df_death[dates_death[-1]]

with open(f'{folder}/steps.dat', 'w') as handle:
    handle.write(' '.join(cols[1:]))

existing_data_covid_csv = glob(f'{folder}/data_covid_????-??-??.csv')

if len(existing_data_covid_csv) > 0:
    for csv in existing_data_covid_csv:
        check_output(f'rm {csv}', shell=True)
        
df_covid.to_csv(f'{folder}/data_covid_{dates_case[-1]}.csv')


df_non_covid = pd.read_csv(f'{folder}/data_non-covid.csv', dtype={'fips': str}).set_index('fips')
df = df_non_covid.join(df_covid)

existing_data_csv = glob(f'{folder}/data_????-??-??.csv')
if len(existing_data_csv) > 0:
    for csv in existing_data_csv:
        check_output(f'rm {csv}', shell=True)  

df.to_csv(f'{folder}/data_{dates_case[-1]}.csv')

