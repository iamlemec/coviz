import numpy as np
import pandas as pd

datadir = '../data'

##
## countries
##

# source:
# HDX: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases

def load_jhu(path, name):
    iso_info = pd.read_csv(f'{datadir}/meta/country_info.csv', dtype={'population': 'Int64'})
    iso_map = iso_info.set_index(['country', 'region'])['country_code']

    iso_pop = iso_info.set_index('country_code')['population']
    iso_pop = iso_pop.groupby('country_code').sum()

    # merge iso and aggregate
    df = pd.read_csv(path)
    df = df.drop(['Lat', 'Long'], axis=1)
    df = df.rename({'Country/Region': 'country', 'Province/State': 'region'}, axis=1)
    df = df.join(iso_map, on=['country', 'region'])
    df = df.drop(['country', 'region'], axis=1)
    df = df.set_index('country_code')
    df = df.groupby('country_code').sum()

    # transpose and fix dates
    df = df.T
    df.index = pd.to_datetime(df.index)
    
    # find per capita rates
    df_pc = df.div(iso_pop, axis=1)

    # make full frame
    return pd.concat({
        f'{name}_cum': df,
        f'{name}_cum_pc': df_pc,
        f'{name}': df.diff(),
        f'{name}_pc': df_pc.diff(),
    }, axis=1)
    
def load_country():
    jhu_dir = f'{datadir}/jhu/csse_covid_19_data/csse_covid_19_time_series'
    return pd.concat([
        load_jhu(f'{jhu_dir}/time_series_covid19_confirmed_global.csv', 'cases'),
        load_jhu(f'{jhu_dir}/time_series_covid19_deaths_global.csv', 'deaths'),
    ], axis=1)

##
## states
##

# source:
# https://github.com/nytimes/covid-19-data

state_cols = [
    'fips',
    'date',
    'cases',
    'deaths'
]

state_names = {
    'cases': 'cases_cum',
    'deaths': 'deaths_cum'
}

state_dtypes = {
    'cases': 'Int64',
    'deaths': 'Int64',
    'fips': 'str'
}

def load_state():
    df_pop_state = pd.read_csv(
        f'{datadir}/pop/state-populations.csv',
        usecols=['fips', 'abbrev', '2018'],
        dtype={'fips': 'str', '2018': 'Int64'}
    )
    df_pop_state = df_pop_state.set_index('fips').rename(columns={'2018': 'pop'})

    df_state = pd.read_csv(f'{datadir}/nyt/us-states.csv', usecols=state_cols, dtype=state_dtypes, parse_dates=['date'])
    df_state = df_state.join(df_pop_state[['abbrev', 'pop']], on='fips').drop(columns='fips')
    df_state = df_state.rename(columns=state_names)
    df_state = df_state.dropna(subset=['abbrev', 'date'])
    df_state = df_state.set_index(['abbrev', 'date']).sort_index().astype(np.float)
    df_state[['cases', 'deaths']] = df_state.groupby(level='abbrev')[['cases_cum', 'deaths_cum']].diff()
    df_state[['cases_pc', 'deaths_pc']] = df_state[['cases', 'deaths']].div(df_state['pop'], axis=0)
    df_state[['cases_cum_pc', 'deaths_cum_pc']] = df_state[['cases_cum', 'deaths_cum']].div(df_state['pop'], axis=0)
    df_state = df_state.unstack(level='abbrev').fillna(0)
    return df_state

##
## counties
##

# source:
# https://github.com/nytimes/covid-19-data

county_dtypes = {
    'cases': 'Int64',
    'deaths': 'Int64',
    'fips': 'str'
}

county_cols = [
    'fips',
    'date',
    'county',
    'cases',
    'deaths'
]

nyc_pop = pd.Series({
    'state_fips': '36',
    'state_code': 'NY',
    'county_name': 'New York City',
    'pop': 8398748
}, name='NYC').to_frame().T

def load_county_stats():
    df_pop_county = pd.read_csv(
        f'{datadir}/pop/county-populations.csv',
        usecols=['state_fips', 'county_fips', 'county_name', 'state_code', 'pop18'],
        dtype={'state_fips': 'str', 'county_fips': 'str', 'pop18': 'Int64'}
    )
    df_pop_county = df_pop_county.set_index('county_fips').rename(columns={'pop18': 'pop'})
    df_pop_county = pd.concat([df_pop_county, nyc_pop])
    return df_pop_county

def load_county():
    df_pop_county = load_county_stats()
    df_county = pd.read_csv(f'{datadir}/nyt/us-counties.csv', usecols=county_cols, dtype=county_dtypes, parse_dates=['date'])
    df_county = df_county.rename(columns={'fips': 'county_fips'})
    df_county.loc[df_county['county']=='New York City', 'county_fips'] = 'NYC'
    df_county = df_county.join(df_pop_county[['state_code', 'pop']], on='county_fips')
    df_county = df_county.rename(columns={'cases': 'cases_cum', 'deaths': 'deaths_cum'})
    df_county['full_name'] = df_county['county'] + ', ' + df_county['state_code']
    df_county = df_county.drop(['county', 'state_code', 'county_fips'], axis=1)
    df_county = df_county.dropna(subset=['full_name', 'date'])
    df_county = df_county.set_index(['full_name', 'date']).sort_index().astype(np.float)
    df_county[['cases', 'deaths']] = df_county.groupby(level='full_name')[['cases_cum', 'deaths_cum']].diff()
    df_county[['cases_pc', 'deaths_pc']] = df_county[['cases', 'deaths']].div(df_county['pop'], axis=0)
    df_county[['cases_cum_pc', 'deaths_cum_pc']] = df_county[['cases_cum', 'deaths_cum']].div(df_county['pop'], axis=0)
    df_county = df_county.unstack(level='full_name').fillna(0)
    return df_county
