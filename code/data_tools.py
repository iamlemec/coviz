import numpy as np
import pandas as pd

datadir = '../data'

##
## countries
##

# source:
# https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases

int_cols = [
    'countryterritoryCode',
    'countriesAndTerritories',
    'dateRep',
    'cases',
    'deaths',
    'popData2018'
]

int_names = {
    'countryterritoryCode': 'country_code',
    'countriesAndTerritories': 'country_name',
    'dateRep': 'date',
    'popData2018': 'pop'
}

int_dtypes = {
    'popData2018': 'Int64'
}

def load_country():
    df_int = pd.read_csv(f'{datadir}/eu/covid_country_data.csv', usecols=int_cols, dtype=int_dtypes, parse_dates=['dateRep'], dayfirst=True)
    df_int = df_int.rename(columns=int_names)
    df_int = df_int[['date', 'country_code', 'deaths', 'cases', 'pop']]
    df_int = df_int.dropna(subset=['date', 'country_code', 'pop']).fillna(0)
    df_int = df_int.set_index(['country_code', 'date']).sort_index()
    df_int[['cases_cum', 'deaths_cum']] = df_int.groupby('country_code')[['cases', 'deaths']].cumsum()
    df_int[['cases_pc', 'deaths_pc']] = df_int[['cases_cum', 'deaths_cum']].apply(lambda s: s/df_int['pop'])
    return df_int

##
## states
##

# source:
# https://github.com/nytimes/covid-19-data

usa_cols = [
    'fips',
    'date',
    'cases',
    'deaths'
]

usa_names = {
    'cases': 'cases_cum',
    'deaths': 'deaths_cum'
}

usa_dtypes = {
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

    df_usa = pd.read_csv(f'{datadir}/nyt/us-states.csv', usecols=usa_cols, dtype=usa_dtypes, parse_dates=['date'])
    df_usa = df_usa.join(df_pop_state[['abbrev', 'pop']], on='fips').drop(columns='fips')
    df_usa = df_usa.rename(columns=usa_names)
    df_usa = df_usa.dropna(subset=['abbrev', 'date']).set_index(['abbrev', 'date'])
    df_usa = df_usa.fillna(0)
    df_usa[['cases_pc', 'deaths_pc']] = df_usa[['cases_cum', 'deaths_cum']].apply(lambda s: s/df_usa['pop'])
    return df_usa

##
## counties
##

# source:
# https://github.com/nytimes/covid-19-data

cnt_dtypes = {
    'cases': 'Int64',
    'deaths': 'Int64',
    'fips': 'str'
}

cnt_cols = [
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

def load_county():
    df_pop_county = pd.read_csv(
        f'{datadir}/pop/county-populations.csv',
        usecols=['state_fips', 'county_fips', 'county_name', 'state_code', 'pop18'],
        dtype={'state_fips': 'str', 'county_fips': 'str', 'pop18': 'Int64'}
    )
    df_pop_county = df_pop_county.set_index('county_fips').rename(columns={'pop18': 'pop'})
    df_pop_county = pd.concat([df_pop_county, nyc_pop])

    df_cnt = pd.read_csv(f'{datadir}/nyt/us-counties.csv', usecols=cnt_cols, dtype=cnt_dtypes, parse_dates=['date'])
    df_cnt = df_cnt.rename(columns={'fips': 'county_fips'})
    df_cnt.loc[df_cnt['county']=='New York City', 'county_fips'] = 'NYC'
    df_cnt = df_cnt.join(df_pop_county[['state_code', 'pop']], on='county_fips')
    df_cnt = df_cnt.rename(columns={'cases': 'cases_cum', 'deaths': 'deaths_cum'})
    df_cnt['full_name'] = df_cnt['county'] + ', ' + df_cnt['state_code']
    df_cnt = df_cnt.dropna(subset=['full_name', 'date']).set_index(['full_name', 'date'])
    df_cnt = df_cnt.fillna(0)
    df_cnt[['cases_pc', 'deaths_pc']] = df_cnt[['cases_cum', 'deaths_cum']].apply(lambda s: s/df_cnt['pop'])
    return df_cnt
