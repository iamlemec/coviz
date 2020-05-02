import streamlit as st
import pandas as pd
import numpy as np

import data_tools as dt
import plot_tools as pt

idx = pd.IndexSlice

start = 1e-6
scale = 1e6

##
## tools
##

def log10(x):
    x1 = np.log10(x)
    x1 = x1.replace({np.inf: np.nan, -np.inf: np.nan})
    return x1

##
## loaders
##

@st.cache()
def load_country():
    country = dt.load_country().unstack(level='country_code')[['cases_pc', 'deaths_pc']]
    align = scale*pt.get_aligned(country, start)
    return align

@st.cache()
def load_state():
    state = dt.load_state().unstack(level='abbrev')[['cases_pc', 'deaths_pc']]
    align = scale*pt.get_aligned(state, start)
    return align

@st.cache()
def load_fips():
    fips = pd.read_csv('../data/pop/county-populations.csv', dtype={'county_fips': 'str'})
    return fips

##
## display
##

data_country = load_country()
data_state = load_state()

# options
list_country = data_country.columns.levels[1].tolist()
list_state = data_state.columns.levels[1].tolist()

# sidebar
country = st.sidebar.multiselect('Country', list_country, default=['USA'])
state = st.sidebar.multiselect('State', list_state, default=['PA'])
st.sidebar.title('Options')
log = st.sidebar.checkbox('Log Scale', False)
cum = st.sidebar.checkbox('Cumulative', True)

# aggregate selections
sel_country = data_country.loc[:, idx[:, country]].dropna()
sel_state = data_state.loc[:, idx[:, state]].dropna()
sel = pd.concat([sel_country, sel_state], axis=1)
print(sel_country)

# transforms
if not cum:
    sel = sel.diff(axis=0)
if log:
    sel = log10(sel)

# cases
st.subheader('Cases (per million people)')
st.line_chart(sel['cases_pc'])

# deaths
st.subheader('Deaths (per million people)')
st.line_chart(sel['deaths_pc'])

##
## county fips
##

fips = load_fips()

st.title('County Statistics')
st.table(fips)
