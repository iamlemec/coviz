import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
from cycler import cycler
from datetime import timedelta

import data_tools as dt
import plot_tools as pt

idx = pd.IndexSlice

start = 1e-6
scale = 1e6

##
## configure
##

pal = ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]
mpl.rcParams['axes.prop_cycle'] = cycler(color=pal)

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
    return dt.load_country()[[
        'cases_pc', 'cases_cum_pc', 'deaths_pc', 'deaths_cum_pc', 'vax_pc', 'vax_cum_pc'
    ]]

@st.cache()
def load_state():
    return dt.load_state()[[
        'cases_pc', 'cases_cum_pc', 'deaths_pc', 'deaths_cum_pc', 'vax_pc', 'vax_cum_pc'
    ]]

@st.cache()
def load_county():
    return dt.load_county()[[
        'cases_pc', 'cases_cum_pc', 'deaths_pc', 'deaths_cum_pc'
    ]]

##
## display
##

# load all data
data_country = load_country()
data_state = load_state()
data_county = load_county()

# get full date range
date_min = min(
    data_country.index.min(), data_state.index.min(), data_county.index.min()
)
date_max = max(
    data_country.index.max(), data_state.index.max(), data_county.index.max()
)
date_range = pd.date_range(date_min, date_max, freq='d')
date_start = pd.to_datetime('2021-06-01')

# options
list_country = data_country.columns.levels[1].tolist()
list_state = data_state.columns.levels[1].tolist()
list_county = data_county.columns.levels[1].tolist()

# unit selection
country = st.sidebar.multiselect('Country', list_country, default=['USA', 'ISR', 'TWN'])
state = st.sidebar.multiselect('State', list_state, default=['PA', 'CA'])
county = st.sidebar.multiselect('County', list_county, default=['Allegheny, PA', 'San Francisco, CA'])

# display options
st.sidebar.title('Options')
log = st.sidebar.checkbox('Log Scale', False)
cumul = st.sidebar.checkbox('Cumulative', False)
smooth = st.sidebar.number_input('Smoothing', min_value=1, value=7)
dsel_min, dsel_max = st.sidebar.select_slider(
    'Date Range', options=date_range, value=(date_start, date_max),
    format_func=lambda d: d.strftime('%Y-%m-%d')
)
dsel_min = max(date_min, dsel_min - timedelta(days=smooth)) # account for smoothin delay

# aggregate selections
sel_country = data_country.loc[:, idx[:, country]].dropna(how='all')
sel_state = data_state.loc[:, idx[:, state]].dropna(how='all')
sel_county = data_county.loc[:, idx[:, county]].dropna(how='all')
sel = pd.concat([sel_country, sel_state, sel_county], axis=1).loc[dsel_min:dsel_max]

# cases
st.subheader('Cases (per million people)')
col_c = 'cases_cum_pc' if cumul else 'cases_pc'
fig_c, ax_c = pt.plot_progress(data=sel[col_c], log=log, smooth=smooth, per=1e6)
ax_c.set_xlabel('')
ax_c.set_ylabel('')
st.pyplot(fig_c, clear=True, bbox_inches='tight')

# deaths
st.subheader('Deaths (per 10 million people)')
col_d = 'deaths_cum_pc' if cumul else 'deaths_pc'
fig_d, ax_d = pt.plot_progress(data=sel[col_d], log=log, smooth=smooth, per=1e7)
ax_d.set_xlabel('')
ax_d.set_ylabel('')
st.pyplot(fig_d, clear=True, bbox_inches='tight')

# vaccinations
st.subheader('Percent vaccinated (total doses / 2)')
col_v = 'vax_cum_pc' if cumul else 'vax_pc'
fig_v, ax_v = pt.plot_progress(data=sel[col_v], log=log, smooth=smooth, per=1e2)
ax_v.set_xlabel('')
ax_v.set_ylabel('')
st.pyplot(fig_v, clear=True, bbox_inches='tight')
