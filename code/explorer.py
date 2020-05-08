import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
from cycler import cycler

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
    country = dt.load_country().unstack(level='country_code')[['cases_pc', 'deaths_pc']]
    align = pt.get_aligned(country, start, 'cases_pc')
    return align

@st.cache()
def load_state():
    state = dt.load_state().unstack(level='abbrev')[['cases_pc', 'deaths_pc']]
    align = pt.get_aligned(state, start, 'cases_pc')
    return align

@st.cache()
def load_county():
    county = dt.load_county().unstack(level='full_name')[['cases_pc', 'deaths_pc']]
    align = pt.get_aligned(county, start, 'cases_pc')
    return align

##
## display
##

data_country = load_country()
data_state = load_state()
data_county = load_county()

# options
list_country = data_country.columns.levels[1].tolist()
list_state = data_state.columns.levels[1].tolist()
list_county = data_county.columns.levels[1].tolist()

# sidebar
country = st.sidebar.multiselect('Country', list_country, default=['USA', 'FRA'])
state = st.sidebar.multiselect('State', list_state, default=['PA', 'CA'])
county = st.sidebar.multiselect('County', list_county, default=['Allegheny, PA', 'King, WA'])
st.sidebar.title('Options')
log = st.sidebar.checkbox('Log Scale', False)
cum = st.sidebar.checkbox('Cumulative', False)
smooth = st.sidebar.number_input('Smoothing', min_value=1, value=7)

# aggregate selections
sel_country = data_country.loc[:, idx[:, country]].dropna(how='all')
sel_state = data_state.loc[:, idx[:, state]].dropna(how='all')
sel_county = data_county.loc[:, idx[:, county]].dropna(how='all')
sel = pd.concat([sel_country, sel_state, sel_county], axis=1)

# cases
st.subheader('Cases (per million people)')
fig_c, ax_c = pt.plot_progress(data=sel['cases_pc'], log=log, cum=cum, smooth=smooth)
ax_c.set_xlabel('Days since 1 case per million people')
ax_c.set_ylabel('Cases per million people')
st.pyplot(fig_c, clear=True, bbox_inches='tight')

# deaths
st.subheader('Deaths (per million people)')
fig_d, ax_d = pt.plot_progress(data=sel['deaths_pc'], log=log, cum=cum, smooth=smooth)
ax_d.set_xlabel('Days since 1 case per million people')
ax_d.set_ylabel('Deaths per million people')
st.pyplot(fig_d, clear=True, bbox_inches='tight')
