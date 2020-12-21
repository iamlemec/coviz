# CoViz

Plotting tools for COVID-19

![country deaths](output/country_deaths.svg)

## Data

This uses comprehensive and continually updated data from the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University for country level data (https://github.com/CSSEGISandData/COVID-19).

The state and county level data comes from the New York Times GitHub repository (https://github.com/nytimes/covid-19-data), which is included as a submodule in this repository. To fetch that run:

``
git submodule init
git submodule update
``

To update these again at a later date, I usually just run:

``
git submodule foreach git pull origin master
``

## Code

The files `data_tools.py` and `plot_tools.py` provide the underlying functionality, and the main interface is through `visualize.ipynb`.
