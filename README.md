# Linear-regression-Decision-Tree-Random-Forest-Regression-on-Housing-Data.
This is a machine learning project which implements three different types of regression techniques and formulates differences amongst them by predicting the price of a house based on Boston housing Data.
# Sklearn joblib numpy pandas SimpleImputer RandomForestRegressor

Scikit-klearn is a Python library for dealing with machine learning problems.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sklearn and others.

```bash
pip install sklearn numpy pandas 

## About Dataset
You can download the Boston House prices dataset from the site: https://www.kaggle.com/vikrishnan/boston-house-prices
Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository1): CRIM: per capita crime rate by town

ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: nitric oxides concentration (parts per 10 million)
1https://archive.ics.uci.edu/ml/datasets/Housing
123
20.2. Load the Dataset 124
RM: average number of rooms per dwelling
AGE: proportion of owner-occupied units built prior to 1940
DIS: weighted distances to ﬁve Boston employment centers
RAD: index of accessibility to radial highways
TAX: full-value property-tax rate per $10,000
PTRATIO: pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
We can see that the input attributes have a mixture of units.

## Usage

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
