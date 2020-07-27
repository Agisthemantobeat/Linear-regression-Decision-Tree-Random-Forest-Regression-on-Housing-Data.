# Linear-regression-Decision-Tree-Random-Forest-Regression-on-Housing-Data.
This is a machine learning project which implements three different types of regression techniques and formulates differences amongst them by predicting the price of a house based on Boston housing Data.
# Sklearn joblib numpy pandas SimpleImputer RandomForestRegressor

Scikit-klearn is a Python library for dealing with machine learning problems.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sklearn and others.

```bash
pip install sklearn numpy pandas 
```

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
