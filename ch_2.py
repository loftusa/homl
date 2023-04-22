#%%
# Use California census data to build a model of housing prices in the state
# output: a prediction of district's median housing price
import os
import tarfile
import urllib
import pandas as pd
import geopandas
import contextily as ctx

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.impute import SimpleImputer


#%%
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = "data/housing.tgz"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# grab and extract to CSV
urllib.request.urlretrieve(HOUSING_URL, filename=HOUSING_PATH)
housing_tgz = tarfile.open(HOUSING_PATH)
housing_tgz.extractall(path="data")

# grab as df
df = pd.read_csv("data/housing.csv")
df.head()
#%%
states = geopandas.read_file("data/usa-states-census-2014.shp")
ax = states[states.NAME == "California"].plot(
    color="white", edgecolor="black", alpha=0.5, figsize=(10, 10)
)

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.longitude, df.latitude)
)
column = "median_house_value"
gdf.plot(ax=ax, column=column, cmap="OrRd", legend=True)
plt.title(f"{column} in California")

print(gdf.columns)
# gdf
#%%
df.describe()
df.hist(bins=50, figsize=(20, 15), grid=False, edgecolor="black")

# %%
train, test = train_test_split(df, test_size=0.2, random_state=42)
df["income_cat"] = pd.cut(
    df["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_idx]
    strat_test_set = df.loc[test_idx]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

#%%
housing = strat_train_set.copy()

# look at housing prices
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.1,
    s=housing["population"] / 100,
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    figsize=(10, 7),
)

#%%
# compute pearson's r
corr_matrix = housing.corr()
sns.heatmap(corr_matrix)
corr_matrix["housing_median_age"].sort_values(ascending=False)

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# take care of missing values with SimpleImputter
def impute(df, strategy="median"):
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(
        strategy="median",
    )
    df_num = df.drop(["ocean_proximity", "geometry", "income_cat"], axis=1)
    X = imp.fit_transform(df_num)
    return pd.DataFrame(X, columns=df_num.columns, index=df_num.index)


housing_tr = impute(housing)
housing_cat = housing[["ocean_proximity"]]
housing_num = housing.drop(["ocean_proximity", "geometry", "income_cat"], axis=1)
housing_num
# %%
# ordinal encoding on housing_cat
from sklearn.preprocessing import OrdinalEncoder

oc = OrdinalEncoder()
hc_enc = oc.fit_transform(housing_cat)
hc_enc

# one-hot encoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
ohe_enc = ohe.fit_transform(housing_cat)

#%%
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
pd.DataFrame(attr_adder.transform(housing.values))
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

housing_num_tr = num_pipeline.fit_transform(housing_num)
pd.DataFrame(housing_num_tr)
housing_num

#%%
# ColumnTransformer
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity", "income_cat"]

full_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)]
)
housing_prepped = full_pipeline.fit_transform(housing)
pd.DataFrame(housing_prepped)

#%%
# train a model
from sklearn.linear_model import LinearRegression

lg = LinearRegression()
lg.fit(housing_prepped, housing_labels)

d = housing.iloc[:5]
l = housing_labels.iloc[:5]
dp = full_pipeline.transform(d)

print(lg.predict(dp))
print(l)

from sklearn.metrics import mean_squared_error

housing_pred = lg.predict(housing_prepped)

np.sqrt(mean_squared_error(housing_labels, housing_pred))

#%%
# decision tree instead
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

dtr = RandomForestRegressor()
scores = cross_val_score(
    dtr, housing_prepped, housing_labels, scoring="neg_mean_squared_error", cv=10
)
#%%
s = np.sqrt(-scores)
print(np.mean(s))
print(np.std(s))

# Grid search
from sklearn.model_selection import GridSearchCV


# k-fold cross validation
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

dtr = RandomForestRegressor()
grid_search = GridSearchCV(
    dtr, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True
)
grid_search.fit(housing_prepped, housing_labels)
#%%
grid_search.best_params_
# %%
grid_search.best_estimator_
# %%
from sklearn.model_selection import RandomizedSearchCV

np.set_printoptions(suppress=True)
grid_search.best_estimator_.feature_importances_

# location and median income predict housing prices more than anything else
sort_orders = sorted(
    {
        k: grid_search.best_estimator_.feature_importances_[i]
        for i, k in enumerate(housing.columns)
    }.items(),
    key=lambda x: x[1],
    reverse=True,
)
for i in sort_orders:
    print(i[0], i[1])
# %%
