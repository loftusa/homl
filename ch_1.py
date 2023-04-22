#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#%%
# Load data
oecd_bli = pd.read_csv(
    "data/oecd_bli_2015.csv",
    thousands=",",
)
gdp_per_capita = pd.read_csv(
    "data/gdp_per_capita.csv",
    thousands=",",
    delimiter="\t",
    encoding="latin1",
    na_values="n/a",
)

# Prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(
        left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True
    )
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    # remove_indices = [0, 1, 6, 8, 33, 34, 35]
    # keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", "Life satisfaction"]]
    # return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[
    #     keep_indices
    # ]


country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
#%%
# Visualize
country_stats.head()
sns.scatterplot(
    data=country_stats,
    x="GDP per capita",
    y="Life satisfaction",
)

# Select linear model
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X, y)
X_ = np.linspace(0, 100000, num=1000)
regression = model.predict(X_[:, None])
regression = np.squeeze(regression)
sns.lineplot(X_, regression, color="black")
plt.title("Life satisfaction compared to GDP, each dot is a country")
#%%
