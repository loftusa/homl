#%% [markdown]
# # Regularized Linear Models


# %%
import numpy as np
from sklearn.linear_model import Ridge

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

ridge_reg = Ridge(alpha=0.1 / m, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

#%% [markdown]

# cross entropy
# $i_1$