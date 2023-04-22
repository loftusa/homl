#%%
# make a perceptron

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import torch

iris = load_iris(as_frame=True)

X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target == 0

model = Perceptron()
model.fit(X, y)
model.predict(torch.tensor([[2, 1]]))

# do it with torch
X = torch.tensor(X[:3, :])
y = torch.tensor(y[:3])

#%%
# california housing MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# make a model with three hidden layers, each with 50 neurons in it
# StandardScale the data first, then run it through the regressor
hidden_layers = [50, 50, 50]
model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hidden_layers))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse
#%%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score

iris = load_iris(as_frame=True)

X = iris.data.values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
hidden_layers = 10

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=42, max_iter=1000),
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(precision_score(y_test, y_pred, average="micro"))
print(recall_score(y_test, y_pred, average="micro"))
#%%

import tensorflow as tf

tf.random.set_seed(42)

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


tf.random.set_seed(42)

# build a classification NLP with two hidden layers
# Flatten the input, then two dense layers (300 neurons and 100) with relu activation
# then a final softmax layer with 10 classes
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


model.compile(
    loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
)

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
#%%
import pandas as pd

df = pd.DataFrame(history.history)

df.plot(
    figsize=(8, 5),
    xlim=[0, 29],
    ylim=[0, 1],
    grid=True,
    xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"],
)

#%%
y_proba = model.predict(X_test[:3])
# %%
y_proba.round(3)
# %%
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# make a normalizeation layer,
# then a nn with 3 dense layers and a dense output layer,
# an adam optimizer with lr=1e-3,
# mse loss, rmse metric,
# fit with 20 epocs, evaluate on testing data
norm = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential(
    [
        norm,
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['RootMeanSquaredError'])
norm.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
evaluation = model.evaluate(X_test, y_test)


#%%
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)