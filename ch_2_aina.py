#%%
import sklearn
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# %%
import matplotlib.pyplot as plt
import matplotlib
X, y = mnist["data"], mnist["target"]

def plot_digit(data, target):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis("off")
    plt.title(f"digit {target}")

some_digit, some_target = X[0], y[0]
plot_digit(some_digit, y[0])
plt.show()
#%%
y[0]
# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# %%
# use cross_val_score to evaluate SGDClassifier

from sklearn.model_selection import cross_val_score

scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#%%
y_scores = sgd_clf.decision_function([some_digit])

y_scores
# %%
threshold = 0
y_some_digit_pred = (y_scores > threshold)

#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# %%
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

plt.legend(loc="center right", fontsize=16)
plt.xlabel("Threshold", fontsize=16)

# %%
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

# %%
