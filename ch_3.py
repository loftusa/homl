#%%
import numpy as np
from sklearn.datasets import fetch_openml

# %%
# load MNIST
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]
X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]
y_train_5 = y_train == "5"
y_test_5 = y_test == "5"
#%%
# train binary classifier
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
model.fit(X_train, y_train_5)
#%%
# get cross-validation score
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train, y_train_5, cv=3, scoring="accuracy")

#%%
# get confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(model, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)

#%%
# get precision and recall score
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)


# get F_1 score
f1_score(y_train_5, y_train_pred)
# %%
# plot roc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_train_pred)
plt.plot(fpr, tpr, linewidth=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
roc_auc_score(y_train_5, y_train_pred)
#%%
# AUC for RF classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
y_pred_rf = cross_val_predict(rf, X_train, y_train_5, cv=3, method="predict_proba")
#%%
y_pred_rf = y_pred_rf[:, 1]
fpr_f, tpr_f, threshold_f = roc_curve(y_train_5, y_pred_rf)
plt.plot(fpr, tpr, "b:", label="SGD")
plt.plot(fpr_f, tpr_f, label="RF")
plt.legend(loc="lower right")

# %%
