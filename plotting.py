import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import GridSearchCV

from sklearn import tree

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt

import xgboost as xgb


df_tree = pd.read_csv("training_analysis/TreeClassificationFeaturesNumber.csv")
df_tree_balanced = pd.read_csv("training_analysis/TreeCLassificationFeaturesNumberBalanced.csv")
df_xgboost = pd.read_csv("training_analysis/XGBoostFeaturesNumber.csv")
df_xgboost_balanced = pd.read_csv("training_analysis/XGBoostFeaturesNumberBalanced.csv")

print(df_tree)


# plt.plot(df_tree[["accuracy", "precision", "recall", "f1_score"]])
# plt.show()

# ax = plt.gca()

# df_tree.plot(kind='line',x="nombre_de_features",y="accuracy", ax=ax)
# df_tree.plot(kind='line',x="nombre_de_features",y='precision', ax=ax)
# df_tree.plot(kind='line',x="nombre_de_features",y='recall', ax=ax)
# df_tree.plot(kind='line',x="nombre_de_features",y='f1_score', ax=ax)
# df_tree_balanced.plot(kind='line',x="nombre_de_features",y="accuracy", ax=ax)
# df_tree_balanced.plot(kind='line',x="nombre_de_features",y='precision', ax=ax)
# df_tree_balanced.plot(kind='line',x="nombre_de_features",y='recall', ax=ax)
# df_tree_balanced.plot(kind='line',x="nombre_de_features",y='f1_score', ax=ax)


plt.figure()

plt.subplot(211)
ax = plt.gca()
df_xgboost.plot(kind='line',x="nombre_de_features",y="accuracy", ax=ax)
df_xgboost.plot(kind='line',x="nombre_de_features",y='precision', ax=ax)
df_xgboost.plot(kind='line',x="nombre_de_features",y='recall', ax=ax)
df_xgboost.plot(kind='line',x="nombre_de_features",y='f1_score', ax=ax)
plt.title("Unbalanced VS Balanced Data XGBoost Score")

plt.subplot(212)
ax2 = plt.gca()
df_xgboost_balanced.plot(kind='line',x="nombre_de_features",y="accuracy", ax=ax2)
df_xgboost_balanced.plot(kind='line',x="nombre_de_features",y='precision', ax=ax2)
df_xgboost_balanced.plot(kind='line',x="nombre_de_features",y='recall', ax=ax2)
df_xgboost_balanced.plot(kind='line',x="nombre_de_features",y='f1_score', ax=ax2)
plt.show()

