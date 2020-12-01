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

from matplotlib import pyplot

import xgboost as xgb

from sklearn import svm

df = pd.read_csv("Data/Train_v2.csv")
df = df.drop(["uniqueid"], axis=1)


# # Vector as cell value
# X_categorical = df.select_dtypes(include=[object])
# enc = OneHotEncoder(handle_unknown='ignore')
# for column in X_categorical.columns:
#     temp_df = pd.DataFrame(enc.fit_transform(X_categorical[[column]]).toarray())
#     X_categorical[column] = temp_df.to_numpy().tolist()


# Dataset balancing
number_of_Yes = df.groupby(["bank_account"])["bank_account"].count()["Yes"]
df_No_account = df[df["bank_account"] == "No"]
df_Yes_account = df[df["bank_account"] == "Yes"]
df_No_account_Sample = df_No_account.sample(number_of_Yes)
df_Balanced = pd.concat([df_Yes_account, df_No_account_Sample], ignore_index=True)


le = LabelEncoder()
y = le.fit_transform(df_Balanced["bank_account"])

X = df_Balanced.drop(["bank_account"], axis=1)
X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Best features selection
# selector = SelectKBest(chi2, k=10)
# selector.fit_transform(X_train, y_train)
# cols = selector.get_support(indices=True)
# best_X_train = X_train.iloc[:,cols]


# # Decision tree classification
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))


def KBestTreeClassificationLoop(X, y):
    row_list = []
    for i in range(1, len(X.columns)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        selector = SelectKBest(chi2, k=i)
        selector.fit_transform(X_train, y_train)
        cols = selector.get_support(indices=True)
        best_X_train = X_train.iloc[:,cols]
        best_X_test = X_test.iloc[:,cols]

        clf = tree.DecisionTreeClassifier()
        clf.fit(best_X_train, y_train)
        y_pred = clf.predict(best_X_test)

        temp_dict = {"nombre_de_features": i, 
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
        }

        print(confusion_matrix(y_test, y_pred))

        row_list.append(temp_dict)
        temp_df = pd.DataFrame(row_list)
        temp_df.to_csv("training_analysis/TreeCLassificationFeaturesNumberBalanced2.csv")
# KBestTreeClassificationLoop(X, y)


# # XGboost classifier
# model=xgb.XGBClassifier(learning_rate=0.01, max_depth=20)
# model.fit(X_train, y_train)
# # plot
# # xgb.plot_importance(model)
# # pyplot.show()
# y_pred = model.predict(X_test)
# print(accuracy_score(y_test, y_pred))
# print(recall_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# # all features with their imortance score
# zipped = list(zip(X_train.columns, model.feature_importances_))
# zipped = sorted(zipped, key = lambda tup: tup[1], reverse=True)
# # only valuable features
# zipped2 = list(filter(lambda tup: tup[1] > 0, zipped))


# #XGBoost number features loop
# row_list = []
# for i in range(0, len(zipped2)):
#     x_temp = X_train[[t[0] for t in zipped2][0: i + 1]]
#     print(x_temp)
#     model.fit(x_temp, y_train)
#     x_temp_test = X_test[[t[0] for t in zipped2][0: i + 1]]
#     print(model.score(x_temp_test,y_test))
#     y_temp_pred = model.predict(x_temp_test)
#     temp_dict = {"nombre_de_features": i+1, 
#         "accuracy": accuracy_score(y_test, y_temp_pred),
#         "precision": precision_score(y_test, y_temp_pred),
#         "recall": recall_score(y_test, y_temp_pred),
#         "f1_score": f1_score(y_test, y_temp_pred)
#         }
#     row_list.append(temp_dict)
# df_xgboost_features_result = pd.DataFrame(row_list)
# df_xgboost_features_result.to_csv("training_analysis/XGBoostFeaturesNumberBalanced.csv")

