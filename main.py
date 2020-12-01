import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def sortOrdinalEducation(x):
    if x =="No formal education":
        return 0
    elif x == "Primary education":
        return 1
    elif x == "Secondary education":
        return 2
    elif x == "Tertiary education":
        return 3
    elif x == "Vocational/Specialised training":
        return 4
    else:
        return 5


def percentageBankAccountByColumnPlot(dF, column):
    columnNoAccount = dF[dF["bank_account"] == "No"][column].value_counts(sort=False).sort_index(axis=0)
    columnYesAccount = dF[dF["bank_account"] == "Yes"][column].value_counts(sort=False).sort_index(axis=0)
    
    columnClassName = columnYesAccount.index.tolist()

    totals = [i+j for i,j in zip(columnNoAccount, columnYesAccount)]
    columnNoAccount = [i / j * 100 for i,j in zip(columnNoAccount, totals)]
    columnYesAccount = [i / j * 100 for i,j in zip(columnYesAccount, totals)]

    r = np.arange(len(columnClassName))
    barWidth = 0.85
    plt.bar(r, columnYesAccount, color='blue', edgecolor='white', width=barWidth, label="Yes")
    plt.bar(r, columnNoAccount, bottom=columnYesAccount, color='red', edgecolor='white', width=barWidth, label="No")

    plt.xticks(r, columnClassName, rotation=10)
    plt.xlabel(column)

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, title="Bank_account")

df = pd.read_csv("Data/Train_v2.csv")



# df["education_level_ordinal"] = df["education_level"].map(sortOrdinalEducation)

# educationLevelNoAccount = df[df["bank_account"] == "No"]["education_level"].value_counts(sort=False).sort_index(axis=0)
# educationLevelYesAccount = df[df["bank_account"] == "Yes"]["education_level"].value_counts(sort=False).sort_index(axis=0)

# namesEducationLevel = educationLevelYesAccount.index.tolist()

# totals = [i+j for i,j in zip(educationLevelNoAccount, educationLevelYesAccount)]
# educationLevelNoAccount = [i / j * 100 for i,j in zip(educationLevelNoAccount, totals)]
# educationLevelYesAccount = [i / j * 100 for i,j in zip(educationLevelYesAccount, totals)]

# print(educationLevelNoAccount)
# print(educationLevelYesAccount)

# r = [1,0,2,3,4,5]

# barWidth = 0.85
# plt.bar(r, educationLevelYesAccount, color='blue', edgecolor='white', width=barWidth, label="Yes")
# plt.bar(r, educationLevelNoAccount, bottom=educationLevelYesAccount, color='red', edgecolor='white', width=barWidth, label="No")

# # Custom x axis
# plt.xticks(r, namesEducationLevel, rotation=10)
# plt.xlabel("Education Level")
 
# plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, title="Bank_account")
# # Show graphic
# plt.show()



categorical_features = df.select_dtypes(include=['object'])
categorical_features = categorical_features.drop(["uniqueid", "bank_account"], axis=1)

print(categorical_features.columns)

ROWS, COLS = 3, 3
# fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 18))
# row, col = 0, 0
# for i, categorical_feature in enumerate(categorical_features):
#     if col == COLS - 1:
#         row += 1
#     col = i % COLS
#     df[categorical_feature].value_counts().plot(kind='bar', ax=ax[row, col]).set_title(categorical_feature)
# plt.savefig("plots/mygraph1.png")  # replace by plt.show() if we are working on Jupyter notebook


fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 18))
row, col = 0, 0
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1:
        row += 1
    col = i % COLS

    columnNoAccount = df[df["bank_account"] == "No"][categorical_feature].value_counts(sort=False).sort_index(axis=0)
    columnYesAccount = df[df["bank_account"] == "Yes"][categorical_feature].value_counts(sort=False).sort_index(axis=0)
    
    columnClassName = columnYesAccount.index.tolist()

    totals = [i+j for i,j in zip(columnNoAccount, columnYesAccount)]
    columnNoAccount = [i / j * 100 for i,j in zip(columnNoAccount, totals)]
    columnYesAccount = [i / j * 100 for i,j in zip(columnYesAccount, totals)]

    r = np.arange(len(columnClassName))
    print(columnClassName)
    print(r)
    barWidth = 0.85
    ax[col, row].bar(columnClassName, columnYesAccount, color='blue', edgecolor='white', width=barWidth, label="Yes")
    ax[col, row].bar(columnClassName, columnNoAccount, bottom=columnYesAccount, color='red', edgecolor='white', width=barWidth, label="No")

    # ax[col, row].set_xticks(r, columnClassName)
    # ax[col, row].set_xlabel(categorical_feature)
    ax[col, row].set_title(categorical_feature)
    ax[col, row].set_xticklabels(columnClassName, rotation=20)

    # ax[col, row].set_legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, title="Bank_account")

plt.savefig("plots/mygraph2.png")  # replace by plt.show() if we are working on Jupyter notebook



# sns.set()

# ax = sns.countplot(x="bank_account", hue="gender_of_respondent", data=df)

# ax = sns.countplot(x="gender_of_respondent", hue="bank_account", data=df)

# plt.show()


# sns.catplot(x="education_level", hue ="bank_account", kind="count", data=df, estimator=lambda x: sum(x==0)*100.0/len(x))
# plt.show()
