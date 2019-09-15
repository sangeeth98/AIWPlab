#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:05:03 2019

@author: astro
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# %%
liver_df = pd.read_csv('indian_liver_patient.csv')
liver_df.head()


# %%
liver_df.info()

# %%
liver_df.describe(include='all')

# %%
liver_df.isnull().sum()

# %% Data visualization
sns.countplot(data=liver_df, x='Dataset', label='Count')
LD, NLD = liver_df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ', LD)
print('Number of patients not diagnosed with liver disease: ', NLD)

# %%
sns.countplot(data=liver_df, x='Gender', label='Count')
M, F = liver_df['Gender'].value_counts()
print('Number of patients that are male: ', M)
print('Number of patients that are female: ', F)

# %%
sns.factorplot(x="Age", y="Gender", hue="Dataset", data=liver_df)

# %%
liver_df[['Gender', 'Dataset', 'Age']]\
    .groupby(['Dataset', 'Gender'], as_index=False)\
    .count()\
    .sort_values(by='Dataset', ascending=False)

# %%
liver_df[['Gender', 'Dataset', 'Age']]\
    .groupby(['Dataset', 'Gender'], as_index=False)\
    .mean()\
    .sort_values(by='Dataset', ascending=False)

# %%
g = sns.FacetGrid(liver_df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age')

# %%
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter, "Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)

# %%
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=liver_df, kind="reg")

# %%
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter, "Aspartate_Aminotransferase", "Alamine_Aminotransferase",
      edgecolor="w")
plt.subplots_adjust(top=0.9)

# %%
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase",
              data=liver_df, kind="reg")

# %%
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter, "Alkaline_Phosphotase", "Alamine_Aminotransferase",
      edgecolor="w")
plt.subplots_adjust(top=0.9)

# %%
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase",
              data=liver_df, kind="reg")

# %%
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter, "Total_Protiens", "Albumin", edgecolor="w")
plt.subplots_adjust(top=0.9)

# %%
sns.jointplot("Total_Protiens", "Albumin", data=liver_df, kind="reg")

# %%
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter, "Albumin", "Albumin_and_Globulin_Ratio", edgecolor="w")
plt.subplots_adjust(top=0.9)

# %%
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin",
              data=liver_df, kind="reg")

# %%
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter, "Albumin_and_Globulin_Ratio", "Total_Protiens",
      edgecolor="w")
plt.subplots_adjust(top=0.9)
"""
Hence from the aforementioned information we will only use the
following features:
    Total_Bilirubin
    Alamine_Aminotransferase
    Total_Protiens
    Albumin_and_Globulin_Ratio
    Albumin
    Age
    Gender
    Dataset
"""
# %%
liver_df = pd.concat([liver_df, pd.get_dummies(
        liver_df['Gender'], prefix='Gender')], axis=1)
liver_df.head()

# %%
liver_df[liver_df['Albumin_and_Globulin_Ratio'].isnull()]
# the columns having null values
# %%
liver_df["Albumin_and_Globulin_Ratio"] = liver_df\
    .Albumin_and_Globulin_Ratio\
    .fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())

# %% Building ML MODEL
Droop_gender = liver_df.drop(labels=['Gender'], axis=1)
X = Droop_gender
y = liver_df['Dataset']

# %%
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)

# %%
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
# Predicting Output
rf_predicted = random_forest.predict(X_test)
random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)

print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test, rf_predicted))
print(confusion_matrix(y_test, rf_predicted))
print(classification_report(y_test, rf_predicted))

# %%
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
# Predict Output
log_predicted = logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
# Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test, log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test, log_predicted))
print('Classification Report: \n',
      classification_report(y_test, log_predicted))

# %%
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Score': [logreg_score, random_forest_score],
    'Test Score': [logreg_score_test, random_forest_score_test]})
models.sort_values(by='Test Score', ascending=False)
