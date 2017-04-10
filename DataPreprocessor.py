from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (Imputer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Importing Data
dataset = pd.read_csv('Data.csv')

# Removing some columns which aren't useful for our calculation
df = dataset.drop([col for col in ['movie_title', 'color', 'plot_keywords', 'movie_imdb_link',
                                   'aspect_ratio', 'genres','facenumber_in_poster']
                   if col in dataset], axis=1)

#get the positions of the columns which are strings
director_name_pos = df.columns.get_loc("director_name")
language_pos = df.columns.get_loc("language")
country_pos = df.columns.get_loc("country")
content_rating_pos = df.columns.get_loc("content_rating")
actor_1_name_pos = df.columns.get_loc("actor_1_name")
actor_2_name_pos = df.columns.get_loc("actor_2_name")
actor_3_name_pos = df.columns.get_loc("actor_3_name")


#create a exclude list of these excluded attributes
categorical_fts = []
categorical_fts.append(director_name_pos)
categorical_fts.append(language_pos)
categorical_fts.append(country_pos)
categorical_fts.append(content_rating_pos)
categorical_fts.append(actor_1_name_pos)
categorical_fts.append(actor_2_name_pos)
categorical_fts.append(actor_3_name_pos)

#Array of features, exludes the last column
X = df.iloc[:, :-1].values

# Last column array, string of length 6 (dtype)
Ystr = np.asarray(df.iloc[:, df.shape[1]-1], dtype="|S6") #numpy is moody
Y = Ystr.astype(np.float)

label_director = LabelEncoder()
X[0:, director_name_pos] = label_director.fit_transform(X[0:, director_name_pos])
label_language = LabelEncoder()
X[0:, language_pos] = label_language.fit_transform(X[0:, language_pos])
label_country = LabelEncoder()
X[0:, country_pos] = label_country.fit_transform(X[0:, country_pos])
label_content_rating = LabelEncoder()
X[0:, content_rating_pos] = label_content_rating.fit_transform(X[0:, content_rating_pos])
label_actor_1_name = LabelEncoder()
X[0:, actor_1_name_pos] = label_actor_1_name.fit_transform(X[0:, actor_1_name_pos])
label_actor_2_name = LabelEncoder()
X[0:, actor_2_name_pos] = label_actor_2_name.fit_transform(X[0:, actor_2_name_pos])
label_actor_3_name = LabelEncoder()
X[0:, actor_3_name_pos] = label_actor_3_name.fit_transform(X[0:, actor_3_name_pos])

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

#Entire dataset is ready
#Performing K Fold cross validation
kfold = KFold(n_splits=5, random_state=7)
model = LinearSVC()
results = cross_val_score(model, X, Ystr, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))