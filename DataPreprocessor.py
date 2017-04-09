from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (Imputer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


# Importing Data
dataset = pd.read_csv('Data.csv')

# Removing some columns which aren't useful for our calculation
df = dataset.drop([col for col in ['movie_title', 'color',  'actor_2_name', 'actor_1_name',
                                   'actor_3_name', 'plot_keywords', 'movie_imdb_link',
                                   'aspect_ratio', 'genres','facenumber_in_poster']
                   if col in dataset], axis=1)

#get the positions of the columns which are strings
director_name_pos = df.columns.get_loc("director_name")
language_pos = df.columns.get_loc("language")
country_pos = df.columns.get_loc("country")
content_rating_pos = df.columns.get_loc("content_rating")

#create a exclude list of these excluded attributes
categorical_fts = []
categorical_fts.append(director_name_pos)
categorical_fts.append(language_pos)
categorical_fts.append(country_pos)
categorical_fts.append(content_rating_pos)

#Array of features, exludes the last column
X = df.iloc[:, :-1].values

# Last column array, string of length 6 (dtype)
Y = np.asarray(df.iloc[:, -1:].values, dtype="|S6") #numpy is moody

label_director = LabelEncoder()
X[0:, director_name_pos] = label_director.fit_transform(X[0:, director_name_pos])
label_language = LabelEncoder()
X[0:, language_pos] = label_language.fit_transform(X[0:, language_pos])
label_country = LabelEncoder()
X[0:, country_pos] = label_country.fit_transform(X[0:, country_pos])
label_content_rating = LabelEncoder()
X[0:, content_rating_pos] = label_content_rating.fit_transform(X[0:, content_rating_pos])

# Missing Values
# df = df.replace(np.nan, ' ', regex=True)  # Only works on dataframe object not on ndarray.
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

# Dummy Variable for CD
o_h1 = OneHotEncoder(categorical_features=categorical_fts)
X = o_h1.fit_transform(X).toarray()
X = X[:,1:]
#To account for Dummy Variable Trap; Removed First Column
#TODO redo it with 0 considered

# Splitting of Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.16, random_state = 0)

# Feature Scaling
#stdsc = StandardScaler()
#X_train = stdsc.fit_transform(X_train)
#X_test = stdsc.fit_transform(X_test)


# #t distribution stochastic neighbor embedding (t-SNE) visualization
# tsne = TSNE(n_components=2, random_state=0)
# x_2d = tsne.fit_transform(X)
# x_train_2d = tsne.fit_transform(X_train)
# x_test_2d = tsne.fit_transform(X_test)