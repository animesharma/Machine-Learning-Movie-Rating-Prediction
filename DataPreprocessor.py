from __future__ import absolute_import, print_function, division
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Importing Data
dataset = pd.read_csv('Data.csv')

# Removal of columns
df = dataset.drop([col for col in ['movie_title', 'color', 'director_name', 'actor_2_name', 'actor_1_name', 'budget',
                                   'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'content_rating',
                                   'title_year', 'aspect_ratio', 'genres', 'num_critic_for_reviews', 'num_voted_users',
                                   'cast_total_facebook_likes', 'num_user_for_reviews', 'movie_facebook_likes']
                   if col in dataset], axis=1)

X = df.iloc[:, :-1].values  # except last col
# Used in asarray Y = df.iloc[:, -1:].values  # Last column array
Y = np.asarray(df.iloc[:, -1:].values, dtype="|S6") #numpy is moody

#Categorical Data Encoding
l_lang = LabelEncoder()
X[0:,6] = l_lang.fit_transform(X[0:,6])
l_count = LabelEncoder()
X[0:,7] = l_count.fit_transform(X[0:,7])

# Missing Values
# df = df.replace(np.nan, ' ', regex=True)  # Only works on dataframe object not on ndarray.
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

# Dummy Variable for CD
o_h1 = OneHotEncoder(categorical_features=[6,7])
X = o_h1.fit_transform(X).toarray()
X = X[:,1:] #To account for Dummy Variable Trap; Removed First Column

# Splitting of Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Feature Scaling
# stdsc = StandardScaler()
# X_train = stdsc.fit_transform(X_train)
# X_test = stdsc.fit_transform(X_test)


# #t distribution stochastic neighbor embedding (t-SNE) visualization
# tsne = TSNE(n_components=2, random_state=0)
# x_2d = tsne.fit_transform(X)
# x_train_2d = tsne.fit_transform(X_train)
# x_test_2d = tsne.fit_transform(X_test)




