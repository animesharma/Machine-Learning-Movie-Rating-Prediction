from DataPreprocessor import *
# Importing Libraries
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators=999, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train.ravel())


# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = Y_pred.reshape(-1,1)
Accuracy = accuracy_score(Y_test, Y_pred, normalize=False, sample_weight=None)

