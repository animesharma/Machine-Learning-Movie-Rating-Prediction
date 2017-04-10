from DataPreprocessor import *
# Importing Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def add(x,y):
	temp = float(x) + y
	return str(temp)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators=99, criterion='entropy', random_state=0,
									class_weight='balanced', n_jobs=-1)
									
classifier.fit(X_train, Y_train.ravel())

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = Y_pred.reshape(-1,1)
z1 = np.vectorize(add)
ARRAYS_WITH_PREDS = []
for i in np.arange(-0.2,0.3,0.1):
	ARRAYS_WITH_PREDS.append(z1(Y_pred,i))
final_accuracy = 0
for preds in ARRAYS_WITH_PREDS:
	final_accuracy += accuracy_score(Y_test,preds,normalize=True,sample_weight=None)

print 'Accuracy = ',final_accuracy*100, '%'