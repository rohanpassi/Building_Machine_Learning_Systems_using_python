from __future__ import print_function
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier




def load_dataset():
	data = []
	labels = []
	with open('seeds.tsv') as ifile:
		for line in ifile:
			tokens = line.strip().split('\t')
			data.append([float(tk) for tk in tokens[:-1]])
			labels.append(tokens[-1])
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

features, labels = load_dataset()
classifier = KNeighborsClassifier(n_neighbors=4)

n = len(features)
correct = 0.0

# Leave one out
for ei in range(n):
	training = np.ones(n, bool)
	training[ei] = 0
	testing = ~training
	classifier.fit(features[training], labels[training])
	pred = classifier.predict(features[ei])
	correct += (pred == labels[ei])
print("Result of leave one out: {}".format(correct/n))


# K Fold cross validation without normalization
means = []
kf = KFold(n, n_folds=3, shuffle=True)
for training, testing in kf:
	classifier.fit(features[training], labels[training])
	prediction = classifier.predict(features[testing])
	curmean = np.mean(prediction == labels[testing])
	means.append(curmean)
print("Mean Accuracy: {}".format(np.mean(means)))
print("Result of cross-validation using KFold: {}".format(means))

crossed = cross_val_score(classifier, features, labels)
print("Result of cross-validation using cross_val_score: {}".format(crossed))

# Using normalization
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
crossed = cross_val_score(classifier, features, labels)
print("Result with prescaling: {}".format(crossed))


names = list(set(labels))
labels = np.array([names.index(ell) for ell in labels])
preds = labels.copy()
preds[:] = -1
for train, test in kf:
	classifier.fit(features[train], labels[train])
	preds[test] = classifier.predict(features[test])

cmat = confusion_matrix(labels, preds)
print()
print("Confusion matrix: [rows represent true outcome, columns predicted outcome]")
print(cmat)
print(cmat.trace())
acc = cmat.trace()/float(cmat.sum())
print("Accuracy: {0:.1%}".format(acc))