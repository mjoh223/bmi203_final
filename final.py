import csv
import numpy as np
import pandas as pd
from Bio import SeqIO
from ann_class import NNet
from sklearn import metrics
import matplotlib.pyplot as plt

####################### Plot Autoencoder########################################
################################################################################
autoencoder = NNet(8,3,8)
X = np.identity(8)
y = np.identity(8)
X.shape
autoencoder.train(X, y, 1000000)
x_axis = np.asarray(range(0,8))
##ground truth
plt.plot(x_axis, X[0])
plt.plot(x_axis, X[1])
plt.plot(x_axis, X[2])
plt.plot(x_axis, X[3])
plt.plot(x_axis, X[4])
plt.plot(x_axis, X[5])
plt.plot(x_axis, X[6])
plt.plot(x_axis, X[7])
plt.title("Identity matrix (ground truth)")

plt.plot(x_axis, autoencoder.predict(X)[0])
plt.plot(x_axis, autoencoder.predict(X)[1])
plt.plot(x_axis, autoencoder.predict(X)[2])
plt.plot(x_axis, autoencoder.predict(X)[3])
plt.plot(x_axis, autoencoder.predict(X)[4])
plt.plot(x_axis, autoencoder.predict(X)[5])
plt.plot(x_axis, autoencoder.predict(X)[6])
plt.plot(x_axis, autoencoder.predict(X)[7])
plt.title("training iterations = 1000000")


##########################Part 2################################################
################################################################################

# The training set.
test = []
with open('/Users/matt/OneDrive/UCSF/algorithms/final/rap1-lieb-test.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        test.append(row)
test_set = np.asarray(test).flatten()

negatives = []
with open('/Users/matt/OneDrive/UCSF/algorithms/final/yeast-upstream-1k-negative.fa', 'r') as handle:
	for record in SeqIO.parse(handle, "fasta"):
		negatives.append(str(record.seq))
negatives = np.asarray(negatives).flatten()
negatives_sample = np.random.choice(negatives, 137)
negatives_sample2 = []
for seq in negatives_sample:
	n = int( np.random.randint(len(seq)-17, size=1) )
	negatives_sample2.append(seq[n:n+17])


positives = []
with open('/Users/matt/OneDrive/UCSF/algorithms/final/rap1-lieb-positives.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        positives.append(row)

positives = np.asarray(positives).flatten()
#encoding
conversion_dictionary = {
"A": [0,0],
"T": [0,1],
"G": [1,0],
"C": [1,1]
}
seq_list = []
for seq in positives:
	encoding_list = []
	for nt in seq:
		encoding = conversion_dictionary[nt]
		encoding_list.append(encoding)
	seq_list.append(encoding_list)

seq_test_list = []
for seq in test_set:
	encoding_list = []
	for nt in seq:
		encoding = conversion_dictionary[nt]
		encoding_list.append(encoding)
	seq_test_list.append(encoding_list)

seq_neg_list = []
for seq in negatives_sample2:
	encoding_list = []
	for nt in seq:
		encoding = conversion_dictionary[nt]
		encoding_list.append(encoding)
	seq_neg_list.append(encoding_list)

from itertools import chain
X_pos = [list(chain.from_iterable(x)) for x in seq_list]
X_neg = [list(chain.from_iterable(x)) for x in seq_neg_list]
test =  [list(chain.from_iterable(x)) for x in seq_test_list]

X_pos = np.array(X_pos)
X_neg = np.array(X_neg)
X = np.concatenate((X_pos,X_neg),axis=0)

y_pos = np.ones(137)
y_neg = np.zeros(137)
y = np.concatenate((y_pos, y_neg), axis =0)
test = np.array(test)

#train
RapNet = NNet(34,200,274,0.001, 0.05)
RapNet.train(X,y,100)

#test
RapNet = NNet(34,200,1,0.001, 0.05)
prediction_list = []
test[0]
for t in test:
	pred = RapNet.predict(t)
	prediction_list.append(pred)
len(test)


##########################Part 3################################################
################################################################################
#Optimal hidden layer size
auc_list = []
for i in [100, 500, 1000, 5000, 10000, 50000, 100000]:
	RapNet = NNet(34, i, 1)
	RapNet.lam = 0.001
	RapNet.alpha = 2
	y_score = []
	for t in X:
		p = RapNet.predict(t)
		y_score.append(p)
	y_score = np.array(y_score)
	fpr, tpr, thresh = metrics.roc_curve(y, y_score)
	auc = metrics.auc(fpr, tpr)
	auc_list.append([i, auc])
plt.plot([x[0] for x in auc_list], [x[1] for x in auc_list])
plt.title("AUC for hidden layer unit size")

#optimal learing rate
auc_list = []
for i in [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]:
	RapNet = NNet(34, 1000, 1)
	RapNet.lam = 0.001
	RapNet.alpha = i
	y_score = []
	for t in X:
		p = RapNet.predict(t)
		y_score.append(p)
	y_score = np.array(y_score)
	fpr, tpr, thresh = metrics.roc_curve(y, y_score)
	auc = metrics.auc(fpr, tpr)
	auc_list.append([i, auc])
plt.plot([x[0] for x in auc_list], [x[1] for x in auc_list])
plt.title("AUC for learning rate")

#system performance
RapNet = NNet(34, 100000, 1, 0.001, 0.05)
RapNet.train(X, y, 10)
test[0]
RapNet.predict(test[0])
y_score = []
for i,t in enumerate(test):
	p = RapNet.predict(t)
	y_score.append([test_set[i], p])
y_score = np.asarray(y_score)
pd.DataFrame(y_score).to_csv("test.out")

fpr, tpr, thresh = metrics.roc_curve(y, y_score)
auc = metrics.auc(fpr, tpr)
