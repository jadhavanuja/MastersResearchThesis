#%matplotlib inline
import csv
import pandas
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
import gensim
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from nltk.tokenize import word_tokenize
import gensim
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [x for x in dataset[i]]
	return dataset	
	
def separate_each_column(docs):
	
	title = [d[0] for d in docs]
	
	url = [d[1] for d in docs]

	content = [d[2] for d in docs]
	
	keyword = [d[3] for d in docs]
	
	description = [d[4] for d in docs]
	
	entry_time = [d[5] for d in docs]
	
	labels = [d[6] for d in docs]
	
	return content, labels

class MeanEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		# if a text is empty we should return a vector of zeros
		# with the same dimensionality as all the other vectors
		self.dim = len(word2vec.itervalues().next())

	def fit(self, X, y):
		return self
	
	def transform(self, X):
		return np.array([
			np.mean([self.word2vec[w] for w in words if w in self.word2vec]
				or [np.zeros(self.dim)], axis=0)
			for words in X
			])

class TfidfEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.word2weight = None
		self.dim = len(word2vec.itervalues().next())
		
	def fit(self, X, y):
		tfidf = TfidfVectorizer(analyzer=lambda x: x)
		tfidf.fit(X)
		max_idf = max(tfidf.idf_)
		self.word2weight = defaultdict(
			lambda: max_idf,
			[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
		return self
		
	def transform(self, X):
		return np.array([
			   np.mean([self.word2vec[w] * self.word2weight[w]
						for w in words if w in self.word2vec] or
						[np.zeros(self.dim)], axis=0)
				for words in X
            ])
			
if __name__ == "__main__":
	filename = '/home/acj03778/Desktop/Publication/Datasets/450-train-cross-val.csv'
	traindata = loadCsv(filename)
	
	""" Vectorise and TF-IDF transform the corpus"""
	train_content, train_label = separate_each_column(traindata)
	

	""" Support Vector Machine (SVM) classifier"""
	svm_clf = Pipeline([('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=   5, random_state=42)),
	])
	
	
	""" Naive Bayes classifier """
	mnb_clf = Pipeline([('vect', CountVectorizer()),
						  ('tfidf', TfidfTransformer()),
						  ('clf', MultinomialNB())
						  ])
	
	"""Cross Validation on the 450 articles using SVM"""
	scores = cross_val_score(svm_clf, train_content, train_label, cv=5)
	print (scores)
	print("Cross Val Accuracy of SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
	predicted = cross_val_predict(svm_clf, train_content, train_label, cv=5)
	print (metrics.accuracy_score(train_label, predicted) )

	"""Cross Validation on the 450 articles using MNB"""
	scores_mnb = cross_val_score(mnb_clf, train_content, train_label, cv=5)
	print (scores_mnb)
	print("Cross Val Accuracy of MNB: %0.2f (+/- %0.2f)" % (scores_mnb.mean(), scores_mnb.std() * 2))
	
	predicted_mnb = cross_val_predict(mnb_clf, train_content, train_label, cv=5)
	print (metrics.accuracy_score(train_label, predicted_mnb) )
	
	"""Test set"""
	svm_clf.fit(train_content, train_label)
	mnb_clf.fit(train_content, train_label)
	
	testfile = '/home/acj03778/Desktop/Publication/Datasets/100-test.csv'
	testdata = loadCsv(testfile)
	test_content, test_label = separate_each_column(testdata)

	""" Predict the test dataset using SVM"""
	predicted = svm_clf.predict(test_content)
	print('SVM correct prediction: {:4.2f}'.format(np.mean(predicted == test_label)))
	print(metrics.classification_report(test_label, predicted, target_names=test_label))
	print(metrics.confusion_matrix(test_label, predicted))
	print('-------------------------------------------------------------------------------------------')

	""" Predict the test dataset using MNB"""
	predicted = mnb_clf.predict(test_content)
	print('SVM correct prediction: {:4.2f}'.format(np.mean(predicted == test_label)))
	print(metrics.classification_report(test_label, predicted, target_names=test_label))
	print(metrics.confusion_matrix(test_label, predicted))
	print('-------------------------------------------------------------------------------------------')