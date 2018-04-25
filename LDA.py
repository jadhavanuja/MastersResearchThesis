from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import figure
import os
import argparse
from time import time
import csv
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import bokeh.plotting as bp
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show, save 
from bokeh.models import HoverTool
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.patheffects as PathEffects
import pymysql

sns.set_style('darkgrid')

sns.set_palette('muted')

sns.set_context("notebook", font_scale=1.5,

                rc={"lines.linewidth": 2.5})


n_features = 1000
n_samples = 2000
n_topics = 2
n_top_words = 20

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
	
	return content

def top_words(model, feature_names, n_top_words):
	top_words = []
	for topic_idx, topic in enumerate(model.components_):
		message = "Topic #%d: " % topic_idx
		message += " ".join([feature_names[i]
		for i in topic.argsort()[:-n_top_words - 1:-1]])
		top_words.append(message)
	return top_words

def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		message = "Topic #%d: " % topic_idx
		message += " ".join([feature_names[i]
		for i in topic.argsort()[:-n_top_words - 1:-1]])
		print(message)


def loadTopic1(dataset,Articles_Topic_1):
	s = []
	id = []
	for i in Articles_Topic_1:
		id.append(i[0])
	
	for item in dataset:
		if(int(item[0]) in id):
			#row = []
			#row.append(int(item[0]))
			#row.append(item[1])
			s.append(item[1])
		else:
			continue
	return s
		
if __name__ == "__main__":

	#Load Dataset
	print("Loading dataset...")
	t0 = time()
	filename = '/home/acj03778/Desktop/Publication/Datasets/824.csv'
	dataset = loadCsv(filename)

	
	#Load Content
	content = separate_each_column(dataset)
	print("done in %0.3fs." % (time() - t0))
		
	#Tf Feature Extraction
	print("Extracting tf features for LDA...")
	tf_vectorizer = CountVectorizer(max_df=0.98, min_df=2,max_features=n_features,stop_words='english')
	t0 = time()
	tf_content = tf_vectorizer.fit_transform(content)
	X = tf_content.toarray()
	
	print("done in %0.3fs." % (time() - t0))
	print("Fitting LDA models with tf features, "
		  "n_samples=%d and n_features=%d..."
		  % (n_samples, n_features))
	
	#LDA Model Definition
	lda = LatentDirichletAllocation(n_topics = n_topics,max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
	
	#Generate topics from LDA model and corresponding top words
	t0 = time()
	X_topics = lda.fit_transform(tf_content)
	print(X_topics)
		
	print("done in %0.3fs." % (time() - t0))
	print("\nTopics in LDA model:")
	tf_feature_names = tf_vectorizer.get_feature_names()
	topic_words = top_words(lda, tf_feature_names, n_top_words)
	print_top_words(lda,tf_feature_names,n_top_words)
	
	
	
	print ('----------------------------------------------------------------------------')
	
	j=0
	X_topics_ID = []
	Articles_Topic_0 = []
	Articles_Topic_1 = []
	consolidated_list = []
	for i in X_topics:
		row = []
		row.append(i[0])
		row.append(i[1])
		
		row1 = []
		row2 = []
		if i[0] > i[1]:
			row1.append(i[0])
			Articles_Topic_0.append(row1)
			
			row2.append('0')
			consolidated_list.append(row2)
			
		else:
			row1.append(i[1])
			Articles_Topic_1.append(row1)
			
			row2.append('1')
			consolidated_list.append(row2)
		j = j +1
	print ('----------------------------------Printing Articles in Topic 0 ------------------------------------------')
	print (Articles_Topic_0)
	print ('----------------------------------Printing Articles in Topic 1 ------------------------------------------')
	print (Articles_Topic_1)
	print ('----------------------------------Printing Topics of all 824 articles ------------------------------------------')
	print (consolidated_list)
	
	
	colormap = np.array(["yellow","green"])
	
	#----------------------------------Plotting X_topics------------------------------------------
	X_tsne = TSNE(learning_rate=100,perplexity = 100).fit_transform(X_topics)

	fig = plt.figure(figsize=(10, 5))
	plt.subplot(121)
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color = colormap)
	fig.savefig('X_topics.png')

	#----------------------------------Threshold & 3D plot ------------------------------------------
	threshold = 0.0
	_idx = np.amax(X_topics, axis=1) > threshold  # idx of news that > threshold
	_topics = X_topics[_idx]
	num_example = len(_topics)
	
	_lda_keys = []
	for i in xrange(_topics.shape[0]):
		_lda_keys += _topics[i].argmax(),
	

	X_tsne = TSNE(learning_rate=100).fit_transform(consolidated_list)
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 0, color = colormap[_lda_keys][:num_example])
	fig.savefig('full_figure_103.png')
	