import os
import sys
import nltk.stem
import scipy as sp
import sklearn.datasets
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def dist_raw(v1, v2):
	delta = v1 - v2
	# norm() calculates the euclidean norm
	return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
	v1_normalized = v1 / sp.linalg.norm(v1.toarray())
	v2_normalized = v2 / sp.linalg.norm(v2.toarray())
	delta = v1_normalized - v2_normalized
	return sp.linalg.norm(delta.toarray())


def tfidf(term, doc, corpus):
	tf = doc.count(term) / len(doc)
	num_docs_with_term = len([d for d in corpus if term in d])
	idf = sp.log(len(corpus) / num_docs_with_term)
	return tf * idf

# stemming
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
		



# min_df parameter determines how CountVectorizer treats seldom words(minimum document frequency)
vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
# print(vectorizer.get_stop_words())
content = ["How to format my hard disk", "Hard disk format problems"]

X = vectorizer.fit_transform(content)

posts = [open(os.path.join('toy/', f)).read() for f in os.listdir('toy/')]

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
# print("#samples: %d, #features: %d" % (num_samples, num_features))
# print(vectorizer.get_feature_names())

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
# print(new_post_vec)
# print(new_post_vec.toarray())

best_doc = None
best_dist = sys.maxsize
best_i = None

for i in range(0, num_samples):
	post = posts[i]
	if post == new_post:
		continue
	post_vec = X_train.getrow(i)
	d = dist_norm(post_vec, new_post_vec)
	# print("=== Post %i with dist=%.2f: %s" % (i, d, post))
	if d < best_dist:
		best_dist = d
		best_i = i
# print("Best post is %i with dist=%.2f" % (best_i, best_dist))

a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
D = [a, abb, abc]
# print(tfidf("b", abc, D))

all_data = sklearn.datasets.fetch_20newsgroups(subset='all')
# print(len(all_data.filenames))
# print(all_data.target_names)


groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.fetch_20newsgroups(subset='train', categories=groups)
# print(len(train_data.filenames))

test_data = sklearn.datasets.fetch_20newsgroups(subset='test', categories=groups)
# print(len(test_data.filenames))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
# print("#samples: %d, #features: %d" % (num_samples, num_features))

num_clusters = 50
km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, random_state=3)
km.fit(vectorized)

# print(km.labels_)
# print(km.labels_.shape)

new_post = """Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks."""

new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_ == new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
	dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
	similar.append((dist, train_data.data[i]))

similar = sorted(similar)
# print("Count similar: %i" % len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

# print("=== #1 ===")
# print(show_at_1)
# print()

# print("=== #2 ===")
# print(show_at_2)
# print()

# print("=== #3 ===")
# print(show_at_3)

post_group = zip(train_data.data, train_data.target)
all = [(len(post[0]), post[0], train_data.target_names[post[1]]) for post in post_group]
graphics = sorted([post for post in all if post[2] == 'comp.graphics'])
# print(graphics[5])

noise_post = graphics[5][1]
analyzer = vectorizer.build_analyzer()
# print(list(analyzer(noise_post)))

useful = set(analyzer(noise_post)).intersection(vectorizer.get_feature_names())
print(sorted(useful))

for term in sorted(useful):
	print('IDF(%s)=%.2f' %(term, vectorizer._tfidf.idf_[vectorizer.vocabulary_[term]]))