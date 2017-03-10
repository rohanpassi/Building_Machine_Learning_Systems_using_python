import numpy as np
from gensim import corpora, models
from matplotlib import pyplot as plt


NUM_TOPICS = 100
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')
model = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word)

doc = corpus.docbyoffset(0)
topics = model[doc]
print(topics)

num_topics_used = [len(model[doc]) for doc in corpus]
fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')
fig.tight_layout()
fig.savefig('Figure_04_01.png')


ALPHA = 1.0

model1 = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=ALPHA)
num_topics_used1 = [len(model1[doc]) for doc in corpus]

fig,ax = plt.subplots()
ax.hist([num_topics_used, num_topics_used1], np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')

# The coordinates below were fit by trial and error to look good
ax.text(9, 223, r'default alpha')
ax.text(26, 156, 'alpha=1.0')
fig.tight_layout()
fig.savefig('Figure_04_02.png')