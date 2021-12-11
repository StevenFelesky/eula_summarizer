"""
Steven Felesky
Dr. Ghosh
CSCI 390 Project
Fall 2021

Unsupervised EULA Summarizer
Input a EULA and recive an 8 sentence summary
EULA must be a .txt file
"""

import re
import sys
import nltk
import time
import numpy
import gensim
import networkx

from scipy import spatial
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize

#### GET USER INPUT

if (len(sys.argv) != 2):
	print("Incorrect number of inputs. Usage: projcet.py path/to/file.txt")
	sys.exit()

file_path = sys.argv[1]

#### END GET USER INPUT


#### FILE IO

file = open(file_path, "r", errors="ignore")
text = file.read()
file.close()

#### END FILE IO

start = time.perf_counter()

#### CLEAN TEXT

sents = sent_tokenize(text)
sents_clean = [re.sub(r'[^\w\s]', '', sent.lower()) for sent in sents] #remove punctuation
sents_clean = [re.sub('\n|\xa0|\t', '', sent.lower()) for sent in sents] #remove weird special characters

stop_words = stopwords.words('english')
sent_tokens=[[words for words in sent.split(' ') if words not in stop_words] for sent in sents_clean]

#### END CLEAN TEXT


#### VECTORIZE TEXT

w2v=Word2Vec(sent_tokens,vector_size=1,min_count=1,epochs=1000) #vectorize with word2vec
sent_vecs=[[w2v.wv[word][0] for word in words] for words in sent_tokens]

max_len=max([len(tokens) for tokens in sent_tokens]) #make all sentence vector array the same size
sent_vecs=[numpy.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sent_vecs]

#create similarity matrix with cosine similarity

similarity_matrix = numpy.zeros([len(sent_tokens), len(sent_tokens)])

for i,row in enumerate(sent_vecs):
    for j,col in enumerate(sent_vecs):
        similarity_matrix[i][j]=1-spatial.distance.cosine(row,col)

#### END VECTORIZE TEXT


#### TEXT RANK 

sim_graph = networkx.from_numpy_array(similarity_matrix)
scores = networkx.pagerank(sim_graph)

sents_dict = {sent:scores[index] for index,sent in enumerate(sents)}

#adjust rank values for sentences with useless data e.g. "you accept this agreement" or something of the like
for sent in sents_dict.keys():
	if re.search('agree|accept|agreement|license|terms and conditions', sent, re.IGNORECASE) != None:
		sents_dict[sent] = sents_dict.get(sent) * 0.09

top_sents = dict(sorted(sents_dict.items(), key=lambda x: x[1], reverse=True)[:8])

#### END TEXT RANK

end = time.perf_counter()

#### RESULTS

print("Time to complete summarization: %.2f seconds." % (end - start))

for sent in sents:
    if sent in top_sents.keys():
        print("\n%s" % sent)

#### END RESULTS