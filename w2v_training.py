"""
Steven Felesky
Dr. Ghosh
CSCI 390 Project
Fall 2021

gensim word2vec training
"""

import re
import nltk
import time
import gensim

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize

file = open('data/eulas/train.txt', "r", errors="ignore")
text = file.read()
file.close()

sents = sent_tokenize(text)
sents_clean = [re.sub(r'[^\w\s]', '', sent.lower()) for sent in sents] #remove punctuation
sents_clean = [re.sub('\n|\xa0|\t', '', sent.lower()) for sent in sents] #remove weird special characters

stop_words = stopwords.words('english')
sent_tokens=[[words for words in sent.split(' ') if words not in stop_words] for sent in sents_clean]

start = time.perf_counter()

model = Word2Vec(sent_tokens,vector_size=1,min_count=1,epochs=1000)

end = time.perf_counter()

model.save('gensim-w2v-eula.model')
print("Time to complete training: %.2f" % (end - start))
