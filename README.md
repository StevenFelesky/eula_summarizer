# Unsupervised EULA Summarization with NLP and ML
Steven Felesky  
Dr. Ghosh  
CSCI 390 Final Project  
Fall 2021

## A. How to run summarizer

### 1. Install the following python packages:
It is recommended that you create a python virtual environment for installing these packages. (virtualenv or conda)  
1. nltk
2. gensim
3. networkx
4. numpy
5. scipy

### 2. Obtain a EULA in .txt format
Dataset of EULAs provided in repository

### 3. Run in the command line
	python3 project.py path/to/eula.txt
	
## B. How to run Word2Vec model
Modify w2v_training.py to point to the correct location of train.txt
	
	python3 w2v_training.py
