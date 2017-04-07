
from __future__ import division
from app import app
from flask import request, jsonify
import json

import re, pprint, nltk

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

import wikipedia as wp

from gensim import corpora, models, similarities
from gensim.similarities import Similarity
import gensim

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

processed_doc = []
document = [['brocolli'], ['tast'], ['nasti']]
#dictionary = corpora.Dictionary(document)

@app.route('/', methods=["POST"])
def index():

	if request.method == "POST":

		json_dict = request.get_json()

		keywords = json_dict['keyword'].split(",")
		for key in keywords:
			print key
	return jsonify(keywords=keywords)

@app.route('/wiki_search_summarize', methods=["POST"])
def wiki_sum():

	json_dict = request.get_json()

	keywords = json_dict['keyword'].split(",")
	summary = []
	for key in keywords:
		summ = wp.summary(key, sentences=10)
		summary.append(summ)

	return jsonify(summary=summary)

def wiki_sum_func(keywords):

	summary = []

	for key in keywords:
		summ = wp.summary(key, sentences=10)
		summary.append(summ)

	return summary


@app.route('/lsi', methods=["POST"])
def lsi_model():
	
	json_dict = request.get_json()

	doc_set = json_dict["summary"]
	keyword = json_dict["keyword"]

	if(lsi_model_func(doc_set)):
		return jsonify(status=True)


	return jsonify(status=False)

def lsi_model_func(doc_set):

	d = []
	for doc in doc_set:
		raw_doc = doc.lower()
		token_doc = tokenizer.tokenize(raw_doc)
		stopped_doc = [i for i in token_doc if not i in en_stop]
		texts_doc = [p_stemmer.stem(i) for i in stopped_doc]
		d.append(texts_doc)

	s = []
	s.append(d)

	dictionary1 = corpora.Dictionary(d)
	#dictionary.merge_with(dictionary1)
	s.append(dictionary1)
	return s



@app.route('/query_similar', methods=["POST"])
def query():

	json_dict = request.get_json()

	query = json_dict["query"]
	sims = []
	if(json_dict.get("summary")):
		sims = query_func(query, json_dict["summary"])
	else :
		sims = query_func(query)
	a = []
	for s in sims:
		b = []
		b.append(s[0])
		b.append(float(s[1]))
		a.append(b)
	b = dict(a)

	return jsonify(b)
	
def query_func(query, summary=False):

	dictionary = None
	d = lsi_model_func(summary)
	dictionary = d[1];
	do = d[0]
	doc_q = query
	raw_q = doc_q.lower()
	token_q = tokenizer.tokenize(raw_q)
	stopped_q = [i for i in token_q if not i in en_stop]
	texts_q = [p_stemmer.stem(i) for i in stopped_q]
	corpus = [dictionary.doc2bow(doc) for doc in do ]



	lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=1)
	index = similarities.MatrixSimilarity(lsi[corpus])

	q_bow = dictionary.doc2bow(texts_q)
	q_lsi = lsi[q_bow]		# convert the query to LSI space

	sims = index[q_lsi]		#perform a similarity query against corpus
	sims = sorted(enumerate(sims), key=lambda item: -item[1])
	return sims
	






