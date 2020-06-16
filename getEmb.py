import pickle
import numpy as np
from model import parse_embeddings
import os, sys

def getEmb(path):
	f = open(path, 'rb')
	embeddings = pickle.load(f)
	f.close()
	entiE, relE, _ = parse_embeddings(embeddings)
	return entiE.E.get_value(), relE.E.get_value()

if __name__ == '__main__':
	path = 'FB15k/FB15k_TransE/best_valid_model.pkl'
	entiE, relE, = getEmb(path)
	if 'KGembedding' not in os.listdir('.'):
		os.mkdir('KGembedding')
	np.save('KGembedding/FB15k_e', entiE)
	np.save('KGembedding/FB15k_r', relE)

