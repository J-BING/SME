import os, sys
import cPickle
import gzip
import time

import numpy as np
import scipy.sparse as sp

# Put the dba data absolute path here
datapath = 'dba_s/'
assert datapath is not None

if 'data' not in os.listdir('../'):
    os.mkdir('../data')


def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

#################################################
### Creation of the entities/indices dictionnaries

t1 = time.time()
np.random.seed(753)

entleftset = set()
entrightset = set()
relset = set()

f = gzip.open(datapath + 'dba_s-train.txt.gz', 'r')
g = (parseline(line[:-1]) for line in f)

# read l, o, r from raw data
for lhs, rel, rhs in g:
    entleftset.add(lhs[0])
    entrightset.add(rhs[0])
    relset.add(rel[0])
f.close()
print 'read data:{}s'.format(time.time() - t1)

t2 = time.time()
entleftlist = list(entleftset)
entrightlist = list(entrightset)
rellist = list(relset)

# calculate # of unique entities
nbshared = len(np.intersect1d(entleftlist, entrightlist))
nbleft = len(entleftlist) - nbshared
nbright = len(entrightlist) - nbshared
nbrel = len(rellist) - len(np.intersect1d(rellist, entrightlist))- len(np.intersect1d(rellist, entleftlist))
print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
print 'computate specific entities:{}s'.format(time.time() - t2)

# create entity2idx and idx2entity
t3 = time.time()
entity2idx = {item: i for i, item in enumerate(entrightlist)}
entrightlist = []
idx = len(entity2idx)
for item in entleftlist:
    if item not in entity2idx:
        entity2idx[item] = idx
        idx += 1
entleftlist = []
for item in rellist:
    if item not in entity2idx:
        entity2idx[item] = idx
        idx += 1
rellist = []
print 'if rel is unique:{}'.format(nbrel == idx - (nbright + nbshared + nbleft))
print 'create entity2idx:{}s'.format(time.time() - t3)

idx2entity = {v: k for k,v in entity2idx.items()}
print('if the length of en2in equals to in2en:{}'.format(len(entity2idx) == len(idx2entity)))

f = open('../data/dba_entity2idx.pkl', 'w')
g = open('../data/dba_idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
idx2entity = {}
f.close()
g.close()

#################################################
### Creation of the dataset files

unseen_ents=[]
remove_tst_ex=[]

for datatyp in ['train', 'valid', 'test']:
    print datatyp
    f = gzip.open(datapath + 'dba_s-%s.txt.gz' % datatyp, 'r')
    length = len(f.readlines())

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(entity2idx.values()) + 1, length),
            dtype='float32')
    inpr = sp.lil_matrix((np.max(entity2idx.values()) + 1, length),
            dtype='float32')
    inpo = sp.lil_matrix((np.max(entity2idx.values()) + 1, length),
            dtype='float32')
    # Fill the sparse matrices
    ct = 0
    gn = (parseline(line[:-1]) for line in f)
    for lhs, rel, rhs in gn:
        if lhs[0] in entity2idx and rhs[0] in entity2idx and rel[0] in entity2idx: 
            inpl[entity2idx[lhs[0]], ct] = 1
            inpr[entity2idx[rhs[0]], ct] = 1
            inpo[entity2idx[rel[0]], ct] = 1
            ct += 1
        else:
            if lhs[0] in entity2idx:
                unseen_ents+=[lhs[0]]
            if rel[0] in entity2idx:
                unseen_ents+=[rel[0]]
            if rhs[0] in entity2idx:
                unseen_ents+=[rhs[0]]
            remove_tst_ex+=['{}\t{}\t{}'.format(lhs, rel, rhs)]
    f.close()

    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('../data/dba_s-%s-lhs.pkl' % datatyp, 'w')
    g = open('../data/dba_s-%s-rhs.pkl' % datatyp, 'w')
    h = open('../data/dba_s-%s-rel.pkl' % datatyp, 'w')
    cPickle.dump(inpl.tocsr(), f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()

unseen_ents=list(set(unseen_ents))
print 'unseen_ents: %s' % len(unseen_ents)
remove_tst_ex=list(set(remove_tst_ex))
print 'remove_tst_ex: %s' % len(remove_tst_ex)

if len(remove_tst_ex) !=0:
    f = gzip.open(datapath + 'dba_s-train.txt.gz', 'a')
    for i in remove_tst_ex:
        f.write(i+'\n')
    f.close()





