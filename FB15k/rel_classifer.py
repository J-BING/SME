from collections import defaultdict
import gzip
import json

# datapath [dbv_s/, dba_s/]
datapath = 'dbv_s/'
assert datapath is not None

def gen_opener():
    for datatyp in ['train', 'valid', 'test']:
        if datapath.startswith('dba'):
            f = gzip.open(datapath + 'dba_s-%s.txt.gz' % datatyp, 'r')
        else:
            f = open(datapath + 'dbv_s-%s.txt' % datatyp, 'r')
        yield f
        f.close()

# def gen_concatenate(iterators):
#     '''
#     Chain a sequence of iterators together into a single sequence.
#     '''
#     for it in iterators:
#         yield from it

def parselines(files):
    for file in files:
        for line in file:
            lhs, rel, rhs = line.split('\t')
            lhs = lhs.split(' ')[0]
            rhs = rhs.split(' ')[0]
            rel = rel.split(' ')[0]
            yield lhs, rel, rhs

left = defaultdict(list)
right = defaultdict(list)
left_sum = defaultdict(list)
right_sum = defaultdict(list)
relset = set()
rel_ctg = defaultdict(list)

files = gen_opener()
item = parselines(files)

# gn = (parseline(line[:-1]) for line in f)
for lhs, rel, rhs in item:
    left['+'.join([rel, rhs])].append(1)
    right['+'.join([rel, lhs])].append(1)
    relset.add(rel)

for k in list(left.keys()):
    v = left.pop(k)
    left_sum[k.split('+')[0]].append(len(v))

for k in list(right.keys()):
    v = right.pop(k)
    right_sum[k.split('+')[0]].append(len(v))

left_cnt = {k: float(sum(v)) / len(v) for k, v in left_sum.items()}
del left_sum
right_cnt = {k: float(sum(v)) / len(v) for k, v in right_sum.items()}
del right_sum

for k, v in left_cnt.items():
    if v < 1.5:
        if right_cnt[k] < 1.5:
            rel_ctg['1to1'].append(k)
        else:
            rel_ctg['1toM'].append(k)
    else:
        if right_cnt[k] < 1.5:
            rel_ctg['Mto1'].append(k)
        else:
            rel_ctg['MtoM'].append(k)

with open(''.join([datapath.split('_')[0], '_rel_ca.json']), 'wt') as f:
    json.dump(rel_ctg, f, indent=4)

total = len(relset)

for k, v in rel_ctg.items():
    print '{}: {}%'.format(k, float(len(v)) / total * 100)
