import os
import numpy as np
from collections import OrderedDict

ATTRIBUTES = 'Attributes'
HIERARCHY = 'Hierarchy'
WORD2VEC = 'Word2Vec'

PARTITIONS = {db: {'classes': 'data/'+db+'/classes.txt',
                   'source': 'data/'+db+'/train_classes.txt',
                   'target': 'data/'+db+'/test_classes.txt'}
              for db in ('AwA', 'CUB', 'IFCB')}


def load_class_partition(dataset):
    classes = [line.strip().split()[1] for line in open(PARTITIONS[dataset]['classes'])]
    source_classes = [line.strip().split()[1] for line in open(PARTITIONS[dataset]['source'])]
    target_classes = [line.strip().split()[1] for line in open(PARTITIONS[dataset]['target'])]

    source_idx = [classes.index(cls) for cls in source_classes]
    target_idx = [classes.index(cls) for cls in target_classes]
    classes = {'all': classes,
               'source': source_classes,
               'source_idx': source_idx,
               'target': target_classes,
               'target_idx': target_idx}

    return classes


def _att(DB, mean_correction=True, src_idx=None, trg_idx=None):
    codes_fn = 'data/%s/Codes_Attributes.txt' % DB
    semantics_fn = 'data/%s/attributes.txt' % DB

    semantics = [line.strip().split()[1] for line in open(semantics_fn)]
    codes = np.loadtxt(codes_fn).astype(float)
    if codes.max() > 1:
        codes /= 100.

    # Set undefined codes (typically marked with -1) to the mean
    code_mean = codes[src_idx, :].mean(axis=0)
    for s in range(len(semantics)):
        codes[codes[:, s] < 0, s] = code_mean[s] if mean_correction else 0.5

    # Mean correction
    if mean_correction:
        for s in range(len(semantics)):
            codes[:, s] = codes[:, s] - code_mean[s] + 0.5

    constrains = OrderedDict([(sem, {'codewords': np.array([[-1, 1]]),
                                     'idxDim': s,
                                     'labels': {'source': None,
                                                'target': None}})
                              for s, sem in enumerate(semantics)])
    for s, sem in enumerate(semantics):
        src_lbl = np.zeros((2, len(src_idx)))
        src_lbl[0, :] = 1 - codes[src_idx, s]
        src_lbl[1, :] = codes[src_idx, s]

        trg_lbl = np.zeros((2, len(trg_idx)))
        trg_lbl[0, :] = 1 - codes[trg_idx, s]
        trg_lbl[1, :] = codes[trg_idx, s]

        constrains[sem]['labels']['source'] = src_lbl
        constrains[sem]['labels']['target'] = trg_lbl

    return constrains


def _hrchy(DB, classes, src_idx, trg_idx):
    from hierarchy import load

    def simplex(num_vecs):
        from scipy.linalg import svd
        cw = np.eye(num_vecs)
        cw -= cw.mean(axis=1, keepdims=True)
        cw, _, _ = svd(cw)
        return cw[:, :num_vecs-1]

    hrchy_fn = 'data/%s/wordnet_hierarchy.txt' % DB
    hrchy = load(hrchy_fn)
    constrains = OrderedDict()
    k = 0
    for q in hrchy.nodes:
        if len(q.fpointer) <= 1:
            continue
        labs = len(q.fpointer)*np.ones((len(classes)))
        for s, cls in enumerate(classes):
            if cls in q.name['classes']:
                labs[s] = [cls in hrchy[cid].name['classes'] for cid in q.fpointer].index(True)
        assigns = np.zeros((len(q.fpointer)+1, len(classes)))
        for c in range(len(q.fpointer)+1):
            assigns[c, labs == c] = 1

        constrains[q.identifier] = {'codewords': simplex(len(q.fpointer)+1).T,
                                    'idxDim': np.arange(k, k+len(q.fpointer)),
                                    'labels': {'source': assigns[:, src_idx],
                                               'target': assigns[:, trg_idx]}}
        k += len(q.fpointer)
    source_deg = [key for key in constrains.keys() if constrains[key]['labels']['source'].sum(axis=1).max() == len(src_idx)]
    [constrains.pop(key) for key in source_deg]
    return constrains


def _word2vec(DB, classes, src_idx, trg_idx):
    codes_fn_base = 'data/%s/Codes_Wiki_W2V' % DB
    windows = [3, 5, 10]
    sizes = [50, 100, 500]

    k = 0
    constrains = OrderedDict()
    for wind in windows:
        for sz in sizes:
            cw = np.loadtxt("%s_%d_%d.txt" % (codes_fn_base, sz, wind))
            cw /= (cw**2.0).sum(axis=1, keepdims=True)**0.5
            assigns = np.eye(len(classes))
            name = 'sz%d_wind%d' % (sz, wind)
            constrains[name] = {'codewords': cw.T,
                                'idxDim': range(k, k+cw.shape[1]),
                                'labels': {'source': assigns[:, src_idx],
                                           'target': assigns[:, trg_idx]}}
            k += cw.shape[1]
    return constrains


def load_semantic_codes(dataset, semantics, classes, src_idx, trg_idx):
    if ATTRIBUTES == semantics:
        constrains = _att(dataset, mean_correction=True, src_idx=src_idx, trg_idx=trg_idx)

    elif HIERARCHY == semantics:
        constrains = _hrchy(dataset, classes, src_idx, trg_idx)

    elif WORD2VEC == semantics:
        constrains = _word2vec(dataset, classes, src_idx, trg_idx)

    else:
        raise ValueError('Unknown semantics')

    return constrains


def select_lmdbs(dataset, semantics):
    baseDB_name = {'train':    'LMDBs/%s/train' % dataset,
                   'testRecg': 'LMDBs/%s/testRecg' % dataset,
                   'testZS':   'LMDBs/%s/testZS' % dataset}

    # Features
    LMDBs = {}
    LMDBs['fts'] = {'train':    '%s.image_lmdb' % baseDB_name['train'],
                    'testRecg': '%s.image_lmdb' % baseDB_name['testRecg'],
                    'testZS':   '%s.image_lmdb' % baseDB_name['testZS']}

    # Semantics
    LMDBs['sem'] = {'train': '%s.%s_lmdb' % (baseDB_name['train'], semantics.lower()),
                    'testRecg': '%s.%s_lmdb' % (baseDB_name['testRecg'], semantics.lower()),
                    'testZS': '%s.%s_lmdb' % (baseDB_name['testZS'], semantics.lower())}

    # Check if LMDBs exist
    assert all([os.path.exists(fn) for fn in LMDBs['fts'].itervalues()])
    assert all([os.path.exists(fn) for fn in LMDBs['sem'].itervalues()])
    return LMDBs

