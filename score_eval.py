import os
import numpy as np
from tools import dataset_parameters as db_params
from score_model import SCoRe
from tools import caffe_lmdb
from tools.dataset_parameters import ATTRIBUTES, HIERARCHY, WORD2VEC


def parse_cmd():
    import sys, argparse
    parser = argparse.ArgumentParser(description='Trains SCoRe Zero-Shot classifier.')
    parser.add_argument('dir',       help='Directory where model is stored.')
    parser.add_argument('dataset',   help="Dataset. Options: 'AwA' or 'CUB'")
    parser.add_argument('semantics', help="Semantics. Options: '{}', '{}', '{}'".format(ATTRIBUTES, HIERARCHY, WORD2VEC))
    parser.add_argument('cnn',       help="CNN model. Options: 'AlexNet', 'GoogLeNet', 'VGG19'")

    # Evaluation parameters
    parser.add_argument('--gpu',        type=int, default=0, help="GPU id")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")

    # Parse arguments
    args = parser.parse_args(sys.argv[1:])

    # Check arguments
    assert args.semantics in (ATTRIBUTES, HIERARCHY, WORD2VEC)
    assert args.dataset in ('AwA', 'CUB', 'IFCB')
    assert args.cnn in ('AlexNet', 'GoogLeNet', 'VGG19')
    return args


def print_eval(mca, acc, semAcc, semAUC, semantics):
    for mode in ('source', 'target'):
        if mode in mca.keys():
            print '='*80 + '\n' + '%s Classes' % ('Source' if mode == 'source' else 'Target')
            print ' - MCA = ' + str(np.nanmean(mca[mode])*100)+'%'
            print ' - Acc = ' + str(np.nanmean(acc[mode])*100)+'%'
            if semantics != 'word2vec':
                print ' - %s Acc = %.2f%%' % (semantics, np.nanmean(semAcc[mode])*100)
            if semantics == 'attributes':
                print ' - %s AUC = %.2f%%' % (semantics, np.nanmean(semAUC[mode])*100)
            print ''


def eval_semantics(scores, gt, args):
    from sklearn.metrics import roc_auc_score
    num_semantics = gt.shape[1]
    acc, auc = np.nan*np.zeros((num_semantics,)), np.nan*np.zeros((num_semantics,))
    if args.semantics == ATTRIBUTES:
        for s, (pred, lbl) in enumerate(zip(scores.T, gt.T)):
            acc[s] = (pred*(lbl-0.5) > 0).astype(float).mean()
            if sum(lbl == 0) > 0 and sum(lbl == 1) > 0:
                auc[s] = roc_auc_score(lbl, pred)

    else:
        for s, (pred, lbl) in enumerate(zip(scores, gt.T)):
            acc[s] = (pred.argmax(axis=1) == lbl).astype(float).mean()
            onehot = np.zeros(pred.shape)
            for i, l in enumerate(lbl):
                onehot[i, int(l)] = 1
            if (onehot.sum(axis=0) == 0).sum() == 0:
                auc[s] = roc_auc_score(onehot, pred)
    return acc, auc


def evaluate_model(classes, constrains, LMDBs, args):
    deploy_proto, model_weights = '%s/deploy.prototxt' % args.dir, '%s/score.caffemodel' % args.dir
    model = SCoRe(source_classes=classes['source'],
                  target_classes=classes['target'],
                  semantics=args.semantics,
                  constrains=constrains,
                  cnn=args.cnn,
                  sem_coeff=None,
                  code_coeff=None)

    acc, mca, semAcc, semAUC = {}, {}, {}, {}
    lmdb_aux = {'source': 'testRecg', 'target': 'testZS'}
    for mode in ['target', 'source']:
        # Load CNN
        model.prep_for_deploy(args.batch_size,
                              source_net=mode == 'source',
                              target_net=mode == 'target',
                              deploy_fn=deploy_proto,
                              caffemodel_fn=model_weights,
                              gpu_id=args.gpu)

        # Load data
        data, labels, _ = caffe_lmdb.CaffeDatumReader(LMDBs['fts'][lmdb_aux[mode]]).load_records(num_records=0)
        if os.path.exists(LMDBs['fts'][lmdb_aux[mode]]):
            semantics = caffe_lmdb.CaffeDatumReader(LMDBs['sem'][lmdb_aux[mode]]).load_records(num_records=0)[0]
        else:
            semantics = None

        # Prep images (extract center crop and de-mean)
        crop_size = model.base_cnn.input_size
        img_center = np.array(data.shape[2:]) / 2.0
        crop_ulc = (img_center - crop_size/2.0).astype(int)
        data = data[:, :, crop_ulc[0]:crop_ulc[0]+crop_size, crop_ulc[1]:crop_ulc[1]+crop_size].astype(float)
        data -= np.array([104., 116., 122.]).reshape((1, 3, 1, 1))

        # Forward pass
        out_blobs = [model.scores['semantics'], model.scores['obj']]
        out = model.forward(data, out_blobs)

        # Compute evaluation measures
        # Semantics
        if args.semantics == ATTRIBUTES:
            scores = out[model.scores['semantics']]
        else:
            sem_lims = [0]+np.cumsum(model.num_states).tolist()
            scores = [out[model.scores['semantics']][:, sem_lims[i]:sem_lims[i+1]] for i in range(len(sem_lims)-1)]
        semAcc[mode], semAUC[mode] = eval_semantics(scores, semantics, args)

        # Objects
        scores = out[model.scores['obj']]
        acc[mode] = (scores.argmax(axis=1) == labels).mean()
        mca[mode] = np.array([(scores.argmax(axis=1)[labels == l] == l).mean() for l in range(len(classes[mode]))])

    return mca, acc, semAcc, semAUC


def main(args):
    if not os.path.exists(args.dir):
        raise ValueError('Folder {} does not exists.'.format(args.dir))

    # Class partitions and codewords
    classes = db_params.load_class_partition(args.dataset)
    constrains = db_params.load_semantic_codes(args.dataset, args.semantics,
                                               classes['all'], classes['source_idx'], classes['target_idx'])

    # Select input LMDBs
    LMDBs = db_params.select_lmdbs(args.dataset, args.semantics)

    # Test model
    mca, acc, semAcc, semAUC = evaluate_model(classes, constrains, LMDBs, args)
    print_eval(mca, acc, semAcc, semAUC, args.semantics)


if __name__ == '__main__':
    main(parse_cmd())
