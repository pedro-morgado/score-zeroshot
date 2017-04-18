import numpy as np
from score_model import SCoRe, TrainingOpts
from tools import dataset_parameters as db_params
from tools.dataset_parameters import ATTRIBUTES, HIERARCHY, WORD2VEC


def parse_cmd():
    import sys, argparse
    parser = argparse.ArgumentParser(description='Trains SCoRe Zero-Shot classifier.')
    parser.add_argument('dir',       help='Directory to store model. Raises exception if a directory with the same name already exists.')
    parser.add_argument('dataset',   help="Dataset. Options: 'AwA' or 'CUB'")
    parser.add_argument('semantics', help="Semantics. Options: '{}', '{}', '{}'".format(ATTRIBUTES, HIERARCHY, WORD2VEC))
    parser.add_argument('cnn',       help="CNN model. Options: 'AlexNet', 'GoogLeNet', 'VGG19'")

    # Regularizers
    parser.add_argument('-g', action="store", dest="gamma",       default='0.0', help="Semantic regularization coefficient")
    parser.add_argument('-c', action="store", dest="cwReg",       default='inf', help="Codeword regularization coefficient")
    parser.add_argument('-p', action="store", dest="paramReg",    default='0.0005', help="Parameter regularization coefficient (weight decay)")

    # Training parameters
    parser.add_argument('--gpu',             type=int,   default=0,     help="GPU id")
    parser.add_argument('--iters',           type=int,   default=10000, help="Number of iterations")
    parser.add_argument('--init_lr',         type=float, default=0.01,  help="Base learning rate")
    parser.add_argument('--num_lr_decays',   type=int,   default=0,     help="Number of steps in lr decay policy.")
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,   help="Decay factor at each step of lr decay policy")
    parser.add_argument('--batch_size',      type=int,   default=64,    help="Batch size")
    parser.add_argument('--snapshot',        default='', help="Caffemodel state to initialize training.")

    # Parse arguments
    args = parser.parse_args(sys.argv[1:])

    # Check arguments
    assert args.semantics in (ATTRIBUTES, HIERARCHY, WORD2VEC)
    assert args.dataset in ('AwA', 'CUB', 'IFCB')
    assert args.cnn in ('AlexNet', 'GoogLeNet', 'VGG19')
    return args


def train_model(classes, constrains, LMDBs, args):
    model_proto, solver_proto, model_weights = '%s/train.prototxt' % args.dir, '%s/solver.prototxt' % args.dir, '%s/score.caffemodel' % args.dir
    model = SCoRe(source_classes=classes['source'],
                  target_classes=classes['target'],
                  semantics=args.semantics,
                  constrains=constrains,
                  cnn=args.cnn,
                  sem_coeff=np.inf if args.gamma == 'inf' else float(args.gamma),
                  code_coeff=np.inf if args.cwReg == 'inf' else float(args.cwReg))

    trainOpts = TrainingOpts(iters=args.iters,
                             init_lr=args.init_lr,
                             lr_decay_factor=args.lr_decay_factor,
                             num_lr_decays=args.num_lr_decays,
                             paramReg=float(args.paramReg))

    model.prep_for_training(model_proto, solver_proto, LMDBs['fts'], LMDBs['sem'], trainOpts, args.batch_size, args.gpu)
    if len(args.snapshot) > 0:
        model.load(args.snapshot)
    try:
        model.train(1)
        for i in range(1, args.iters+1, 20):
            model.train(20)
    except KeyboardInterrupt:
        print 'Training interrupted. Current model will be saved at {}.'.format(model_weights)
    finally:
        model.save(model_weights)


def main(args):
    # Class partitions and codewords
    classes = db_params.load_class_partition(args.dataset)
    constrains = db_params.load_semantic_codes(args.dataset, args.semantics, classes['all'], classes['source_idx'], classes['target_idx'])

    # Select input LMDBs
    LMDBs = db_params.select_lmdbs(args.dataset, args.semantics)

    # Train model
    train_model(classes, constrains, LMDBs, args)


if __name__ == '__main__':
    main(parse_cmd())
