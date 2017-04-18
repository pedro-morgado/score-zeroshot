import argparse
import os
import sys
import numpy as np
import dataset_parameters as ds
from dataset_parameters import ATTRIBUTES, HIERARCHY, WORD2VEC


def parse_cmd():
    parser = argparse.ArgumentParser(description='Prepares datasets.')
    parser.add_argument('dataset',   help="Dataset. Options: 'AwA' or 'CUB'.")
    parser.add_argument('type', default='Image', help="What LMDB to generate. Options: 'Image', '{}', '{}', '{}'".format(ATTRIBUTES, HIERARCHY, WORD2VEC))
    parser.add_argument('partition', help="Partition. Options: 'train', 'testRecg', 'testZS'.")
    parser.add_argument('--image_dir', default='', help='Directory containing all images. Images should be stored at '
                                                        '{BASEDIR}/{FN} where file names FN are given by the lines of '
                                                        'the partition file.')
    parser.add_argument('-o', '--output', default='', help='Output LMDB directory name. Raises exception if directory '
                                                           'already exists.')
    args = parser.parse_args(sys.argv[1:])

    assert args.dataset in ('AwA', 'CUB', 'IFCB')
    assert args.partition in ('train', 'testRecg', 'testZS')
    assert args.type in ('Image', ATTRIBUTES, HIERARCHY, WORD2VEC)
    return args


def create_semantics_lmdb(labels, constrains, train, lmdb_fn):
    from caffe_lmdb import save_records_to_lmdb
    sem_labels = [const['labels']['source' if train else 'target'] for const in constrains.itervalues()]
    sem_labels = np.concatenate([np.expand_dims(s.argmax(axis=0), 1) for s in sem_labels], 1)
    sem_labels = sem_labels[labels, :]

    save_records_to_lmdb(lmdb_fn, sem_labels, labels)


def create_images_lmdb(images_fn, labels, lmdb_fn):
    from caffe_lmdb import CaffeDatumWriter
    import skimage.io as sio
    import skimage.transform as skt

    def pre_processing(img):
        if img.ndim == 2:   # Gray image
            img = np.concatenate([np.expand_dims(img, 2)]*3, 2)
        elif img.shape[2] > 3:  # RGB+depth or RGB+alpha. Ignore 4th channel
            img = img[:, :, :3]
        img = skt.resize(img, (256, 256)) * 255
        img = img[:, :, (2, 1, 0)]
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.uint8)

    if os.path.exists(lmdb_fn):
        raise ValueError(lmdb_fn + ' already exists!')

    print 'Adding %d records to lmdb %s.' % (len(images_fn), lmdb_fn)
    map_size = (256*256*3 + 1)*len(images_fn)*64*2.0
    writer = CaffeDatumWriter(lmdb_fn, map_size)
    for i, (fn, lbl) in enumerate(zip(images_fn, labels)):
        image = pre_processing(sio.imread(fn))
        writer.add_records(np.expand_dims(image, 0), labels=[lbl])
        if i % 1000 == 0:
            print 'Samples saved: %5d/%5d' % (i, len(images_fn))
            sys.stdout.flush()


def read_partition(dataset, partition, classes):
    images_fn = [line.strip().split()[0] for line in open('data/%s/%s.txt' % (dataset, partition))]
    labels_str = [fn.split('/')[0] for fn in images_fn]
    labels = [classes['source' if partition != 'testZS' else 'target'].index(l) for l in labels_str]
    return images_fn, labels


def main(args):
    # Read partition
    classes = ds.load_class_partition(args.dataset)
    images_fn, labels = read_partition(args.dataset, args.partition, classes)

    if len(args.output) == 0:
        # Default LMDB location
        outp_fn = 'LMDBs/{}/{}.{}_lmdb'.format(args.dataset, args.partition, args.type.lower())
    else:
        outp_fn = args.output

    # Create LMDBs
    if args.type == 'Image':
        if len(args.image_dir) > 0:
            images_fn = [os.path.join(args.image_dir, fn) for fn in images_fn]
        create_images_lmdb(images_fn, labels, outp_fn)
    else:
        source = args.partition.lower() != 'testzs'
        constrains = ds.load_semantic_codes(args.dataset, args.type, classes['all'], classes['source_idx'], classes['target_idx'])
        create_semantics_lmdb(labels, constrains, source, outp_fn)


if __name__ == '__main__':
    main(parse_cmd())
