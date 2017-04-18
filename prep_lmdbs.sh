#!/usr/bin/env bash
DB=$1
IMG_DIR=$2

if [ -z ${DB} ]; then
     echo 'usage:'
     echo '  ./prep_lmdbs.sh <DB> [<IMG_DIR>]'
     echo '  <DB> must be either AwA, CUB or IFCB.'
     exit 0
fi

if [ -z ${IMG_DIR} ]; then
     IMG_DIR='/PATH/TO/IMAGE/FOLDER'
fi

if [ ! -d LMDBs ]; then
    mkdir LMDBs
fi
if [ ! -d LMDBs/${DB} ]; then
    mkdir LMDBs/${DB}
fi

# Image
SUBSETS=( 'train' 'testRecg' 'testZS' )
for subset in "${SUBSETS[@]}"; do
    if [ ! -d 'LMDBs/'${DB}'/'${subset}'.image_lmdb' ]; then
        python tools/create_lmdb.py ${DB} Image ${subset} --image_dir ${IMG_DIR}
    fi
done

# Attributes
for subset in "${SUBSETS[@]}"; do
    if [ ! -d 'LMDBs/'${DB}'/'${subset}'.attributes_lmdb' ]; then
        python tools/create_lmdb.py ${DB} Attributes ${subset}
    fi
done

# Hierarchy
for subset in "${SUBSETS[@]}"; do
    if [ ! -d 'LMDBs/'${DB}'/'${subset}'.hierarchy_lmdb' ]; then
        python tools/create_lmdb.py ${DB} Hierarchy ${subset}
    fi
done

# Word2Vec
for subset in "${SUBSETS[@]}"; do
    if [ ! -d 'LMDBs/'${DB}'/'${subset}'.word2vec_lmdb' ]; then
        python tools/create_lmdb.py ${DB} Word2Vec ${subset}
    fi
done
