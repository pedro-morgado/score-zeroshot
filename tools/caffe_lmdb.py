import os, shutil, sys
import lmdb, numpy as np
from caffe.proto.caffe_pb2 import Datum
from caffe.io import datum_to_array


class CaffeDatumReader(object):
    def __init__(self, fn):
        self.fn = fn

        env = lmdb.open(fn, readonly=True)
        self.num_records = env.stat()['entries']
        env.close()

    def loop_records(self, num_records=0, init_key=None):
        env = lmdb.open(self.fn, readonly=True)
        datum = Datum()
        with env.begin() as txn:
            cursor = txn.cursor()
            if init_key is not None:
                if not cursor.set_key(init_key):
                    raise ValueError('key ' + init_key + ' not found in lmdb ' + self.fn + '.')

            num_read = 0
            for key, value in cursor:
                datum.ParseFromString(value)
                label = datum.label
                data = datum_to_array(datum).squeeze()
                yield (data, label, key)
                num_read += 1
                if num_records != 0 and num_read == num_records:
                    break
        env.close()

    def loop_records_batch(self, batch_size, num_records=0, init_key=None):
        batch_data, batch_labels, batch_keys = [], [], []
        for dt, lb, ke in self.loop_records(num_records, init_key=init_key):
            (batch_data.append(dt), batch_labels.append(lb), batch_keys.append(ke))
            if len(batch_data) == batch_size:
                # Make sure yielded data is using new memory
                yield_data = np.concatenate([np.expand_dims(dt, axis=0) for dt in batch_data], axis=0)
                yield_labels = np.array(batch_labels)
                yield_keys = batch_keys[:]
                batch_data, batch_labels, batch_keys = [], [], []
                yield (yield_data, yield_labels, yield_keys)

        if len(batch_data) > 0:
            # Make sure yielded data is using new memory
            yield_data = np.concatenate([np.expand_dims(dt, axis=0) for dt in batch_data], axis=0)
            yield_labels = np.array(batch_labels)
            yield_keys = batch_keys[:]
            yield (yield_data, yield_labels, yield_keys)

    def load_records(self, num_records=0, init_key=None):
        if num_records == 0:
            num_records = self.num_records

        data, labels, keys = self.loop_records_batch(batch_size=num_records, num_records=num_records, init_key=init_key).next()
        return data, labels, keys


class CaffeDatumWriter(object):
    """
    CaffeDatumWriter: Object class to help with the creation of LMDB files storing Datum objects.
    """
    def __init__(self, fn, map_size, overwrite=False):
        self.fn = fn
        if os.path.exists(fn):
            if overwrite:
                shutil.rmtree(fn)
            else:
                raise ValueError(fn + ' already exists.')
        self.env = lmdb.open(fn, map_size=map_size)
        self.num = 0

    def __del__(self):
        self.env.close()

    def _add_record(self, data, label=None, key=None):
        data_dims = data.shape
        if data.ndim == 1:
            data_dims = np.array([data_dims[0], 1, 1], dtype=int)
        elif data.ndim == 2:
            data_dims = np.array([data_dims[0], data_dims[1], 1], dtype=int)

        datum = Datum()
        datum.channels, datum.height, datum.width = data_dims[0], data_dims[1], data_dims[2]
        if data.dtype == np.uint8:
            datum.data = data.tostring()
        else:
            datum.float_data.extend(data.tolist())
        datum.label = int(label) if label is not None else -1

        key = ('{:08}'.format(self.num) if key is None else key).encode('ascii')
        with self.env.begin(write=True) as txn:
            txn.put(key, datum.SerializeToString())
        self.num += 1

    def add_records(self, data, labels=None, keys=None, quiet=True):
        num = data.shape[0] if isinstance(data, np.ndarray) else len(data)
        labels = [-1]*num if labels is None else labels
        keys = [None for _ in range(num)] if keys is None else keys
        assert num == (len(labels) if not isinstance(labels, np.ndarray) else labels.shape[0])

        if not quiet:
            print 'Adding %d records to lmdb %s.' % (num, self.fn)
            sys.stdout.flush()

        for i, (dt, lb, ke) in enumerate(zip(data, labels, keys)):
            self._add_record(dt, label=lb, key=ke)

            if not quiet and i % 1000 == 0:
                print 'Samples saved: %5d/%5d' % (i, num)
                sys.stdout.flush()


def save_records_to_lmdb(lmdb_fn, data, labels=None, keys=None, overwrite=False):
    nSmpl = data.shape[0]
    map_size = (data.size + nSmpl)*32*2.0
    lmdb = CaffeDatumWriter(lmdb_fn, map_size, overwrite=overwrite)
    lmdb.add_records(data, labels=labels, keys=keys, quiet=False)
