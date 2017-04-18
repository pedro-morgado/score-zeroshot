import os
from caffe import layers as L, params as P

MODEL_DIR = os.path.dirname(os.path.realpath(__file__))

CONV_W_INIT = {'type': 'gaussian', 'std': 0.01}
CONV_B_INIT = {'type': 'constant', 'value': 0}
FC_W_INIT = {'type': 'gaussian', 'std': 0.001}
FC_B_INIT = {'type': 'constant', 'value': 0}


class AlexNet(object):
    def __init__(self, netspec):
        self.caffemodel = os.path.join(MODEL_DIR, 'alexnet.caffemodel')
        self.input_size = 227
        self.netspec = netspec

    def inference_proto(self, bottom, mult=1., truncate_at=None, deploy=False):
        ns = self.netspec
        w_params = dict(lr_mult=mult, decay_mult=mult)
        b_params = dict(lr_mult=mult, decay_mult=0)
        conv_opt_params = dict(weight_filler=CONV_W_INIT, bias_filler=CONV_B_INIT, param=[w_params, b_params]) if not deploy else {}
        fc_opt_params = dict(weight_filler=FC_W_INIT, bias_filler=FC_B_INIT, param=[w_params, b_params]) if not deploy else {}

        ns.conv1 = L.Convolution(bottom, num_output=96, kernel_size=11, stride=4, **conv_opt_params)
        ns.relu1 = L.ReLU(ns.conv1, in_place=True)
        ns.norm1 = L.LRN(ns.relu1, local_size=5, alpha=0.0001, beta=0.75)
        ns.pool1 = L.Pooling(ns.norm1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if truncate_at == 'pool1':
            return ns.pool1

        ns.conv2 = L.Convolution(ns.pool1, num_output=256, kernel_size=5, pad=2, group=2, **conv_opt_params)
        ns.relu2 = L.ReLU(ns.conv2, in_place=True)
        ns.norm2 = L.LRN(ns.relu2, local_size=5, alpha=0.0001, beta=0.75)
        ns.pool2 = L.Pooling(ns.norm2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if truncate_at == 'pool2':
            return ns.pool2

        ns.conv3 = L.Convolution(ns.pool2, num_output=384, kernel_size=3, pad=1, **conv_opt_params)
        ns.relu3 = L.ReLU(ns.conv3, in_place=True)
        if truncate_at == 'conv3':
            return ns.relu3

        ns.conv4 = L.Convolution(ns.relu3, num_output=384, kernel_size=3, pad=1, group=2, **conv_opt_params)
        ns.relu4 = L.ReLU(ns.conv4, in_place=True)
        if truncate_at == 'conv4':
            return ns.relu4

        ns.conv5 = L.Convolution(ns.relu4, num_output=256, kernel_size=3, pad=1, group=2, **conv_opt_params)
        ns.relu5 = L.ReLU(ns.conv5, in_place=True)
        ns.pool5 = L.Pooling(ns.relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if truncate_at == 'pool5':
            return ns.pool5

        ns.fc6 = L.InnerProduct(ns.pool5, num_output=4096, **fc_opt_params)
        ns.relu6 = L.ReLU(ns.fc6, in_place=True)
        ns.drop6 = L.Dropout(ns.relu6, dropout_ratio=0.5, in_place=True)
        if truncate_at == 'fc6':
            return ns.drop6

        ns.fc7 = L.InnerProduct(ns.drop6, num_output=4096, **fc_opt_params)
        ns.relu7 = L.ReLU(ns.fc7, in_place=True)
        ns.drop7 = L.Dropout(ns.relu7, dropout_ratio=0.5, in_place=True)
        if truncate_at == 'fc7':
            return ns.drop7

        ns.fc8 = L.InnerProduct(ns.fc7, num_output=1000, **fc_opt_params)
        return ns.fc8

    def load_pretrained(self, net):
        net.copy_from(self.caffemodel)


def test_net():
    import caffe
    import skimage.data as dt
    import skimage.transform as skt
    import numpy as np
    from matplotlib import pyplot as plt

    def pre_processing(img):
        if img.ndim == 2:   # Gray image
            img = np.concatenate([np.expand_dims(img, 2)]*3, 2)
        elif img.shape[2] > 3:  # RGB+depth or RGB+alpha. Ignore 4th channel
            img = img[:, :, :3]
        img = skt.resize(img, (256, 256)) * 255
        img = img[:, :, (2, 1, 0)]
        img = np.transpose(img, (2, 0, 1))
        img = img[:, 16:243, 16:243]
        img -= np.array([104, 117, 123]).reshape((3, 1, 1))
        return img

    # Load and process image
    img = dt.chelsea()
    img_prep = pre_processing(img)

    # Specify model
    ns = caffe.NetSpec()
    alexnet = AlexNet(ns)
    ns.data = L.Input(shape=dict(dim=[1, 3, 227, 227]))
    alexnet.inference_proto(ns.data, mult=1.)
    deploy_fn = '/tmp/deploy.prototxt'
    with open(deploy_fn, 'w') as f:
        f.write(str(ns.to_proto()))

    # Load pretrained model
    net = caffe.Net(deploy_fn, caffe.TEST)
    alexnet.load_pretrained(net)

    # Forward image and infer class
    net.blobs['data'].data[0] = img_prep
    net.forward()

    label = np.argmax(net.blobs['fc8'].data, axis=1)
    classes = [l.strip() for l in open('imagenet_synsets.txt')]
    print classes[label[0]]
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    test_net()
