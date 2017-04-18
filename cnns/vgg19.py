import os
from caffe import layers as L, params as P

MODEL_DIR = os.path.dirname(os.path.realpath(__file__))

CONV_W_INIT = {'type': 'gaussian', 'std': 0.01}
CONV_B_INIT = {'type': 'constant', 'value': 0}
FC_W_INIT = {'type': 'gaussian', 'std': 0.001}
FC_B_INIT = {'type': 'constant', 'value': 0}


class VGG19(object):
    def __init__(self, netspec):
        self.caffemodel = os.path.join(MODEL_DIR, 'vgg19.caffemodel')
        self.input_size = 224
        self.netspec = netspec

    def VGG_block(self, scale, bottom, num_filters, block_depth, mult=1.0, deploy=False):
        ns = self.netspec

        w_params = {'lr_mult': mult, 'decay_mult': mult}
        b_params = {'lr_mult': mult, 'decay_mult': 0}
        conv_opt_params = dict(weight_filler=CONV_W_INIT, bias_filler=CONV_B_INIT, param=[w_params, b_params]) if not deploy else {}

        x = bottom
        for d in range(1, block_depth+1):
            name = 'conv%d_%d' % (scale, d)
            x = ns[name] = L.Convolution(x, name=name, num_output=num_filters, kernel_size=3, pad=1, **conv_opt_params)
            name = 'relu%d_%d' % (scale, d)
            x = ns[name] = L.ReLU(x, name=name, in_place=True)

        return x

    def inference_proto(self, bottom, mult=1., truncate_at=None, deploy=False):
        ns = self.netspec
        w_params = {'lr_mult': mult, 'decay_mult': mult}
        b_params = {'lr_mult': mult, 'decay_mult': 0}
        fc_opt_params = dict(weight_filler=FC_W_INIT, bias_filler=FC_B_INIT, param=[w_params, b_params]) if not deploy else {}

        # Scale 1
        x = self.VGG_block(scale=1, bottom=bottom, num_filters=64, block_depth=2, mult=mult, deploy=deploy)
        x = ns.pool1 = L.Pooling(x, name='pool1', pool=P.Pooling.MAX, kernel_size=2, stride=2)
        if truncate_at == 'pool1':
            return x

        # Scale 2
        x = self.VGG_block(scale=2, bottom=x, num_filters=128, block_depth=2, mult=mult, deploy=deploy)
        x = ns.pool2 = L.Pooling(x, name='pool2', pool=P.Pooling.MAX, kernel_size=2, stride=2)
        if truncate_at == 'pool2':
            return x

        # Scale 3
        x = self.VGG_block(scale=3, bottom=x, num_filters=256, block_depth=4, mult=mult, deploy=deploy)
        x = ns.pool3 = L.Pooling(x, name='pool3', pool=P.Pooling.MAX, kernel_size=2, stride=2)
        if truncate_at == 'pool3':
            return x

        # Scale 4
        x = self.VGG_block(scale=4, bottom=x, num_filters=512, block_depth=4, mult=mult, deploy=deploy)
        x = ns.pool4 = L.Pooling(x, name='pool4', pool=P.Pooling.MAX, kernel_size=2, stride=2)
        if truncate_at == 'pool4':
            return x

        # Scale 5
        x = self.VGG_block(scale=5, bottom=x, num_filters=512, block_depth=4, mult=mult, deploy=deploy)
        x = ns.pool5 = L.Pooling(x, name='pool5', pool=P.Pooling.MAX, kernel_size=2, stride=2)
        if truncate_at == 'pool5':
            return x

        # FC 6
        x = ns.fc6 = L.InnerProduct(x, name='fc6', num_output=4096, **fc_opt_params)
        x = ns.relu6 = L.ReLU(x, name='relu6', in_place=True)
        x = ns.drop6 = L.Dropout(x, name='drop6', dropout_ratio=0.5, in_place=True)
        if truncate_at == 'fc6':
            return x

        # FC 7
        x = ns.fc7 = L.InnerProduct(x, name='fc7', num_output=4096, **fc_opt_params)
        x = ns.relu7 = L.ReLU(x, name='relu7', in_place=True)
        x = ns.drop7 = L.Dropout(x, name='drop7', dropout_ratio=0.5, in_place=True)
        if truncate_at == 'fc7':
            return ns.drop7

        # FC 8
        x = ns.fc8 = L.InnerProduct(x, name='fc8', num_output=1000, **fc_opt_params)
        return x

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
        img = img[:, 17:241, 17:241]
        img -= np.array([104, 117, 123]).reshape((3, 1, 1))
        return img

    # Load and process image
    img = dt.chelsea()
    img_prep = pre_processing(img)

    # Specify model
    ns = caffe.NetSpec()
    vgg = VGG19(ns)
    ns.data = L.Input(shape=dict(dim=[1, 3, 224, 224]))
    vgg.inference_proto(ns.data, mult=1.)
    deploy_fn = '/tmp/deploy.prototxt'
    with open(deploy_fn, 'w') as f:
        f.write(str(ns.to_proto()))

    # Load pretrained model
    net = caffe.Net(deploy_fn, caffe.TEST)
    vgg.load_pretrained(net)

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
