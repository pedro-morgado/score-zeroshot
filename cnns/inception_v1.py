import os
from caffe import layers as L, params as P

MODEL_DIR = os.path.dirname(os.path.realpath(__file__))

CONV_W_INIT = {'type': 'gaussian', 'std': 0.01}
CONV_B_INIT = {'type': 'constant', 'value': 0}
FC_W_INIT = {'type': 'gaussian', 'std': 0.001}
FC_B_INIT = {'type': 'constant', 'value': 0}


class InceptionV1(object):
    def __init__(self, netspec):
        self.caffemodel = os.path.join(MODEL_DIR, 'inception_v1.caffemodel')
        self.input_size = 224
        self.netspec = netspec

    def inception_v1_block(self, incp_name, bottom, c1, c3r, c3, c5r, c5, cp, mult=1., deploy=False):
        ns = self.netspec

        w_params = {'lr_mult': mult, 'decay_mult': mult}
        b_params = {'lr_mult': mult, 'decay_mult': 0}
        conv_opt_params = dict(weight_filler=CONV_W_INIT, bias_filler=CONV_B_INIT, param=[w_params, b_params]) if not deploy else {}

        # 1x1 branch
        x1 = ns[incp_name+'/1x1'] = L.Convolution(bottom, num_output=c1, kernel_size=1, **conv_opt_params)
        x1 = ns[incp_name+'/relu_1x1'] = L.ReLU(x1, in_place=True)

        # 3x3 branch
        x3 = ns[incp_name+'/3x3_reduce'] = L.Convolution(bottom, num_output=c3r, kernel_size=1, **conv_opt_params)
        x3 = ns[incp_name+'/relu_3x3_reduce'] = L.ReLU(x3, in_place=True)
        x3 = ns[incp_name+'/3x3'] = L.Convolution(x3, num_output=c3, kernel_size=3, pad=1, **conv_opt_params)
        x3 = ns[incp_name+'/relu_3x3'] = L.ReLU(x3, in_place=True)

        # 5x5 branch
        x5 = ns[incp_name+'/5x5_reduce'] = L.Convolution(bottom, num_output=c5r, kernel_size=1, **conv_opt_params)
        x5 = ns[incp_name+'/relu_5x5_reduce'] = L.ReLU(x5, in_place=True)
        x5 = ns[incp_name+'/5x5'] = L.Convolution(x5, num_output=c5, kernel_size=5, pad=2, **conv_opt_params)
        x5 = ns[incp_name+'/relu_5x5'] = L.ReLU(x5, in_place=True)

        # Pooling branch
        xp = ns[incp_name+'/pool'] = L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=3, pad=1)
        xp = ns[incp_name+'/pool_proj'] = L.Convolution(xp, num_output=cp, kernel_size=1, **conv_opt_params)
        xp = ns[incp_name+'/relu_pool_proj'] = L.ReLU(xp, in_place=True)

        outp = ns[incp_name+'/output'] = L.Concat(*[x1, x3, x5, xp])
        return outp

    def inference_proto(self, bottom, mult=1., truncate_at=None, deploy=False):
        ns = self.netspec
        w_params = {'lr_mult': mult, 'decay_mult': mult}
        b_params = {'lr_mult': mult, 'decay_mult': 0}
        conv_opt_params = dict(weight_filler=CONV_W_INIT, bias_filler=CONV_B_INIT, param=[w_params, b_params]) if not deploy else {}
        fc_opt_params = dict(weight_filler=FC_W_INIT, bias_filler=FC_B_INIT, param=[w_params, b_params]) if not deploy else {}

        # 1st Scale
        x = ns['conv1/7x7_s2'] = L.Convolution(bottom, num_output=64, kernel_size=7, stride=2, pad=3, **conv_opt_params)
        x = ns['conv1/relu_7x7'] = L.ReLU(x, in_place=True)
        if truncate_at == 'conv1':
            return x

        x = ns['pool1/3x3_s2'] = L.Pooling(x, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        x = ns['pool1/norm1'] = L.LRN(x, local_size=5, alpha=0.0001, beta=0.75)
        if truncate_at == 'pool1':
            return x

        # 2nd Scale
        x = ns['conv2/3x3_reduce'] = L.Convolution(x, num_output=64, kernel_size=1, **conv_opt_params)
        x = ns['conv2/relu_3x3_reduce'] = L.ReLU(x, in_place=True)
        x = ns['conv2/3x3'] = L.Convolution(x, num_output=192, kernel_size=3, pad=1, **conv_opt_params)
        x = ns['conv2/relu_3x3'] = L.ReLU(x, in_place=True)
        if truncate_at == 'conv2':
            return x

        x = ns['conv2/norm2'] = L.LRN(x, local_size=5, alpha=0.0001, beta=0.75)
        x = ns['pool2/3x3_s2'] = L.Pooling(x, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if truncate_at == 'pool2':
            return x

        # 3rd scale
        block = {'c1': [64, 128],
                 'c3r': [96, 128],
                 'c3': [128, 192],
                 'c5r': [16, 32],
                 'c5': [32, 96],
                 'cp': [32, 64],
                 'name': ['inception_3a', 'inception_3b']}
        for i, nm in enumerate(block['name']):
            x = self.inception_v1_block(nm, x,
                                        c1=block['c1'][i],
                                        c3r=block['c3r'][i],
                                        c3=block['c3'][i],
                                        c5r=block['c5r'][i],
                                        c5=block['c5'][i],
                                        cp=block['cp'][i],
                                        mult=mult, deploy=deploy)
            if truncate_at == block['name'][i]:
                return x

        # Pooling
        x = ns['pool3/3x3_s2'] = L.Pooling(x, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if truncate_at == 'pool3':
            return x

        # 4th scale
        block = {'c1': [192, 160, 128, 112, 256],
                 'c3r': [96, 112, 128, 144, 160],
                 'c3': [208, 224, 256, 288, 320],
                 'c5r': [16, 24, 24, 32, 32],
                 'c5': [48, 64, 64, 64, 128],
                 'cp': [64, 64, 64, 64, 128],
                 'name': ['inception_%s' % s for s in ['4a', '4b', '4c', '4d', '4e']]}
        for i, nm in enumerate(block['name']):
            x = self.inception_v1_block(nm, x,
                                        c1=block['c1'][i],
                                        c3r=block['c3r'][i],
                                        c3=block['c3'][i],
                                        c5r=block['c5r'][i],
                                        c5=block['c5'][i],
                                        cp=block['cp'][i],
                                        mult=mult, deploy=deploy)
            if truncate_at == block['name'][i]:
                return x

        # Pooling
        x = ns['pool4/3x3_s2'] = L.Pooling(x, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        if truncate_at == 'pool4':
            return x

        # 5rd scale
        block = {'c1': [256, 384],
                 'c3r': [160, 192],
                 'c3': [320, 384],
                 'c5r': [32, 48],
                 'c5': [128, 128],
                 'cp': [128, 128],
                 'name': ['inception_5a', 'inception_5b']}
        for i, nm in enumerate(block['name']):
            x = self.inception_v1_block(nm, x,
                                        c1=block['c1'][i],
                                        c3r=block['c3r'][i],
                                        c3=block['c3'][i],
                                        c5r=block['c5r'][i],
                                        c5=block['c5'][i],
                                        cp=block['cp'][i],
                                        mult=mult, deploy=deploy)
            if truncate_at == block['name'][i]:
                return x

        # Pooling
        x = ns['pool5/7x7_s1'] = L.Pooling(x, pool=P.Pooling.AVE, kernel_size=7)
        x = ns['pool5/drop_7x7_s1'] = L.Dropout(x, dropout_ratio=0.4, in_place=True)
        if truncate_at == 'pool5':
            return x

        # Classifier
        x = ns['loss3/classifier'] = L.InnerProduct(x, num_output=1000, **fc_opt_params)
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
        img = img[:, 16:243, 16:243]
        img -= np.array([104, 117, 123]).reshape((3, 1, 1))
        return img

    # Load and process image
    img = dt.chelsea()
    img_prep = pre_processing(img)

    # Specify model
    ns = caffe.NetSpec()
    goog = InceptionV1(ns)
    ns.data = L.Input(shape=dict(dim=[1, 3, 227, 227]))
    goog.inference_proto(ns.data, mult=1.)
    deploy_fn = '/tmp/deploy.prototxt'
    with open(deploy_fn, 'w') as f:
        f.write(str(ns.to_proto()))

    # Load pretrained model
    net = caffe.Net(deploy_fn, caffe.TEST)
    goog.load_pretrained(net)

    # Forward image and infer class
    net.blobs['data'].data[0] = img_prep
    net.forward()

    label = np.argmax(net.blobs['loss3/classifier'].data, axis=1)
    classes = [l.strip() for l in open('imagenet_synsets.txt')]
    print classes[label[0]]
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    test_net()


