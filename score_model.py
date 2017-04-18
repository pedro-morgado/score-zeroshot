import os
import numpy as np
import caffe
from caffe import layers as L, params as P
from tools.dataset_parameters import ATTRIBUTES, HIERARCHY, WORD2VEC

CONV_W_INIT = {'type': 'gaussian', 'std': 0.01}
CONV_B_INIT = {'type': 'constant', 'value': 0}
FC_W_INIT = {'type': 'gaussian', 'std': 0.001}
FC_B_INIT = {'type': 'constant', 'value': 0}


class TrainingOpts(object):
    def __init__(self, iters=1000, init_lr=0.01, lr_decay_factor=0.5, num_lr_decays=2, paramReg=0.0005):
        self.iters = iters
        self.init_lr = init_lr
        self.lr_decay_factor = lr_decay_factor
        self.num_lr_decays = num_lr_decays
        self.paramReg = paramReg


class SCoRe(object):
    def __init__(self,
                 source_classes,        # List of training classes
                 target_classes,        # List of zero-shot classes
                 semantics,             # Semantics (string)
                 constrains,            # Dictionary of semantic concepts
                 cnn,                   # CNN to be used (string)
                 sem_coeff=0.0,         # Lagrangian coefficient associated with semantic loss
                 code_coeff=np.inf):    # Lagrangian coefficient associated with codeword regularization
        self.train_classes = source_classes
        self.test_classes = target_classes
        self.semantics = semantics
        self.constrains = constrains

        # Instead of loss L1 + gamma L2, we use the equivalent loss 1/(1+gamma)L1 + gamma/(1+gamma) L2.
        # This allows gamma to be large, without having to worry about decreasing the learning rate.
        self.sem_coeff = sem_coeff / (1.+sem_coeff) if sem_coeff is not None and sem_coeff < np.inf else 1.
        self.code_coeff = code_coeff

        self.code_dim = [const['codewords'].shape[0] for const in constrains.itervalues()]
        self.num_states = [const['codewords'].shape[1] for const in constrains.itervalues()]

        self.cnn = cnn
        self.netspec = None
        self.deploy = None
        self.solver = None
        self.scores = {'classes': None, 'semantics': None}

        assert cnn in ('GoogLeNet', 'AlexNet', 'VGG19')
        assert semantics in (ATTRIBUTES, HIERARCHY, WORD2VEC)

    def _new_model(self):
        self.netspec = caffe.NetSpec()
        if self.cnn == 'AlexNet':
            from cnns import AlexNet
            self.base_cnn = AlexNet(netspec=self.netspec)
            self.feat_layer = 'fc7'
            self.feat_dim = 4096
        elif self.cnn == 'GoogLeNet':
            from cnns import InceptionV1
            self.base_cnn = InceptionV1(netspec=self.netspec)
            self.feat_layer = 'pool5'
            self.feat_dim = 1024
        elif self.cnn == 'VGG19':
            from cnns import VGG19
            self.base_cnn = VGG19(netspec=self.netspec)
            self.feat_layer = 'fc7'
            self.feat_dim = 4096
        return self.netspec

    def _semantic_regularization(self, xSemPr, xSemLb, semReg):
        ns = self.netspec

        if self.semantics == ATTRIBUTES:
            name = 'SCoRe/semLoss'
            ns[name] = L.SigmoidCrossEntropyLoss(*[xSemPr, xSemLb], name=name,
                                                 loss_weight=semReg/(len(self.constrains)*np.sqrt(2.))*10.,
                                                 include=dict(phase=caffe.TRAIN))
        else:
            c_keys = [key for key in self.constrains.keys()]
            losses = ['SCoRe/semLoss/%s' % key for key in c_keys]
            scores = ['SCoRe/semLoss/%s/scores' % key for key in c_keys]
            labels = ['SCoRe/semLoss/%s/labels' % key for key in c_keys]

            # Slice semantic scores
            xSemPr_name = [k for k, v in ns.tops.iteritems() if v ==xSemPr][0]
            slice_scores = L.Slice(name='SCoRe/semLoss/slice_scores', bottom=[xSemPr_name], ntop=len(scores), top=scores, in_place=True,
                                   slice_point=np.cumsum(self.num_states)[:-1].tolist(),
                                   include=dict(phase=caffe.TRAIN))

            # Slice semantic labels
            xSemLb_name = [k for k, v in ns.tops.iteritems() if v ==xSemLb][0]
            slice_labels = L.Slice(name='SCoRe/semLoss/slice_labels', bottom=[xSemLb_name], ntop=len(labels), top=labels, in_place=True,
                                   slice_point=range(1, len(self.constrains)),
                                   include=dict(phase=caffe.TRAIN))

            # Add supervision to each slice
            for i, xLoss in enumerate(losses):
                ns[xLoss] = L.SoftmaxWithLoss(*[slice_scores[i], slice_labels[i]], name=xLoss, loss_weight=semReg/len(self.constrains),
                                              include=dict(phase=caffe.TRAIN))

            # Summarize supervisions for display
            ns['SCoRe/semLoss'] = L.Eltwise(*[ns[l] for l in losses], name='SCoRe/semLoss',
                                            operation=P.Eltwise.SUM, coeff=[semReg/len(self.constrains)]*len(losses),
                                            include=dict(phase=caffe.TRAIN))

    def _code_regularization(self, lCW):
        ns = self.netspec

        # Semantic codes. Needs to be initialized.
        code_shape = [sum(self.code_dim), len(self.train_classes) if self.semantics == ATTRIBUTES else sum(self.num_states)]

        name = 'SCoRe/cwReg/codewords'
        sem_cw = ns[name] = L.DummyData(name=name, shape=dict(dim=code_shape), include=dict(phase=caffe.TRAIN))

        # Classification codes.
        name = 'SCoRe/cwReg/eye'
        x = ns[name] = L.DummyData(name=name, shape=dict(dim=[code_shape[0], code_shape[0]]), include=dict(phase=caffe.TRAIN))

        name = 'SCoRe/cwReg/cls_codewords'
        clf_cw = ns[name] = L.InnerProduct(x, name=name, num_output=code_shape[1], bias_term=False,
                                           param=[{'name': lCW}], include=dict(phase=caffe.TRAIN))

        # Compute \sum |S-C|^2
        name = 'SCoRe/cwReg/diff'
        x_diff = ns[name] = L.Eltwise(*[sem_cw, clf_cw], name=name,
                                      operation=P.Eltwise.SUM, coeff=[1., -1.], include=dict(phase=caffe.TRAIN))

        name = 'SCoRe/cwReg'
        ns[name] = L.Reduction(x_diff, name=name,
                               operation=P.Reduction.SUMSQ, axis=0,
                               loss_weight=self.code_coeff, include=dict(phase=caffe.TRAIN))

    def _score_proto(self, xFeat, source_net=False, target_net=False, mult=1., deploy=False):
        from caffe.proto import caffe_pb2
        ns = self.netspec
        w_params = {'lr_mult': mult, 'decay_mult': mult}

        # Compute semantic space
        name = 'SCoRe/sem/fc1'
        layer_params = dict(weight_filler=FC_W_INIT, param=[w_params]) if not deploy else {}
        x = ns[name] = L.InnerProduct(xFeat, name=name, num_output=sum(self.code_dim), bias_term=False, **layer_params)

        # Note: In the case of completely binary semantics (Attributes), the two layers codewords+selector are compressed in 'SCoRe/obj/fc'.
        # Otherwise, semantic state scores are first computed in SCoRe/sem/fc2 and then grouped into class scores using a selector in SCoRe/obj/fc.
        # The selector is always kept fixed, and the codewords are learned whenever code_coeff<inf.
        xSem = 'SCoRe/sem/fc1' if self.semantics == ATTRIBUTES else 'SCoRe/sem/fc2'
        xObj = 'SCoRe/obj/fc'
        lCW = xObj + '/params' if self.semantics == ATTRIBUTES else xSem + '/params'
        if self.semantics != ATTRIBUTES:
            w_params = {'name': xSem+'/params',
                        'share_mode': caffe_pb2.ParamSpec.STRICT,
                        'lr_mult': mult if self.code_coeff < np.inf else 0.0,       # Lock weights if code_coeff is inf
                        'decay_mult': mult if self.code_coeff < np.inf else 0.0}
            layer_params = dict(weight_filler=FC_W_INIT, param=[w_params]) if not deploy else {}
            ns[xSem] = L.InnerProduct(x, name=xSem, num_output=sum(self.num_states), bias_term=False, **layer_params)

        # Compute object scores
        if source_net:
            w_params = {'name': xObj+'/params',
                        'share_mode': caffe_pb2.ParamSpec.STRICT,
                        'lr_mult': mult if self.code_coeff < np.inf and self.semantics == ATTRIBUTES else 0.0,     # If Attributes than codewords are used in this layer
                        'decay_mult': mult if self.code_coeff < np.inf and self.semantics == ATTRIBUTES else 0.0}  # Lock weights if code_coeff is inf
            layer_params = dict(weight_filler=FC_W_INIT, param=[w_params],
                                include=dict(not_stage='TestZeroShot')) if not deploy else {}
            ns[xObj] = L.InnerProduct(ns[xSem], name=xObj, num_output=len(self.train_classes), bias_term=False, **layer_params)

        if target_net:
            name = xObj+'_target'
            w_params = {'name': name+'/params', 'share_mode': caffe_pb2.ParamSpec.STRICT,
                        'lr_mult': 0.0, 'decay_mult': 0.0}
            layer_params = dict(weight_filler=FC_W_INIT, param=[w_params],
                                include=dict(phase=caffe.TEST, stage='TestZeroShot')) if not deploy else {}

            # NetSpec cannot handle two layers with same top blob defined for different phases/stages.
            # Workaround: Set in_place=True with no inputs, then define bottom and top fields manually.
            ns[name] = L.InnerProduct(name=name, bottom=[xSem], ntop=1, top=[xObj], in_place=True,
                                      num_output=len(self.test_classes), bias_term=False, **layer_params)
        return xObj, xSem, lCW

    @staticmethod
    def generate_solver_proto(solver_fn, model_fn, trainOpts):
        from caffe.proto import caffe_pb2
        solver = caffe_pb2.SolverParameter()
        solver.net = model_fn

        if trainOpts.num_lr_decays > 0:
            solver.lr_policy = 'step'
            solver.gamma = trainOpts.lr_decay_factor
            solver.stepsize = int(trainOpts.iters/(trainOpts.num_lr_decays+1))
        else:
            solver.lr_policy = 'fixed'
        solver.base_lr = trainOpts.init_lr
        solver.max_iter = trainOpts.iters
        solver.display = 20
        solver.momentum = 0.9
        solver.weight_decay = trainOpts.paramReg

        solver.test_state.add()
        solver.test_state.add()
        solver.test_state[0].stage.append('TestRecognition')
        solver.test_state[1].stage.append('TestZeroShot')
        solver.test_iter.extend([20, 20])
        solver.test_interval = 100

        solver.snapshot = 5000
        solver.snapshot_prefix = os.path.splitext(model_fn)[0]

        with open(solver_fn, 'w') as f:
            f.write(str(solver))

    def _set_semantics(self, net, source=False, init_cw=False):
        # Semantic codewords
        constrain_cw = [const['codewords'] for const in self.constrains.itervalues()]
        codes, c1, c2 = np.zeros((sum(self.code_dim), sum(self.num_states))), 0, 0
        for cw in constrain_cw:
            codes[c1:c1+cw.shape[0], c2:c2+cw.shape[1]], c1, c2 = cw, c1+cw.shape[0], c2+cw.shape[1]

        if source:
            # Codeword selector for training classes
            selector = np.concatenate([const['labels']['source'] for const in self.constrains.itervalues()], axis=0)

            if self.semantics == ATTRIBUTES:
                cw = np.dot(codes, selector)
                cw /= ((cw**2).sum(axis=0, keepdims=True))**0.5
                if init_cw:
                    _set_weight_params(net, 'SCoRe/obj/fc', cw.T)
                _set_blob(net, 'SCoRe/cwReg/codewords', cw)

            else:
                if init_cw:
                    _set_weight_params(net, 'SCoRe/sem/fc2', codes.T)
                _set_blob(net, 'SCoRe/cwReg/codewords', codes)
                selector /= ((np.dot(codes, selector)**2).sum(axis=0, keepdims=True))**0.5
                _set_weight_params(net, 'SCoRe/obj/fc', selector.T)

            _set_blob(net, 'SCoRe/cwReg/eye', np.eye(sum(self.code_dim)))

        else:
            # Codeword selector for test classes
            selector = np.concatenate([const['labels']['target'] for const in self.constrains.itervalues()], axis=0)

            if self.semantics == ATTRIBUTES:
                cw = np.dot(codes, selector)
                cw /= ((cw**2).sum(axis=0, keepdims=True))**0.5
                _set_weight_params(net, 'SCoRe/obj/fc_target', cw.T)
            else:
                selector /= ((np.dot(codes, selector)**2).sum(axis=0, keepdims=True))**0.5
                _set_weight_params(net, 'SCoRe/obj/fc_target', selector.T)

    # Methods for deploy mode
    def generate_deploy_proto(self, deploy_fn, batch_size, source_net=False, target_net=False):
        ns = self._new_model()
        batch_size = batch_size

        # Inputs
        img_dim = self.base_cnn.input_size
        ns.data = L.Input(shape=dict(dim=[batch_size, 3, img_dim, img_dim]))

        # Inference
        xFt = self.base_cnn.inference_proto(ns.data, truncate_at=self.feat_layer, deploy=True)
        xObj, xSem, _ = self._score_proto(xFt, source_net=source_net, target_net=target_net, deploy=True)
        self.scores = {'obj': xObj, 'semantics': xSem}

        with open(deploy_fn, 'w') as f:
            f.write(str(ns.to_proto()))

    def prep_for_deploy(self, batch_size, source_net=False, target_net=False, deploy_fn='deploy.proto', caffemodel_fn='score.caffemodel', gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.generate_deploy_proto(deploy_fn, batch_size, source_net=source_net, target_net=target_net)
        self.deploy = caffe.Net(deploy_fn, caffe.TEST, weights=caffemodel_fn)

        self._set_semantics(self.deploy, source=False, init_cw=False)
        self._set_semantics(self.deploy, source=True, init_cw=False)

    def _forward_batch(self, data, blobs=None):
        if isinstance(blobs, str):
            blobs = [blobs]
        nImgs = len(data) if isinstance(data, list) else data.shape[0]

        self.deploy.blobs['data'].data[:nImgs] = data
        self.deploy.forward()
        return {blb: np.copy(self.deploy.blobs[blb].data[:nImgs, ]) for blb in blobs}

    def forward(self, data, blobs=None):
        batch_size = self.deploy.blobs['data'].data.shape[0]
        nImgs = len(data) if isinstance(data, list) else data.shape[0]
        batch_lims = range(0, nImgs-1, batch_size)+[nImgs]
        blobs = list(set(blobs))
        outp = {blb: [] for blb in blobs}
        for i in range(len(batch_lims)-1):
            tmp = self._forward_batch(data[batch_lims[i]:batch_lims[i+1]], blobs=blobs)
            for blb in blobs:
                outp[blb].append(tmp[blb])
        for blb in blobs:
            outp[blb] = np.concatenate(outp[blb], axis=0) if len(outp[blb]) > 1 else outp[blb][0]
        return outp

    # Methods for training mode
    def _loss_proto(self, xPr, xLb, xSemPr, xSemLb, lCW):
        ns = self.netspec

        # Classification loss
        if self.sem_coeff < 1:
            name = 'SCoRe/objLoss'
            ns[name] = L.SoftmaxWithLoss(*[xPr, xLb], name=name, loss_weight=1.0 - self.sem_coeff, include=dict(phase=caffe.TRAIN))

        # Semantic regularization
        if self.sem_coeff > 0:
            self._semantic_regularization(xSemPr, xSemLb, self.sem_coeff)

        # Codeword regularization
        if 0 < self.code_coeff < np.inf:
            self._code_regularization(lCW)

    def _eval_proto(self, xObj, xLb):
        ns = self.netspec
        name = 'SCoRe/eval/obj/accuracy'
        ns[name] = L.Accuracy(*[xObj, xLb], name=name)

    def generate_train_proto(self, model_fn, fts_lmdb, sem_lmdb, batch_size):
        ns = self._new_model()

        # Inputs
        mean = [104., 116., 122.]
        stage = {'testRecg': 'TestRecognition',
                 'testZS': 'TestZeroShot'}
        for subset in ['train', 'testRecg', 'testZS']:
            if subset == 'train':
                include = {'phase': caffe.TRAIN}
            else:
                include = {'phase': caffe.TEST, 'stage': stage[subset]}
            ns[subset+'_data'], ns[subset+'_labels'] = L.Data(name='data', ntop=2, top=['data', 'labels'], in_place=True,
                                                              source=fts_lmdb[subset], batch_size=batch_size, backend=P.Data.LMDB,
                                                              transform_param=dict(mirror=True if subset == 'train' else False,
                                                                                   crop_size=self.base_cnn.input_size,
                                                                                   mean_value=mean),
                                                              include=include)

        # Semantic labels for training
        if self.sem_coeff > 0:
            ns.semantics = L.Data(name='semantics',
                                  source=sem_lmdb['train'], batch_size=batch_size, backend=P.Data.LMDB,
                                  include=dict(phase=caffe.TRAIN))

        # Run base CNN
        xFt = self.base_cnn.inference_proto(ns.train_data, mult=1., truncate_at=self.feat_layer)

        # Run score
        xObj, xSem, lCW = self._score_proto(xFt, source_net=True, target_net=self.test_classes is not None, mult=1.0)
        self.scores = {'obj': xObj, 'semantics': xSem}

        # Loss
        self._loss_proto(ns[xObj], ns.train_labels, ns[xSem], ns.semantics if self.sem_coeff > 0 else None, lCW)

        # Evaluation
        self._eval_proto(ns[xObj], ns.train_labels)

        with open(model_fn, 'w') as f:
            f.write(str(ns.to_proto()))

    def prep_for_training(self, model_fn, solver_fn, dt_lmdbs, sem_lmdbs, trainOpts, batch_size, gpu_id):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.generate_train_proto(model_fn, dt_lmdbs, sem_lmdbs, batch_size)
        self.generate_solver_proto(solver_fn, model_fn, trainOpts=trainOpts)

        solver = caffe.NesterovSolver(solver_fn)
        self.base_cnn.load_pretrained(solver.net)
        self._set_semantics(solver.net, source=True, init_cw=True)
        self._set_semantics(solver.test_nets[1], source=False, init_cw=True)

        self.solver = solver

    def train(self, n=1):
        self.solver.step(n)

    def save(self, model_fn):
        self.solver.net.save(model_fn)

    def load(self, model_fn):
        self.solver.net.copy_from(model_fn)


def _set_weight_params(net, name, weigths):
    if name in net.params.keys():
        net.params[name][0].data[...] = weigths
        print 'Layer %s (weights) initialized' % name


def _set_bias_params(net, name, bias):
    if name in net.params.keys():
        net.params[name][1].data[...] = bias
        print 'Layer %s (biases) initialized' % name


def _set_blob(net, name, data):
    if name in net.blobs.keys():
        net.blobs[name].data[...] = data
        print 'Blob %s initialized' % name

