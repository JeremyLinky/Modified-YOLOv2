from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
import gc
import numpy as np
from memory_profiler import profile

class YOLOv2:
    def __init__(self, config, data_provider):

        assert len(config['data_shape']) == 3
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.data_shape = config['data_shape']
        self.num_classes = config['num_classes']
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']        # 保持Dropout活性的概率
        self.data_format = config['data_format']
        self.mode = config['mode']                  # train or test
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1  # test：batch_size开1

        self.coord_sacle = config['coord_scale']    # coefficient of bbox locations and sizes loss:1
        self.noobj_scale = config['noobj_scale']    # coefficient of noobj loss:1
        self.obj_scale = config['obj_scale']        # coefficient of obj loss:5
        self.class_scale = config['class_scale']    # coefficient of class loss:1

        self.nms_score_threshold = config['nms_score_threshold']    # prob:0.5
        self.nms_max_boxes = config['nms_max_boxes']                # num_boxes:10
        self.nms_iou_threshold = config['nms_iou_threshold']        # iou:0.5
        self.rescore_confidence = config['rescore_confidence']      # False

        self.path_backone = config['path_backone']
        self.path_head = config['path_head']
        self.pretrain = config['is_pretraining']
        self.num_priors = len(config['priors'])
        priors = tf.convert_to_tensor(config['priors'], dtype=tf.float32)
        self.priors = tf.reshape(priors, [1, 1, self.num_priors, 2])    # 1*1*5*2
        self.final_units = (self.num_classes + 5) * self.num_priors     # anchor:5  (classes+5):类别数+每个anchor5个输出
        self.num_gpu = config['num_gpu']

        if self.mode == 'train':
            self.num_train = data_provider['num_train']     #5011 images
            self.num_val = data_provider['num_val']         #0 images
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator   #返回数据集迭代器以及迭代器的初始化操作
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator

        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True
        self.input_images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None, 3],
                                           name='input_images')
        self.anchor_shape = tf.placeholder(dtype=tf.int64, shape=[], name='anchor_shape')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')  # 设置 lr 为训练时的输入
        self.define_inputs()

        if self.mode == 'train':
            self._build_graph()
        else:
            self._test_graph()

        if self.mode == 'train':
            self._create_summary()
        self._init_session()    # 构建好计算图之后最后初始化
        self._create_saver(self.path_backone, self.path_head)

    @profile
    def define_inputs(self):
        self.shape = [self.batch_size]
        self.shape.extend(self.data_shape)       # [batch,data_shape]
        self.mean = tf.convert_to_tensor([114.375, 108.375, 99.96], dtype=tf.float32)       # VOC2007像素均值
        self.std = tf.convert_to_tensor([61.186, 59.753, 60.371], dtype=tf.float32)
        if self.data_format == 'channels_last':
            self.mean = tf.reshape(self.mean, [1, 1, 1, 3])   # 1*1*1*3
        else:
            self.mean = tf.reshape(self.mean, [1, 3, 1, 1])   # 1*3*1*1
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()     # 迭代获取下一组数据
            self.images = (self.images - self.mean)/self.std    # normilazation
            self.images.set_shape(self.shape)              # resize
        else:
            self.images = tf.placeholder(tf.float32, self.shape, name='images')
            self.images = (self.images - self.mean) / self.std

    def change_iterator(self, train_gen):
        self.train_generator = train_gen
        self.train_initializer, self.train_iterator = self.train_generator

    def _build_graph(self):
        self.grad = []
        for i in range(self.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(('backone'), reuse=tf.compat.v1.AUTO_REUSE):      # get feature extractor
                    features, passthrough, downsampling_rate = self._feature_extractor(self.input_images)
                with tf.variable_scope(('head'), reuse=tf.compat.v1.AUTO_REUSE):    # get yolov2 head
                    conv1 = self._conv_layer(features, 1024, 3, 1, 'conv1')     # 23
                    lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
                    conv2 = self._conv_layer(lrelu1, 1024, 3, 1, 'conv2')       # 24
                    lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')

                    conv_passthrough = self._conv_layer(passthrough, 64, 1, 1, 'conv_pass')
                    lrelu_passthrough = tf.nn.leaky_relu(conv_passthrough, 0.1, 'lrelu_pass')

                    passthrough = tf.reshape(lrelu_passthrough, [self.batch_size, self.anchor_shape,
                                                                 self.anchor_shape, 256], 'passthrough')

                    axes = 3 if self.data_format == 'channels_last' else 1
                    concatation = tf.concat([passthrough, lrelu2], axis=axes)    # output and passthrough concat

                    conv3 = self._conv_layer(concatation, 1024, 3, 1, 'conv3')
                    lrelu3 = tf.nn.leaky_relu(conv3, 0.1, name='lrelu3')
                    pred = self._conv_layer(lrelu3, self.final_units, 1, 1, 'predictions')
                    # 全卷积的输出层：final_units对应着anchor输出, 13*13*125  125=5(20+5)

                    if self.data_format == 'channels_first':
                        pred = tf.transpose(pred, [0, 2, 3, 1])     # 0维：数量  2 3 1 使channel置于最后
                    pshape = tf.shape(pred)     # 输出张量维度

                    pred = tf.reshape(pred, [pshape[0], pshape[1], pshape[2], self.num_priors, -1])
                    # num_priors:暂时为5个anchor  -1维度:即是 classes+5
                    pclass = tf.nn.softmax(pred[..., :self.num_classes], axis=-1)         # 每个class的概率
                    pbbox_yx = pred[..., self.num_classes:self.num_classes + 2]     # ty tx
                    pbbox_hw = pred[..., self.num_classes + 2:self.num_classes + 4] # th tw
                    pobj = pred[..., self.num_classes + 4:]                         # t0
                    abbox_yx, abbox_hw, abbox_y1x1, abbox_y2x2 = self._get_priors(pshape, self.priors)
                    # 13*13个cell中 5个anchor的（center hw tl br） 基于13*13
                self.total_loss = []
                for i in range(self.batch_size):
                    gn_yxi, gn_hwi, gn_labeli = self._get_normlized_gn(downsampling_rate, i)    # gt的（center hw class）
                    gn_floor_ = tf.cast(tf.floor(gn_yxi), tf.int64)                             # gt的yx向下取整 得到gt中心点所在的cell
                    with tf.device('/cpu:0'):
                        nogn_mask = tf.sparse.SparseTensor(gn_floor_, tf.ones_like(gn_floor_[..., 0]),
                                                           dense_shape=[pshape[1], pshape[2]])
                        # dense_shape: 生成的矩阵大小  gt中心所在的cell:值为非0的下标  tf.ones_like:对应值设为1
                        nogn_mask = (1 - tf.sparse.to_dense(nogn_mask, validate_indices=False)) > 0
                        # 原来为0：即gt为0的cell的部分构成nogn_mask  即是gt中没有label的中心点的区域
                    rpbbox_yx = tf.gather_nd(pbbox_yx[i, ...], gn_floor_)   # 取预测值yx中gt有的部分
                    rpbbox_hw = tf.gather_nd(pbbox_hw[i, ...], gn_floor_)   # 取预测值hw中gt有的部分
                    rpclass = tf.gather_nd(pclass[i, ...], gn_floor_)       # 取预测值中gt有的类别的概率
                    rpobj = tf.gather_nd(pobj[i, ...], gn_floor_)           # 取预测值的置信度的部分

                    rabbox_hw = tf.gather_nd(abbox_hw, gn_floor_)           # 取anchor中hw gt有的部分
                    rabbox_y1x1 = tf.gather_nd(abbox_y1x1, gn_floor_)       # 取anchor中y1x1 gt有的部分
                    rabbox_y2x2 = tf.gather_nd(abbox_y2x2, gn_floor_)       # 取anchor中y2x2 gt有的部分

                    gn_y1x1i = tf.expand_dims(gn_yxi - gn_hwi / 2., axis=1)
                    gn_y2x2i = tf.expand_dims(gn_yxi + gn_hwi / 2., axis=1)
                    rgaiou_y1x1 = tf.maximum(gn_y1x1i, rabbox_y1x1)         # 取最右下的左上角
                    rgaiou_y2x2 = tf.minimum(gn_y2x2i, rabbox_y2x2)         # 取最左上的右下角

                    rgaiou_area = tf.reduce_prod(rgaiou_y2x2 - rgaiou_y1x1, axis=-1)    # 计算交集

                    garea = tf.reduce_prod(gn_y2x2i-gn_y1x1i, axis=-1)                  # GT的面积
                    aarea = tf.reduce_prod(rabbox_hw, axis=-1)                          # 预测的面积
                    rgaiou = rgaiou_area / (aarea + garea - rgaiou_area)                # 计算iou:定值

                    # 以上部分为计算负责该gt的cell的 gt与anchor 的最大iou

                    rgaindex = tf.expand_dims(tf.cast(tf.argmax(rgaiou, axis=-1), tf.int32), -1)    # 取每个cell最大值iou的anchor下标
                    rgaindex = tf.concat([tf.expand_dims(tf.range(tf.shape(rgaindex)[0]), -1), rgaindex], axis=-1)
                    # 修正下标表达[0~boxes对应anchor_index]

                    rpbbox_yx = tf.reshape(tf.gather_nd(rpbbox_yx, rgaindex), [-1, 2])              # 取当前anchor预测的yx
                    rpbbox_hw = tf.reshape(tf.gather_nd(rpbbox_hw, rgaindex), [-1, 2])              # 取当前anchor预测的hw
                    rpclass = tf.reshape(tf.gather_nd(rpclass, rgaindex), [-1, self.num_classes])   # 取当前anchor的类别
                    rpobj = tf.reshape(tf.gather_nd(rpobj, rgaindex), [-1, 1])                      # 取当前anchor的置信度

                    rabbox_hw = tf.reshape(tf.gather_nd(rabbox_hw, rgaindex), [-1, 2])              # 取该下标对应的anchor的hw
                    gn_labeli = tf.one_hot(gn_labeli, self.num_classes)                             # one-hot编码
                    rpbbox_yx_target = gn_yxi - tf.floor(gn_yxi)                                    # 中心点的目标预测补偿值 sigmoid(tx ty)目标值
                    rpbbox_hw_target = gn_hwi / rabbox_hw                                           # loge(x) hw目标预测的补偿值

                    # coord_loss
                    yx_loss = tf.reduce_sum(tf.square(rpbbox_yx_target - tf.nn.sigmoid(rpbbox_yx)))
                    hw_loss = tf.reduce_sum(tf.square(tf.exp(rpbbox_hw) - rpbbox_hw_target))
                    coord_loss = yx_loss + hw_loss
                    # class_loss obj_loss
                    class_loss = tf.reduce_sum(tf.square(gn_labeli - rpclass))
                    obj_loss = tf.reduce_sum(tf.square(tf.ones_like(rpobj) - tf.nn.sigmoid(rpobj)))
                    nogn_mask = tf.reshape(nogn_mask, [-1])

                    # 非最优预测: 即非gt所在的cell所出现的anchor
                    abbox_y1x1_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(abbox_yx-abbox_hw/2.,
                                                                                [-1, self.num_priors, 2]), nogn_mask), 1)
                    abbox_y2x2_nobest = tf.expand_dims(tf.boolean_mask(tf.reshape(abbox_yx+abbox_hw/2.,
                                                                                [-1, self.num_priors, 2]), nogn_mask), 1)
                    pobj_nobest = tf.boolean_mask(tf.reshape(tf.nn.sigmoid(pobj[i, ...]), [-1, self.num_priors]), nogn_mask)

                    num_g = tf.shape(gn_y1x1i)[0]               #gt的label个数
                    num_p = tf.shape(abbox_y1x1_nobest)[0]      #prediction的数目
                    gn_y1x1i = tf.tile(tf.expand_dims(gn_y1x1i, 0), [num_p, 1, 1, 1])   #g*num_p
                    gn_y2x2i = tf.tile(tf.expand_dims(gn_y2x2i, 0), [num_p, 1, 1, 1])

                    abbox_y1x1_nobest = tf.tile(abbox_y1x1_nobest, [1, num_g, 1, 1])    #p*num_g
                    abbox_y2x2_nobest = tf.tile(abbox_y2x2_nobest, [1, num_g, 1, 1])
                    agiou_y1x1 = tf.maximum(gn_y1x1i, abbox_y1x1_nobest)        #最右下的左上角
                    agiou_y2x2 = tf.minimum(gn_y2x2i, abbox_y2x2_nobest)        #最左上的右下角

                    agiou_area = tf.reduce_prod(agiou_y2x2 - agiou_y1x1, axis=-1)           #交集
                    aarea = tf.reduce_prod(abbox_y2x2_nobest - abbox_y1x1_nobest, axis=-1)
                    garea = tf.reduce_prod(gn_y2x2i - gn_y1x1i, axis=-1)
                    agiou = agiou_area / (aarea + garea - agiou_area)                       #iou
                    agiou = tf.reduce_max(agiou, axis=1)                                    #取nobest'max iou

                    #iou<0.5 认为检测到没有 若有物体则有误检的loss
                    noobj_loss = tf.reduce_sum(tf.square(tf.zeros_like(pobj_nobest) -
                                                         pobj_nobest)*tf.cast(agiou <= self.nms_iou_threshold, tf.float32))
                    loss = self.coord_sacle * coord_loss + self.class_scale * class_loss + \
                           self.obj_scale * obj_loss + self.noobj_scale * noobj_loss
                    self.total_loss.append(loss)
                total_loss = tf.reduce_mean(self.total_loss)

                self.loss = total_loss + self.weight_decay * tf.add_n(      #对于可训练变量使用L2正则化
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
                )
                with tf.variable_scope(('loss'), reuse=tf.compat.v1.AUTO_REUSE):
                    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
                    grads = self.optimizer.compute_gradients(loss=self.loss)
                    self.grad.append(grads)
        grad_value = self._average_gradients(self.grad)
        apply_gradient_op = self.optimizer.apply_gradients(grad_value, global_step=self.global_step)
        self.train_op = apply_gradient_op

    def _test_graph(self):
        with tf.variable_scope(('backone'), reuse=tf.AUTO_REUSE):  # get feature extractor
            features, passthrough, downsampling_rate = self._feature_extractor(self.images)
        with tf.variable_scope(('head'), reuse=tf.AUTO_REUSE):  # get yolov2 head
            conv1 = self._conv_layer(features, 1024, 3, 1, 'conv1')  # 23
            lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
            conv2 = self._conv_layer(lrelu1, 1024, 3, 1, 'conv2')  # 24
            lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')

            conv_passthrough = self._conv_layer(passthrough, 64, 1, 1, 'conv_pass')
            lrelu_passthrough = tf.nn.leaky_relu(conv_passthrough, 0.1, 'lrelu_pass')
            passthrough = tf.reshape(lrelu_passthrough, [self.batch_size, int(self.data_shape[0] / 32),
                                                         int(self.data_shape[1] / 32), -1], 'passthrough')

            axes = 3 if self.data_format == 'channels_last' else 1
            concatation = tf.concat([passthrough, lrelu2], axis=axes)  # output and passthrough concat

            conv3 = self._conv_layer(concatation, 1024, 3, 1, 'conv3')
            lrelu3 = tf.nn.leaky_relu(conv3, 0.1, name='lrelu3')
            pred = self._conv_layer(lrelu3, self.final_units, 1, 1, 'predictions')
            # 全卷积的输出层：final_units对应着anchor输出, 13*13*125  125=5(20+5)

            if self.data_format == 'channels_first':
                pred = tf.transpose(pred, [0, 2, 3, 1])  # 0维：数量  2 3 1 使channel置于最后
            pshape = tf.shape(pred)  # 输出张量维度

            pred = tf.reshape(pred, [pshape[0], pshape[1], pshape[2], self.num_priors, -1])
            # num_priors:暂时为5个anchor  -1维度:即是 classes+5
            pclass = tf.nn.softmax(pred[..., :self.num_classes], axis=-1)  # 每个class的概率
            pbbox_yx = pred[..., self.num_classes:self.num_classes + 2]  # ty tx
            pbbox_hw = pred[..., self.num_classes + 2:self.num_classes + 4]  # th tw
            pobj = pred[..., self.num_classes + 4:]  # t0
            abbox_yx, abbox_hw, abbox_y1x1, abbox_y2x2 = self._get_priors(pshape, self.priors)

        pclasst = tf.reshape(pclass[0, ...], [-1, self.num_classes])
        pobjt = tf.sigmoid(tf.reshape(pobj[0, ...], [-1, 1]))
        pbbox_yx = tf.reshape(pbbox_yx[0, ...], [-1, 2])
        pbbox_hw = tf.reshape(pbbox_hw[0, ...], [-1, 2])
        abbox_yx = tf.reshape(abbox_yx, [-1, 2])
        abbox_hw = tf.reshape(abbox_hw, [-1, 2])
        bbox_yx = tf.floor(abbox_yx) + tf.sigmoid(pbbox_yx)
        bbox_hw = abbox_hw * tf.exp(pbbox_hw)
        bbox_y1x1y2x2 = tf.concat([bbox_yx - bbox_hw / 2., bbox_yx + bbox_hw / 2.], axis=-1) * downsampling_rate
        confidence = pclasst * pobjt
        filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
        scores = []
        class_id = []
        bbox = []
        for i in range(self.num_classes):
            scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
            bboxi = tf.boolean_mask(bbox_y1x1y2x2, filter_mask[:, i])
            selected_indices = tf.image.non_max_suppression(
                bboxi, scoresi, self.nms_max_boxes, self.nms_iou_threshold,
            )
            scores.append(tf.gather(scoresi, selected_indices))
            bbox.append(tf.gather(bboxi, selected_indices))
            class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
        bbox = tf.concat(bbox, axis=0)
        scores = tf.concat(scores, axis=0)
        class_id = tf.concat(class_id, axis=0)
        self.detection_pred = [scores, bbox, class_id, pclasst]

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            if self.train_initializer is not None:
                self.sess.run(self.train_initializer)

    def _create_saver(self, path_backone, path_head):
        weights_backone = tf.trainable_variables(scope='backone')
        weights_head = tf.trainable_variables(scope='head')
        self.pretraining_weight_saver1 = tf.compat.v1.train.Saver(weights_backone, max_to_keep=1)
        self.pretraining_weight_saver2 = tf.compat.v1.train.Saver(weights_head, max_to_keep=1)
        if self.pretrain is True:
            self.pretraining_weight_saver1.restore(self.sess, path_backone)
            self.pretraining_weight_saver2.restore(self.sess, path_head)
            print('Load weight successfully!')


    def _create_summary(self):
        with tf.variable_scope('summaries'):
            tf.compat.v1.summary.scalar('loss', self.loss)
            self.summary_op = tf.compat.v1.summary.merge_all()

    def _get_priors(self, pshape, priors):
        tl_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)   #13
        tl_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)   #13
        tl_y_ = tf.reshape(tl_y, [-1, 1, 1, 1])                                 #13*1*1*1
        tl_x_ = tf.reshape(tl_x, [1, -1, 1, 1])                                 #1*13*1*1
        tl_y_ = tf.tile(tl_y_, [1, pshape[2], 1, 1])                            #tf.tile 对相应维度进行复制 ：13*13*1*1
        tl_x_ = tf.tile(tl_x_, [pshape[1], 1, 1, 1])                            #13*13*1*1
        tl = tf.concat([tl_y_, tl_x_], -1)                                      #13*13*1*2
        abbox_yx = tl + 0.5                                                     #每个格点的中心点
        abbox_yx = tf.tile(abbox_yx, [1, 1, self.num_priors, 1])                #13*13*5*2
        abbox_hw = priors                                                       #1*1*5*2
        abbox_hw = tf.tile(abbox_hw, [pshape[1], pshape[2], 1, 1])              #13*13*5*2
        abbox_y1x1 = abbox_yx - abbox_hw / 2                                    #中心点减去半个宽高
        abbox_y2x2 = abbox_yx + abbox_hw / 2                                    #中心点加上半个宽高
        return abbox_yx, abbox_hw, abbox_y1x1, abbox_y2x2                       #每个cell的5个anchor的中心点 宽高 左上角 右下角

    def _get_normlized_gn(self, downsampling_rate, i):

        slice_index = tf.argmin(self.ground_truth[i, ...], axis=0)[0]
        ground_truth = tf.gather(self.ground_truth[i, ...], tf.range(0, slice_index, dtype=tf.int64))   #获取gt[i]中的0到slice_index维度
        scale = tf.constant([downsampling_rate, downsampling_rate,
                             downsampling_rate, downsampling_rate, 1], dtype=tf.float32)    #y x h w class
        scale = tf.reshape(scale, [1, 5])
        gn = ground_truth / scale                                                           #压缩至输出图：13*13的比例
        return gn[..., :2], gn[..., 2:4], tf.cast(gn[..., 4], tf.int32)

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grads_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grads_and_vars:
                expand_g = tf.expand_dims(g, axis=0)
                grads.append(expand_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _feature_extractor(self, image):    #image为读入的一组数据
        conv1 = self._conv_layer(image, 32, 3, 1, 'conv1')
        lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
        pool1 = self._max_pooling(lrelu1, 2, 2, 'pool1')

        conv2 = self._conv_layer(pool1, 64, 3, 1, 'conv2')
        lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')
        pool2 = self._max_pooling(lrelu2, 2, 2, 'pool2')

        conv3 = self._conv_layer(pool2, 128, 3, 1, 'conv3')
        lrelu3 = tf.nn.leaky_relu(conv3, 0.1, 'lrelu3')
        conv4 = self._conv_layer(lrelu3, 64, 1, 1, 'conv4')
        lrelu4 = tf.nn.leaky_relu(conv4, 0.1, 'lrelu4')
        conv5 = self._conv_layer(lrelu4, 128, 3, 1, 'conv5')
        lrelu5 = tf.nn.leaky_relu(conv5, 0.1, 'lrelu5')
        pool3 = self._max_pooling(lrelu5, 2, 2, 'pool3')

        conv6 = self._conv_layer(pool3, 256, 3, 1, 'conv6')
        lrelu6 = tf.nn.leaky_relu(conv6, 0.1, 'lrelu6')
        conv7 = self._conv_layer(lrelu6, 128, 1, 1, 'conv7')
        lrelu7 = tf.nn.leaky_relu(conv7, 0.1, 'lrelu7')
        conv8 = self._conv_layer(lrelu7, 256, 3, 1, 'conv8')
        lrelu8 = tf.nn.leaky_relu(conv8, 0.1, 'lrelu8')
        pool4 = self._max_pooling(lrelu8, 2, 2, 'pool4')

        conv9 = self._conv_layer(pool4, 512, 3, 1, 'conv9')
        lrelu9 = tf.nn.leaky_relu(conv9, 0.1, 'lrelu9')
        conv10 = self._conv_layer(lrelu9, 256, 1, 1, 'conv10')
        lrelu10 = tf.nn.leaky_relu(conv10, 0.1, 'lrelu10')
        conv11 = self._conv_layer(lrelu10, 512, 3, 1, 'conv11')
        lrelu11 = tf.nn.leaky_relu(conv11, 0.1, 'lrelu11')
        conv12 = self._conv_layer(lrelu11, 256, 1, 1, 'conv12')
        lrelu12 = tf.nn.leaky_relu(conv12, 0.1, 'lrelu12')
        conv13 = self._conv_layer(lrelu12, 512, 3, 1, 'conv13')
        lrelu13 = tf.nn.leaky_relu(conv13, 0.1, 'lrelu13')
        pool5 = self._max_pooling(lrelu13, 2, 2, 'pool5')

        conv14 = self._conv_layer(pool5, 1024, 3, 1, 'conv14')
        lrelu14 = tf.nn.leaky_relu(conv14, 0.1, 'lrelu14')
        conv15 = self._conv_layer(lrelu14, 512, 1, 1, 'conv15')
        lrelu15 = tf.nn.leaky_relu(conv15, 0.1, 'lrelu15')
        conv16 = self._conv_layer(lrelu15, 1024, 3, 1, 'conv16')
        lrelu16 = tf.nn.leaky_relu(conv16, 0.1, 'lrelu16')
        conv17 = self._conv_layer(lrelu16, 512, 1, 1, 'conv17')
        lrelu17 = tf.nn.leaky_relu(conv17, 0.1, 'lrelu17')
        conv18 = self._conv_layer(lrelu17, 1024, 3, 1, 'conv18')
        lrelu18 = tf.nn.leaky_relu(conv18, 0.1, 'lrelu18')          #22
        downsampling_rate = 32.0
        return lrelu18, lrelu13, downsampling_rate      #lrelu13 :26*26*512 resize 13*13*2048 as passthrough

    def train_one_epoch(self, lr, writer=None):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // (self.batch_size*self.num_gpu)
        for i in range(num_iters):
            images = self.sess.run([self.images])
            images = np.squeeze(images)
            _, loss= self.sess.run([self.train_op, self.loss],
                                    feed_dict={self.lr: lr, self.input_images: images,
                                               self.anchor_shape: int(self.data_shape[0]/32)})
            sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
            # if writer is not None:
            #     writer.add_summary(summaries, global_step=self.global_step)
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def test_one_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images})
        return pred

    def save_section_weight(self, mode, path, loss, i):
        assert (mode in ['latest', 'best'])
        if not tf.io.gfile.exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')

        if mode == 'latest':
            self.pretraining_weight_saver1.save(self.sess, path + '-backone', global_step=self.global_step)
            self.pretraining_weight_saver2.save(self.sess, path + '-head', global_step=self.global_step)
            print('>> save', mode, 'model in', path, 'successfully')
        else:
            if i == 0:
                self.best_loss = loss
            else:
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.pretraining_weight_saver1.save(self.sess, path + '-backone', global_step=self.global_step)
                    self.pretraining_weight_saver2.save(self.sess, path + '-head', global_step=self.global_step)
                    tf.io.write_graph(tf.get_default_graph(), './yolo2/pb/',
                                         'Yolov2.pb', as_text=False)
                    print('>> save', mode, 'model in', path, 'successfully')
                else:
                    pass

    def count_para(self):
        from functools import reduce
        from operator import mul
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print(num_params)

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_layer(self, bottom, filters, kernel_size, strides, name):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format
        )
        bn = self._bn(conv)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )
