import os
import gc
import random
import utils.tfrecord_voc_utils as voc_utils
import tensorflow as tf
import YOLOv2 as yolov2

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lr = 0.0005
batch_size = 16
buffer_size = 800
epochs = 1000
data = os.listdir('./data/')
data = [os.path.join('./data/', name) for name in data]

shape = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
input_shape = [320, 320, 3]
factor = float(input_shape[0]/320)
path_backone = './yolo2/test-backone-1560'
path_head = './yolo2/test-head-1560'
config = {
    'mode': 'train',                                 # 'train', 'test'
    'is_pretraining': True,
    'data_shape': input_shape,
    'num_classes': 20,
    'weight_decay': 5e-4,
    'keep_prob': 0.5,
    'data_format': 'channels_last',                  # 'channels_last' 'channels_first'
    'batch_size': batch_size,
    'num_gpu': 2,
    'coord_scale': 1.,
    'noobj_scale': .5,
    'obj_scale': 5.,
    'class_scale': 5.,

    'nms_score_threshold': 0.5,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.6,

    'rescore_confidence': False,
    'priors': [[0.42*factor, 0.72*factor], [0.92*factor, 1.63*factor], [1.78*factor, 3.2*factor],
               [3.68*factor, 5.2*factor], [7.825*factor, 7.72*factor]],
    'path_backone': path_backone,
    'path_head': path_head
}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [input_shape[0], input_shape[1]],
    # 'zoom_size': [520, 520],
    # 'crop_method': 'random',
    # 'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    # 'color_jitter_prob': 0.5,
    # 'rotate': [0.5, -10., 10.],
    'pad_truth_to': 60,
}

dataset = voc_utils.Dataset()
train_gen = dataset.get_generator(data, batch_size, buffer_size, image_augmentor_config)

train_provider = {
    'data_shape': input_shape,
    'num_train': 5011,
    'num_val': None,
    'train_generator': train_gen,
    'val_generator': None
}

testnet = yolov2.YOLOv2(config, train_provider)
for i in range(epochs):
    if i % 1 == 0 and i != 0:
        index = random.randint(0, 9)
        input_shape = [shape[0], shape[0], 3]
        factor = float(input_shape[0] / 320)
        testnet.data_shape = input_shape
        priors = [[round(0.42*factor, 3), round(0.72*factor, 3)],
                  [round(0.92*factor, 3), round(1.63*factor, 3)],
                  [round(1.78*factor, 3), round(3.2*factor, 3)],
                  [round(3.68*factor, 3), round(5.2*factor, 3)],
                  [round(7.825*factor, 3), round(7.72*factor, 3)]]
        priors = tf.convert_to_tensor(priors, dtype=tf.float32)
        testnet.priors = tf.reshape(priors, [1, 1, testnet.num_priors, 2])
        image_augmentor_config['output_shape'] = [input_shape[0], input_shape[1]]
        del dataset, train_gen, testnet.train_initializer, testnet.train_iterator, testnet.train_generator
        gc.collect()
        dataset = voc_utils.Dataset()
        train_gen = dataset.get_generator(data, batch_size, buffer_size, image_augmentor_config)
        testnet.change_iterator(train_gen)
        testnet.define_inputs()

    print('-'*25, 'epoch', i, '-'*25)
    print('Resize:  ' + str(input_shape))
    if i == 500:
        lr = lr/5.
        print('reduce lr, lr=', lr, 'now')
    if i == 800:
        lr = lr/4
        print('reduce lr, lr=', lr, 'now')

    mean_loss = testnet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    testnet.save_section_weight('best', './yolo2/test', mean_loss, i)  # 'latest', 'best'

