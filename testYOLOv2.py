import os
import utils.tfrecord_voc_utils as voc_utils
import YOLOv2 as yolov2

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lr = 0.0001
batch_size = 32
buffer_size = 2400
epochs = 1000
input_shape = [416, 416, 3]
reduce_lr_epoch = []
path_backone = './yolo2/test-backone-20748'
path_head = './yolo2/test-head-20748'
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
    'nms_iou_threshold': 0.7,

    'rescore_confidence': False,
    'priors': [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],
    'path_backone': path_backone,
    'path_head': path_head
}

data = os.listdir('./data/')
data = [os.path.join('./data/', name) for name in data]

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [416, 416],
    # 'zoom_size': [520, 520],
    # 'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    # 'color_jitter_prob': 0.5,
    # 'rotate': [0.5, -10., 10.],
    'pad_truth_to': 60,
}

train_gen = voc_utils.get_generator(data, batch_size, buffer_size, image_augmentor_config)
val_gen = voc_utils.get_generator(data, batch_size, buffer_size, image_augmentor_config)

train_provider = {
    'data_shape': input_shape,
    'num_train': 5011,
    'num_val': 256,
    'train_generator': train_gen,
    'val_generator': val_gen
}

testnet = yolov2.YOLOv2(config, train_provider)

for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i == 150:
        lr = lr/5.
    if i == 500:
        lr = lr/4
        print('reduce lr, lr=', lr, 'now')
    mean_loss = testnet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    testnet.save_section_weight('best', './yolo2/test', mean_loss, i)  # 'latest', 'best'
