import os
import random
import utils.tfrecord_voc_utils as voc_utils
import tensorflow as tf
import YOLOv2 as yolov2
from memory_profiler import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lr = 0.0001
batch_size = 16
buffer_size = 800
epochs = 2000
resize_num = 20
data = os.listdir('./data/')
data = [os.path.join('./data/', name) for name in data]

shape = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
input_shape = [320, 320, 3]
factor = float(input_shape[0]/320)
path_backone = './yolo2/best/best-backone-1560'
path_head = './yolo2/best/best-head-1560'
config = {
    'mode': 'train',                                 # 'train', 'test'
    'is_pretraining': False,
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
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    # 'color_jitter_prob': 0.5,
    # 'rotate': [0.5, -10., 10.],
    'pad_truth_to': 60,
}

train_gen = voc_utils.get_generator(data, batch_size*config['num_gpu'], buffer_size, image_augmentor_config)

train_provider = {
    'data_shape': input_shape,
    'num_train': 5011,
    'num_val': None,
    'train_generator': train_gen,
    'val_generator': None
}


def run(j, learn_rate, test_net, init_list, mean_loss):
    if j % resize_num == 0 and j != 0:
        global input_shape
        test_net.save_section_weight('latest', './yolo2/latest/latest', mean_loss, j)
        test_net.sess.close()
        init_list.append(test_net.train_initializer)
        index = random.randint(0, 9)
        input_shape = [shape[index], shape[index], 3]
        print('Resize:  ' + str(input_shape))
        new_factor = float(input_shape[0] / 320)
        test_net.data_shape = input_shape
        priors = [[round(0.42*new_factor, 3), round(0.72*new_factor, 3)],
                  [round(0.92*new_factor, 3), round(1.63*new_factor, 3)],
                  [round(1.78*new_factor, 3), round(3.2*new_factor, 3)],
                  [round(3.68*new_factor, 3), round(5.2*new_factor, 3)],
                  [round(7.825*new_factor, 3), round(7.72*new_factor, 3)]]
        priors = tf.convert_to_tensor(priors, dtype=tf.float32)
        test_net.priors = tf.reshape(priors, [1, 1, test_net.num_priors, 2])
        image_augmentor_config['output_shape'] = [input_shape[0], input_shape[1]]

        train_gen1 = voc_utils.get_generator(data, batch_size*config['num_gpu'], buffer_size, image_augmentor_config)
        test_net.train_generator = train_gen1
        test_net.train_initializer, test_net.train_iterator = test_net.train_generator
        test_net.sess = tf.Session()
        test_net.sess.run(tf.global_variables_initializer())
        test_net.sess.run(init_list)
        test_net.define_inputs()
        for file in os.listdir('./yolo2/latest/'):
            if (file.split('.')[0]).startswith('latest-backone'):
                global backone
                backone = file.split('.')[0]
            elif (file.split('.')[0]).startswith('latest-head'):
                global head
                head = file.split('.')[0]
        test_net.latest_weight_saver1.restore(test_net.sess, './yolo2/latest/'+str(backone))
        test_net.latest_weight_saver2.restore(test_net.sess, './yolo2/latest/'+str(head))

    print('-'*25, 'epoch', j, '-'*25)
    if j == 500:
        learn_rate = learn_rate/5.
        print('reduce lr, lr=', learn_rate, 'now')
    if j == 800:
        learn_rate = learn_rate/2.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = test_net.train_one_epoch(learn_rate, j)
    print('>> mean loss', mean_loss)
    test_net.save_section_weight('best', './yolo2/best/best', mean_loss, j)  # 'latest', 'best'


if __name__ == "__main__":
    yolonet = yolov2.YOLOv2(config, train_provider)
    Init_list = []
    Mean_loss = 0
    print('Resize:  ' + str(input_shape))
    for i in range(epochs):
        run(i, lr, yolonet, Init_list, Mean_loss)
    '''
    yolonet = yolov2.YOLOv2(config, train_provider)
    yolonet.test_input()
    '''
