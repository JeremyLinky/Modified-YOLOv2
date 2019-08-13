import tensorflow as tf
import numpy as np
import os
import utils.tfrecord_voc_utils as voc_utils
import YOLOv2 as yolov2
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
import time
from utils.voc_classname_encoder import classname_to_ids

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

batch_size = 1
input_shape = [416, 416, 3]
reduce_lr_epoch = []
path_backone = './yolo2/test-backone-20748'
path_head = './yolo2/test-head-20748'
config = {
    'mode': 'test',                                 # 'train', 'test'
    'is_pretraining': True,
    'data_shape': input_shape,
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,
    'data_format': 'channels_last',                  # 'channels_last' 'channels_first'
    'batch_size': batch_size,
    'num_gpu': 1,
    'coord_scale': 1.,
    'noobj_scale': 1.,
    'obj_scale': 5.,
    'class_scale': 1.,

    'nms_score_threshold': 0.5,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.5,

    'rescore_confidence': False,
    'priors': [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],
    'path_backone': path_backone,
    'path_head': path_head
}

testnet = yolov2.YOLOv2(config, None)
testnet.count_para()
id_to_clasname = {k:v for (v, k) in classname_to_ids.items()}


def test_image():
    img = io.imread('./voc2007/JPEGImages/000407.jpg')
    # mean = [123.68, 116.779, 103.979]
    # mean = np.reshape(np.array(mean), [1, 1, 3])
    # img = img - mean
    img = transform.resize(img, [416, 416])
    img = np.expand_dims(img, 0)
    result = testnet.test_one_image(img)
    scores = result[0]
    bbox = result[1]
    class_id = result[2]
    print(scores, bbox, class_id)
    plt.figure(1)
    plt.imshow(np.squeeze(img))
    axis = plt.gca()
    for i in range(len(scores)):
        rect = patches.Rectangle((bbox[i][1], bbox[i][0]), bbox[i][3]-bbox[i][1], bbox[i][2]-bbox[i][0],
                                 linewidth=1, edgecolor='c', facecolor='none')
        axis.add_patch(rect)
        plt.text(bbox[i][1], bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
    plt.show()


def output_one_epoch():
    path = os.path.join(os.getcwd(), 'voc2007', 'JPEGImages')
    for i in os.listdir(path):
        index = i[0:6]
        image = cv2.imread(os.path.join(path, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #均值为RGB
        height, width, _ = image.shape              # y x
        y_scalar = round(height/416., 2)
        x_scalar = round(width/416., 2)
        image = cv2.resize(image, (416, 416))
        image_np = np.expand_dims(image, axis=0)
        result = testnet.test_one_image(image_np)    # 15ms
        scores = result[0]
        bbox = result[1]        # y1x1y2x2
        class_id = result[2]
        # pclass = result[3]
        length = len(scores)
        # print(pclass)
        result = [[(str(id_to_clasname[class_id[i]])+' ').encode('utf-8'),
                   (str(scores[i])+' ').encode('utf-8'),
                   (str(round(bbox[i][1]*x_scalar, 1))+' ').encode('utf-8'),
                   (str(round(bbox[i][0]*y_scalar, 1))+' ').encode('utf-8'),
                   (str(round(bbox[i][3]*x_scalar, 1))+' ').encode('utf-8'),
                   (str(round(bbox[i][2]*y_scalar, 1))+' ').encode('utf-8'),
                   ] for i in range(length)]
        #print(result)
        with open(os.path.join(os.getcwd(), 'voc2007', 'detection-results', str(index)+'.txt'), 'wb') as f:
            for i in range(length):
                f.writelines(result[i])
                f.write('\n'.encode('utf-8'))
        cv2.waitKey(0)


def test_demo():
    capture = cv2.VideoCapture(0)
    while True:
        time_begin = time.time()
        _, image = capture.read()
        image = cv2.resize(image, (416, 416))
        image_np = np.expand_dims(image, axis=0)
        result = testnet.test_one_image(image_np)
        print(result)
        scores, bbox, class_id = result[0], result[1], result[2]
        for i in range(len(scores)):
            cv2.rectangle(image, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), (255, 255, 255))
            cv2.putText(image, str(scores[i]).zfill(3) + str(id_to_clasname[class_id[i]]),
                        (bbox[i][1], bbox[i][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        cv2.imshow('demo', image)
        if cv2.waitKey(10) == ord('q'):
            cv2.destroyAllWindows()
            break
        time_end = time.time()
        print(1/(time_end-time_begin))
    capture.release()


if __name__ == "__main__":
    output_one_epoch()






