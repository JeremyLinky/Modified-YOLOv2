import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document
import math
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import tensorflow as tf
# 获取路径下所有文件的完整路径，用于读取文件用
def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

def limit_value(a, b):
    if a < 1:
        a = 1
    if a > b:
        a = b - 1
    return a


# 读取xml文件，xmlfile参数表示xml的路径
def readXml(xmlfile):
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    sizelist = annotation.getElementsByTagName('size')  # [<DOM Element: filename at 0x381f788>]
    heights = sizelist[0].getElementsByTagName('height')
    height = int(heights[0].childNodes[0].data)
    widths = sizelist[0].getElementsByTagName('width')
    width = int(widths[0].childNodes[0].data)
    depths = sizelist[0].getElementsByTagName('depth')
    depth = int(depths[0].childNodes[0].data)
    objectlist = annotation.getElementsByTagName('object')
    bboxes = []
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        class_label = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')[0]
        x1_list = bndbox.getElementsByTagName('xmin')
        x1 = int(float(x1_list[0].childNodes[0].data))
        y1_list = bndbox.getElementsByTagName('ymin')
        y1 = int(float(y1_list[0].childNodes[0].data))
        x2_list = bndbox.getElementsByTagName('xmax')
        x2 = int(float(x2_list[0].childNodes[0].data))
        y2_list = bndbox.getElementsByTagName('ymax')
        y2 = int(float(y2_list[0].childNodes[0].data))
        # 这里我box的格式【xmin，ymin，xmax，ymax，classname】
        bbox = [x1, y1, x2, y2, class_label]
        bboxes.append(bbox)
        print(bboxes)
    return bboxes, width, height, depth


# 图像旋转用，里面的angle是角度制的
def im_rotate(im, angle, center=None, scale=1.0):
    h, w = im.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    im_rot = cv2.warpAffine(im, M, (w, h))
    return im_rot
#噪声
def add_Noise(img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        #time==10
        #random.seed(int(time.time()))
        #noise_im=random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        noise_im=random_noise(img, mode='s&p', clip=True)*255
        return noise_im

#亮度
def changeLight(img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.3, 4) #flag>1为调暗,小于1为调亮
        Light_img= exposure.adjust_gamma(img, flag)
        return Light_img
def im_flip(im, method='H'):  # 翻转图像
    if method == 'H':  # Flipped Horizontally 水平翻转
        im_flip = cv2.flip(im, 1)
    elif method == 'V':  # Flipped Vertically 垂直翻转
        im_flip = cv2.flip(im, 0)
    # elif method == 'HV':# Flipped Horizontally & Vertically 水平垂直翻转
    #    im_flip = cv2.flip(im, -1)
    return im_flip

def cutout(img, bboxes, length=15, n_holes=15, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape

        mask = np.ones((h, w, c), np.float32)

        for n in range(n_holes):

            chongdie = True  # 看切割的区域是否与box重叠太多

            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img
def Cut_img(img_path, anno_new_dir, img_new_dir):

    im = cv2.imread(img_path)
    file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
    ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
    new_img_name = '%s_%s%s' % ('Cut', file_name, ext)
    anno = os.path.join(anno_path, '%s.xml' % file_name)
    [gts,w, h, d] = readXml(anno)
    Cut_img=cutout(im,gts)
    cv2.imwrite(os.path.join(img_new_dir, new_img_name), Cut_img)

    writeXml(anno_new_dir, new_img_name, w, h, d, gts)
# 写xml文件，参数中tmp表示路径，imgname是文件名（没有尾缀）ps有尾缀也无所谓
def writeXml(tmp, imgname, w, h, d, bboxes):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[4]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(bbox[0]))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(bbox[1]))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(bbox[2]))
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(bbox[3]))
        ymax.appendChild(ymax_txt)

        print(bbox[0], bbox[1], bbox[2], bbox[3])

    xmlname = os.path.splitext(imgname)[0]
    #print(xmlname)
    tempfile = tmp + "/%s.xml" % xmlname
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

def flip_image(img_path, method, anno_new_dir, img_new_dir):

    # 读取原图像
    im = cv2.imread(img_path)
    flip_img = im_flip(im, method)  # 翻转
    (H, W, D) = flip_img.shape  # 得到翻转后的图像的高、宽、深度，用于书写xml

    file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称

    ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
    # 保存翻转后图像
    new_img_name = '%s_%s%s' % (method, file_name, ext)
    #global new_img_name
    cv2.imwrite(os.path.join(img_new_dir, new_img_name), flip_img)  # 新的命名方式为H/V+原图名称
    # 读取anno标签数据，返回相应的信息
    anno = os.path.join(anno_path, '%s.xml' % file_name)
    [gts, w, h, d] = readXml(anno)
    gt_new = []
    for gt in gts:
        x1 = gt[0]  # xmin
        y1 = gt[1]  # ymin
        x2 = gt[2]  # xmax
        y2 = gt[3]  # ymax
        classname = str(gt[4])
        if method == 'H':
            x1 = w - 1 - x1  # xmax
            x2 = w - 1 - x2  # xmin
            x1 = limit_value(x1, w)
            x2 = limit_value(x2, w)
            gt_new.append([x2, y1, x1, y2, classname])
        elif method == 'V':
            y1 = h - 1 - y1  # ymax
            y2 = h - 1 - y2  # ymin
            y1 = limit_value(y1, h)
            y2 = limit_value(y2, h)
            gt_new.append([x1, y2, x2, y1, classname])
    writeXml(anno_new_dir, new_img_name, W, H, D, gt_new)
def _filp_pic_bboxes(img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon: #水平翻转
            flip_img =  cv2.flip(flip_img, 1)   #1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w-x_max, y_min, w-x_min, y_max])
            else:
                flip_bboxes.append([x_min, h-y_max, x_max, h-y_min])

        return flip_img, flip_bboxes
#平移
def shift_pic_bboxes(img,bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
            classname = str(bbox[4])
        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) , (d_to_right - 1))
        y = random.uniform(-(d_to_top - 1)  , (d_to_bottom - 1))

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = []
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y,classname])

        return shift_img, shift_bboxes
def Shift_img(img_path, anno_new_dir, img_new_dir):

    # 读取原图像
    im = cv2.imread(img_path)
    file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
    ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
    new_img_name = '%s_%s%s' % ('Shift_', file_name, ext)
    anno = os.path.join(anno_path, '%s.xml' % file_name)
    [gts,w, h, d] = readXml(anno)
    Shift_img,gt_new=shift_pic_bboxes(im,gts)
    cv2.imwrite(os.path.join(img_new_dir, new_img_name), Shift_img)

    writeXml(anno_new_dir, new_img_name, w, h, d, gt_new)

def crop_img_bboxes(img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
            classname = str(bbox[4])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min,classname])

        return crop_img, crop_bboxes
def Crop_img(img_path, anno_new_dir, img_new_dir):

        # 读取原图像
    im = cv2.imread(img_path)
    file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
    ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
    new_img_name = '%s_%s%s' % ('Crop_', file_name, ext)
    anno = os.path.join(anno_path, '%s.xml' % file_name)
    [gts,w, h, d] = readXml(anno)
    Crop_img,gt_new=crop_img_bboxes(im,gts)
    cv2.imwrite(os.path.join(img_new_dir, new_img_name),Crop_img)

    writeXml(anno_new_dir, new_img_name, w, h, d, gt_new)

def ChangeLight_img(img_path, anno_new_dir, img_new_dir):

    # 读取原图像
    im = cv2.imread(img_path)
    ChangeLight_img=changeLight(im)
    (H, W, D) = ChangeLight_img.shape  # 得到变化后的图像的高、宽、深度，用于书写xml
    file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
    ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
    # 保存变化后图像
    new_img_name = '%s_%s%s' % ('ChangeLight', file_name, ext)
    cv2.imwrite(os.path.join(img_new_dir, new_img_name), ChangeLight_img)
    anno = os.path.join(anno_path, '%s.xml' % file_name)
    [gts, w, h, d] = readXml(anno)
    gt_new = []
    for gt in gts:
        x1 = gt[0]  # xmin
        y1 = gt[1]  # ymin
        x2 = gt[2]  # xmax
        y2 = gt[3]  # ymax
        classname = str(gt[4])
        gt_new.append([x1, y2, x2, y1, classname])
    writeXml(anno_new_dir, new_img_name, w, h, d, gt_new)
def noise_image(img_path, anno_new_dir, img_new_dir):

    # 读取原图像
    im = cv2.imread(img_path)
    noise_image=add_Noise(im)
    file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
    ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
    # 保存加噪声后图像
    new_img_name = '%s_%s%s' % ('addnoise', file_name, ext)
    cv2.imwrite(os.path.join(img_new_dir, new_img_name), noise_image)
    anno = os.path.join(anno_path, '%s.xml' % file_name)
    [gts, w, h, d] = readXml(anno)
    gt_new = []
    for gt in gts:
        x1 = gt[0]  # xmin
        y1 = gt[1]  # ymin
        x2 = gt[2]  # xmax
        y2 = gt[3]  # ymax
        classname = str(gt[4])
        gt_new.append([x1, y2, x2, y1, classname])
    writeXml(anno_new_dir, new_img_name, w, h, d, gt_new)

def rotate_image(angles, angle_rad, img_path, anno_new_dir, img_new_dir):
    j = 0  # 计数用
    angle_num = len(angles)
    im = cv2.imread(img_path)
    for i in range(angle_num):
        gt_new = []
        im_rot = im_rotate(im, angles[i])  # 旋转
        (H, W, D) = im_rot.shape  # 得到旋转后的图像的高、宽、深度，用于书写xml
        file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
        ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
        # 保存旋转后图像
        new_img_name = 'P%s_%s%s' % (angles[i], file_name, ext)
        cv2.imwrite(os.path.join(img_new_dir, new_img_name), im_rot)  # 新的命名方式为P+角度+原图名称
        # 读取anno标签数据，返回相应的信息
        anno = os.path.join(anno_path, '%s.xml' % file_name)
        [gts, w, h, d] = readXml(anno)
        # 计算旋转后gt框四点的坐标变换
        [xc, yc] = [float(w) / 2, float(h) / 2]
        for gt in gts:
            # 计算左上角点的变换
            x1 = (gt[0] - xc) * math.cos(angle_rad[i]) - (yc - gt[1]) * math.sin(angle_rad[i]) + xc
            if int(x1) <= 0: x1 = 1.0
            if int(x1) > w - 1: x1 = w - 1
            y1 = yc - (gt[0] - xc) * math.sin(angle_rad[i]) - (yc - gt[1]) * math.cos(angle_rad[i])
            if int(y1) <= 0: y1 = 1.0
            if int(y1) > h - 1: y1 = h - 1
            # 计算右上角点的变换
            x2 = (gt[2] - xc) * math.cos(angle_rad[i]) - (yc - gt[1]) * math.sin(angle_rad[i]) + xc
            if int(x2) <= 0: x2 = 1.0
            if int(x2) > w - 1: x2 = w - 1
            y2 = yc - (gt[2] - xc) * math.sin(angle_rad[i]) - (yc - gt[1]) * math.cos(angle_rad[i])
            if int(y2) <= 0: y2 = 1.0
            if int(y2) > h - 1: y2 = h - 1
            # 计算左下角点的变换
            x3 = (gt[0] - xc) * math.cos(angle_rad[i]) - (yc - gt[3]) * math.sin(angle_rad[i]) + xc
            if int(x3) <= 0: x3 = 1.0
            if int(x3) > w - 1: x3 = w - 1
            y3 = yc - (gt[0] - xc) * math.sin(angle_rad[i]) - (yc - gt[3]) * math.cos(angle_rad[i])
            if int(y3) <= 0: y3 = 1.0
            if int(y3) > h - 1: y3 = h - 1
            # 计算右下角点的变换
            x4 = (gt[2] - xc) * math.cos(angle_rad[i]) - (yc - gt[3]) * math.sin(angle_rad[i]) + xc
            if int(x4) <= 0: x4 = 1.0
            if int(x4) > w - 1: x4 = w - 1
            y4 = yc - (gt[2] - xc) * math.sin(angle_rad[i]) - (yc - gt[3]) * math.cos(angle_rad[i])
            if int(y4) <= 0: y4 = 1.0
            if int(y4) > h - 1: y4 = h - 1
            xmin = min(x1, x2, x3, x4)
            xmax = max(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            ymax = max(y1, y2, y3, y4)
            # 把因为旋转导致的特别小的 长线型的去掉
            # w_new = xmax-xmin+1
            # h_new = ymax-ymin+1
            # ratio1 = float(w_new)/h_new
            # ratio2 = float(h_new)/w_new
            # if(1.0/6.0<ratio1<6 and 1.0/6.0<ratio2<6 and w_new>9 and h_new>9):
            classname = str(gt[4])
            gt_new.append([xmin, ymin, xmax, ymax, classname])
            # 写出新的xml
            writeXml(anno_new_dir, new_img_name, W, H, D, gt_new)
        j = j + 1
        if j % 100 == 0: print('----%s----' % j)




if __name__ == '__main__':

    # 数据路径
    root = '/home/pwj/Deeplearning/2019/2019.1/darknet_digit/scripts/VOCdevkit/voc2007'
    img_dir = root + '/JPEGImages'
    anno_path = root + '/Annotations'
    imgs_path = GetFileFromThisRootDir(img_dir)  # 返回每一张原图的路径
    for img_path in imgs_path:
        strs = ['Rotate', 'Flip', 'Noise', 'ChangeLight', 'Shife', 'Cut','Crop']
        n=random.randint(0,len(strs)-1) # 数据扩增的方式 Rotate代表旋转，Flip表示翻转
        AUG=strs[n]
        # 存储新的anno位置
        anno_new_dir = os.path.join(root, 'AfterChange_Annotations')
        if not os.path.isdir(anno_new_dir):
            os.makedirs(anno_new_dir)
        # 扩增后图片保存的位置
        img_new_dir = os.path.join(root, 'AfterChange_JPEGImage')
        if not os.path.isdir(img_new_dir):
            os.makedirs(img_new_dir)

        if AUG == 'Rotate':
            # 旋转角的大小，正数表示逆时针旋转
            angles = [5, 90, 180, 270, 355]  # 角度im_rotate用到的是角度制
            angle_rad = [angle * math.pi / 180.0 for angle in angles]  # cos三角函数里要用到弧度制的
            # 开始旋转
            rotate_image(angles, angle_rad, img_path, anno_new_dir, img_new_dir)
        elif AUG == 'Flip':
            method = 'H'
            flip_image(img_path, method, anno_new_dir, img_new_dir)
        elif AUG=='Noise':
            noise_image(img_path, anno_new_dir, img_new_dir)
        elif AUG=='ChangeLight':
            ChangeLight_img(img_path, anno_new_dir, img_new_dir)
        elif AUG=='Shife':
            Shift_img(img_path, anno_new_dir, img_new_dir)
        elif AUG=='Crop':
            Crop_img(img_path, anno_new_dir, img_new_dir)
        elif AUG=='Cut':
            Cut_img(img_path, anno_new_dir, img_new_dir)


