import json
import os
import numpy as np
import cv2

def convert_polygon_to_yolo(polygons, image_width, image_height):
    annotations = []

    for polygon in polygons:
        # 获取多边形的坐标点
        points = np.array(polygon['points'])

        # 计算多边形的最小外接矩形
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算外接矩形的中心点坐标和宽高
        x_center = rect[0][0] / image_width
        y_center = rect[0][1] / image_height
        width = rect[1][0] / image_width
        height = rect[1][1] / image_height

        # 将标签映射到类别
        label = polygon['label']
        class_name = label_to_class(label)

        # 将转换后的标注添加到列表中
        annotations.append(f'{class_name} {x_center} {y_center} {width} {height}')

    return annotations
    
def label_to_class(label):
    # 在这里进行标签到类别的映射
    # 此次任务中只有一个分类
    if label == 'surface':
        return '0'
    else:
        return 'none'
    

# 定义输入文件夹和输出文件夹
input_folder = './datasets/images/train/label file'
output_folder = './datasets/labels/train'

# input_folder = './datasets/images/test/label file'
# output_folder = './datasets/labels/test'

# input_folder = './datasets/images/val/label file'
# output_folder = './datasets/labels/val'

# 遍历输入文件夹中的所有json文件
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        # 构建输入文件的路径
        input_file_path = os.path.join(input_folder, filename)

        # 读取json文件
        with open(input_file_path, 'r') as json_file:
            data = json.load(json_file)

        # 获取图像的宽度和高度
        image_width = data['imageWidth']
        image_height = data['imageHeight']

        # 将多边形标注转换为yolov5格式
        annotations = convert_polygon_to_yolo(data['shapes'], image_width, image_height)

        # 构建输出文件的路径
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.txt')

        # 将转换后的标注写入文本文件
        with open(output_file_path, 'w') as txt_file:
            for annotation in annotations:
                txt_file.write(annotation + '\n')

