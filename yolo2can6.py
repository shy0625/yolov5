import cv2
import can
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import pyrealsense2 as rs
import numpy as np

# 初始化CAN总线
# bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate='500000')

# 模型文件路径
model_path = './weights/best.pt'
classes=['surface']
# 加载YOLOv5模型
model = torch.hub.load('.', 'custom', path=model_path, source='local')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device).eval()

# 设置间隔拍照的时间间隔（单位：毫秒）
interval = 3000

# 设置置信度阈值
conf_threshold = 0.90

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

cfg = pipeline.start(config)

align = rs.align(rs.stream.color)

def get_aligned_images():
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    # 获取相机参数
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

    img_color = np.asanyarray(aligned_color_frame.get_data())
    img_depth = np.asanyarray(aligned_depth_frame.get_data())

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.008), cv2.COLORMAP_JET)

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame

def detect_objects(frame):
    results = model(frame)

    # 解析模型结果
    detections = results.xyxy[0].cpu().numpy()

    # 将检测结果解析为物体的像素坐标信息
    detected_objects = []
    for detection in detections:
        # 提取每个物体的左上角和右下角坐标
        x1, y1, x2, y2, confidence, class_idx = detection

        # 保留置信度的检测结果
        if confidence >= conf_threshold:
            # 获取物体的中心坐标
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # 添加物体的像素坐标信息到列表中
            detected_objects.append([center_x, center_y])

    return detected_objects

try:
    while True:
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()

        # if not img_color.any() or not img_depth.any():
        #     continue
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((img_color, depth_colormap))

        center_xy = detect_objects(img_color)
        print ('pixel center_xy : ', center_xy)
        if center_xy:
            for i in range(len(center_xy)):
                ux = center_xy[i][0]
                uy = center_xy[i][1]

                print ('ux, uy : ', ux, uy)

                dis = aligned_depth_frame.get_distance(ux, uy) # 摄像头到检测物体的距离需大于约50厘米
                print ('distance: ', dis)

                # 计算相机坐标系
                camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                # 转换成三位小数
                camera_xyz = np.round(np.array(camera_xyz), 3)
                camera_xyz = np.array(list(camera_xyz))  # 输出的单位是米
                camera_xyz = list(camera_xyz)
                print ('camera_xyz: ', camera_xyz)

                cv2.circle(img_color, (ux, uy), 2, (0, 255, 0), 2)
                cv2.putText(img_color, str(camera_xyz), (ux+20, uy+10), 0, 1,
                            [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        else:
            print ('no detected objects')
            
        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow('detection', 640, 480)
        cv2.imshow('detection', img_color)
        cv2.waitKey(interval)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            pipeline.stop()
            break
finally:
    pipeline.stop()



