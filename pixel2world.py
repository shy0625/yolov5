import pyrealsense2 as rs
import numpy as np
import cv2
import time

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

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.008), cv2.COLORMAP_JET)

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame

get_aligned_images()



    


    

