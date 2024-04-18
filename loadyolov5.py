import torch
import cv2
import can
from PIL import Image

model = torch.hub.load('.', 'custom', path='./weights/best.pt', source='local')


# results = model(img)

# results.show()


# 单步测试     
# frame = '0501_Color.png'
# objects = detect_objects(frame)

# # 将目标位置信息转换为CAN消息的数据格式，并发送到CAN总线上
# for obj in objects:
#     x, y = obj
#     print ('x , y ', x, y)
#     can_data = [x & 0xFF, (x >> 8) & 0xFF, y & 0xFF, (y >> 8) & 0xFF, 0, 0, 0, 0]
#     print ('can_data: ', can_data)

 # 拍照获取图像
# cap = cv2.VideoCapture(1)
#     # 检查摄像头是否成功打开
# if not cap.isOpened():
#     print ('Error: Failed to open camera.')
#     exit()
    
# while True:
#     ret, frame = cap.read()
#     # 检查是否成功读取帧
#     if not ret:
#         print ('Error: Failed to read frame.')
        
#     cv2.imshow('Camera', frame)
#     # cap.release()

#     # 检测按下键盘上的q键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate='500000')

# can_data = [0, 1, 2, 3, 4, 5, 6, 7]
# print ('can_data: ', can_data)
# msg = can.Message(arbitration_id=0x123, data=can_data)
# print (msg)
# bus.send(msg)
# print ('send msg: {}'.format(bus.channel_info))
# bus.shutdown()

# classes=['surface']
# conf_threshold = 0.90

# def detect_objects(frame):
#     results = model(frame)
    
#     # 获取检测到的工件上表面的位置信息
#     detections = results.xyxy[0].cpu().numpy()
#     objects = [(int((box[0] + box[2]) / 2),
#                int((box[1] + box[3]) / 2)) for box in detections[:, :4]]
    
#     # 在检测框上显示类别名称、置信度和中心位置坐标
#     for detect in results.pred[0]:
#         # 获取分类
#         class_idx = int(detect[-1])
#         class_name = classes[class_idx]
#         # 获取置信度
#         conf = float(detect[4])
#         if conf > conf_threshold:
#             # 获取中心位置坐标
#             box = detect[:4].tolist()
#             center_x = int((box[0] + box[2]) / 2)
#             center_y = int((box[1] + box[3]) / 2)
#             img = frame.copy()
#             start_point = (int(box[0]), int(box[1]))
#             end_point = (int(box[2]), int(box[3]))
#             cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
#             text = f'{class_name}: {conf:.2f}, Center: ({center_x:.2f}, {center_y:.2f})'
#             cv2.putText(img, text, (int(box[0]), int(box[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale=0.5, color=(0, 255, 0))
#             cv2.imshow('Detection', img)
            
#             return objects

# image = Image.open('0501_Color.png')
# detect_objects(image)

# image = Image.open('0501_Color.png')
# def detect_objects(frame):
#     results = model(frame)

#     # 解析模型的输出结果
#     detections = results.xyxy[0].cpu().numpy()
#     print ('detections: ', detections)
#     objects = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)) for box in detections[:, :4]]

#     print ('numbers: ', len(objects))
#     print ('objects: ', objects)
#     return objects

# detect_objects(image)




# # 初始化CAN总线
# bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate='500000')
# classes=['surface']
# conf_threshold = 0.90
# # 设置间隔拍照的时间间隔（单位：毫秒）
# interval = 3000
# def detect_objects(frame):
#     results = model(frame)

#     # 解析模型结果
#     detections = results.xyxy[0].cpu().numpy()

#     # 将检测结果解析为物体的像素坐标信息
#     # detected_objects = []
#     for detection in detections:
#         # 提取每个物体的左上角和右下角坐标
#         x1, y1, x2, y2, confidence, class_idx = detection

#         # 保留置信度的检测结果
#         if confidence >= conf_threshold:
#             # 获取物体的中心坐标
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             # 添加物体的像素坐标信息到列表中
#             # detected_objects.append((center_x, center_y))

#             # 在图像上绘制待抓取的目标物体
#             image = frame.copy()
#             # 绘制边界框
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             # 标注物体的位置
#             class_name = classes[int(class_idx)]
#             text = f'{class_name}: {confidence:.2f}, Center: ({center_x:.2f}, {center_y:.2f})'
#             cv2.putText(image, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             # 显示图像
#             cv2.imshow('Grabbed object', image)
#             cv2.waitKey(interval)
#             cv2.destroyWindow('Grabbed object')

#             # 将像素坐标信息转换为可以发送的格式，并通过PCAN发送出去
#             print ('center_x , center_y ', center_x, center_y)
#             can_data = [center_x & 0xFF, (center_x >> 8) & 0xFF, center_y & 0xFF, (center_y >> 8) & 0xFF, 0, 0, 0, 0]
#             print ('sending data via PCAN: ', can_data)
#             msg = can.Message(arbitration_id=0x123, data=can_data)
#             # 发送到数据总线
#             try:
#                 bus.send(msg)
#                 print ('message sent on {}'.format(bus.channel_info))
#             except can.CanError:
#                 print ('Failed to send data via PCAN')
#         else:
#             print ('no detected objects')
#             pass
    
    
# image = cv2.imread('0577_Color.png')
# detect_objects(image)


#-----------------------------------------yolo2can5.py
# 初始化CAN总线
bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate='500000')
classes=['surface']
conf_threshold = 0.90
# 设置间隔拍照的时间间隔（单位：毫秒）
interval = 3000
def detect_objects(frame):
    results = model(frame)

    # 解析模型结果
    detections = results.xyxy[0].cpu().numpy()

    # 将检测结果解析为物体的像素坐标信息
    # detected_objects = []
    for detection in detections:
        # 提取每个物体的左上角和右下角坐标
        x1, y1, x2, y2, confidence, class_idx = detection

        # 保留置信度的检测结果
        if confidence >= conf_threshold:
            # 获取物体的中心坐标
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # 添加物体的像素坐标信息到列表中
            # detected_objects.append((center_x, center_y))

            # 在图像上绘制待抓取的目标物体
            image = frame.copy()
            # 绘制边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 标注物体的位置
            class_name = classes[int(class_idx)]
            text = f'{class_name}: {confidence:.2f}, Center: ({center_x:.2f}, {center_y:.2f})'
            cv2.putText(image, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 显示图像
            cv2.imshow('Grabbed object', image)
            cv2.waitKey(interval)
            cv2.destroyWindow('Grabbed object')

            # 将像素坐标信息转换为可以发送的格式，并通过PCAN发送出去
            print ('center_x , center_y ', center_x, center_y)

            # --------------------------------------------------------yolo2can5.py
            can_data = [center_x & 0xFF, (center_x >> 8) & 0xFF, center_y & 0xFF, (center_y >> 8) & 0xFF, 0, 0, 0, 0]
            print ('sending data via PCAN: ', can_data)

            
            msg = can.Message(arbitration_id=0x123, data=can_data)
            # 发送到数据总线
            try:
                bus.send(msg)
                print ('message sent on {}'.format(bus.channel_info))
            except can.CanError:
                print ('Failed to send data via PCAN')
        else:
            print ('no detected objects')
            pass
    
    
image = cv2.imread('0577_Color.png')
detect_objects(image)

