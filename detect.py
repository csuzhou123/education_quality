import numpy as np
from deepface import DeepFace
import cv2
import traceback
import dlib
from imutils import face_utils
import imutils
from eye_blink_detection.detect_blinks import eye_aspect_ratio
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import datasets, hopenet, utils
from PIL import Image
import torchvision
import torch.nn.functional as F
import tensorflow as tf
from keras.models import load_model
from object_detection.builders import model_builder
from object_detection.utils import config_util


CONFIG_PATH = 'Student_Logger-Gaze_Tracking-master/pupil track dependencies/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
att_detection_model = model_builder.build(model_config=configs['model'], is_training=False)
@tf.function
def detect_pupil(image):
    image, shapes = att_detection_model.preprocess(image)
    prediction_dict = att_detection_model.predict(image, shapes)
    detections = att_detection_model.postprocess(prediction_dict, shapes)
    return detections

def main_pupil(frame):
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_pupil(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    box = np.squeeze(detections['detection_boxes'])
    j = []
    for i in range(3):
        ymin = (int(box[i, 0] * 480))
        xmin = (int(box[i, 1] * 640))
        ymax = (int(box[i, 2] * 480))
        xmax = (int(box[i, 3] * 640))
        z = (xmin, ymin, xmax, ymax)
        j.append(z)
    return j[0][0], j[0][1], j[0][2], j[0][3], j[1][0], j[1][1], j[1][2], j[1][3], j[2][0], j[2][1], j[2][2], j[2][3]

def get_face_points(face):
    (x, y, w, h) = face_utils.rect_to_bb(face)
    return x,y,w,h

def get_head_pose(img, facial_landmarks):
    hp = [30,8,36,45,59,55]
    size = img.shape
    image_points = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in hp], dtype="double")
    model_points = np.array([(0.0, 0.0, 0.0),(0.0, -330.0, -65.0),(-225.0, 170.0, -135.0),(225.0, 170.0, -135.0),(-150.0, -150.0, -125.0),(150.0, -150.0, -125.0)])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return p1[0],p1[1],p2[0],p2[1]

def get_eye_points(img, facial_landmarks):
    lp = [36, 37, 38, 39, 40, 41]
    rp = [42, 43, 44, 45, 46, 47]
    #mask = np.zeros((480, 640), np.uint8)
    mask = np.zeros(img.shape[:2], np.uint8)

    lr = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in lp])
    rr = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in rp])

    mask1 = cv2.fillPoly(mask, [lr], 255)
    mask1 = cv2.fillPoly(mask1, [rr], 255)
    eye = cv2.bitwise_and(img, img, mask=mask1)

    leftEyeCenter = lr.mean(axis=0).astype("int")
    rightEyeCenter = rr.mean(axis=0).astype("int")
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))
    angle = round(angle, 2)

    return eye, angle


def compute_blink_frequency(blinks, duration):
    """
    计算眨眼频率
    :param blinks: 眨眼次数
    :param duration: 时间间隔（秒）
    :return: 眨眼频率（每分钟）
    """
    return (blinks / duration) * 60


def compute_head_movement_amplitude_and_frequency(yaw_predicteds, pitch_predicteds, roll_predicteds, duration):
    """
    计算头部晃动的幅度和频率
    :param yaw_predicteds: yaw 方向的预测值列表
    :param pitch_predicteds: pitch 方向的预测值列表
    :param roll_predicteds: roll 方向的预测值列表
    :param duration: 时间间隔（秒）
    :return: 头部晃动幅度和频率
    """
    threshold = 10

    # 计算大幅度晃动的时间段
    large_yaw_movements = sum(
        1 for i in range(1, len(yaw_predicteds)) if abs(yaw_predicteds[i][0] - yaw_predicteds[i - 1][0]) > threshold)
    large_pitch_movements = sum(
        1 for i in range(1, len(pitch_predicteds)) if abs(pitch_predicteds[i][0] - pitch_predicteds[i - 1][0]) > threshold)
    large_roll_movements = sum(
        1 for i in range(1, len(roll_predicteds)) if abs(roll_predicteds[i][0] - roll_predicteds[i - 1][0]) > threshold)
    # 计算大幅度晃动的频率
    yaw_frequency = large_yaw_movements / duration
    pitch_frequency = large_pitch_movements / duration
    roll_frequency = large_roll_movements / duration

    return {
        'yaw_amplitude': large_yaw_movements,
        'pitch_amplitude': large_pitch_movements,
        'roll_amplitude': large_roll_movements,
        'yaw_frequency': yaw_frequency,
        'pitch_frequency': pitch_frequency,
        'roll_frequency': roll_frequency
    }

def convert_tensors_to_floats(tensor_list):
    """
    将列表中的 tensor 对象转换为浮点数值
    :param tensor_list: 包含 tensor 对象的列表
    :return: 包含浮点数值的列表
    """
    return [float(tensor.item()) for tensor in tensor_list]

def evaluate_ratio(attention_list, threshold=0.65, consecutive_threshold=5):
    """
    评估学生上课的注意力是否集中，并记录注意力变化的时间点
    :param attention_list: 学生的注意力列表，True 表示注意力集中，False 表示不集中
    :param threshold: 注意力集中的阈值，默认为 0.8
    :return: 注意力集中时长和注意力变化时间点
    """
    print(attention_list)
    attention_list = list(attention_list)
    print(attention_list)
    if not attention_list:
        return 0, []

    # 记录注意力变化的时间点
    attention_changes = []
    # 记录注意力集中时长
    focused_duration = 0
    consecutive_focused_duration = 0
    # 上一个状态
    prev_state = attention_list[0]

    for i, state in enumerate(attention_list[1:], start=1):
        if state != prev_state:
            if prev_state:  # 注意力从 True 变为 False
                if consecutive_focused_duration >= consecutive_threshold:
                    attention_changes.append((i - consecutive_focused_duration, i))
                consecutive_focused_duration = 0
            else:  # 注意力从 False 变为 True
                consecutive_focused_duration = 1
        else:
            if state:  # 当前状态为 True
                consecutive_focused_duration += 1

        prev_state = state
        # 检查最后一段连续的注意力集中
    if prev_state and consecutive_focused_duration >= consecutive_threshold:
        attention_changes.append((len(attention_list) - consecutive_focused_duration, len(attention_list)))
    # 计算注意力集中时长
    total_duration = len(attention_list)
    focused_ratio = sum(end - start for start, end in attention_changes) / total_duration
    return focused_ratio, attention_changes

def evaluate_attention(emotion_data):
    """
    评估学生在30秒内的注意力。
    :param emotion_data: 30秒内的情绪数据列表，每个元素是一个情绪（字符串）
    :return: 一个字典，包含注意力评估结果和各情绪出现的频次
    """
    print("开始进行评估")
    high_attention_emotions = ['happy', 'surprise']
    medium_attention_emotions = ['sad', 'fear','neutral']
    low_attention_emotions = ['angry', 'disgust']

    emotion_counts = {
        'happy': 0,
        'surprise': 0,
        'neutral': 0,
        'sad': 0,
        'fear': 0,
        'angry': 0,
        'disgust': 0
    }
    total_emotions = 0
    total_blinks = 0
    total_head_movements = 0

    for entry in data:
        emotion = entry["max_emotion"]
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        total_emotions += 1
        total_blinks += entry["blinks"]
        total_head_movements += len(entry["yaw_predicteds"])

    high_attention_count = sum([emotion_counts[e] for e in high_attention_emotions])
    medium_attention_count = sum([emotion_counts[e] for e in medium_attention_emotions])
    low_attention_count = sum([emotion_counts[e] for e in low_attention_emotions])

    total_emotions = len(emotion_data)
    if total_emotions == 0:
        return {
            'attention_level': 'unknown',
            'emotion_counts': emotion_counts
        }

    high_attention_ratio = high_attention_count / total_emotions
    medium_attention_ratio = medium_attention_count / total_emotions
    low_attention_ratio = low_attention_count / total_emotions

    if high_attention_ratio > 0.6:
        attention_level = 'high'
    elif medium_attention_ratio > 0.6:
        attention_level = 'medium'
    elif low_attention_ratio > 0.6:
        attention_level = 'low'
    else:
        attention_level = 'mixed'

    blink_frequency = compute_blink_frequency(total_blinks, duration=len(data))
    well, attention_changes = evaluate_ratio(entry["is_att"] for entry in data)
    head_movement_data = compute_head_movement_amplitude_and_frequency(
        [convert_tensors_to_floats(entry["yaw_predicteds"]) for entry in data],
        [convert_tensors_to_floats(entry["pitch_predicteds"]) for entry in data],
        [convert_tensors_to_floats(entry["roll_predicteds"]) for entry in data],
        duration=len(data)
    )
    return {
        '积极程度': attention_level,
        'blink_frequency': blink_frequency,
        'head_movement_data': head_movement_data,
        '是否注意力集中':well,
        'attention_changes':attention_changes
    }

video_path = r'E:\ai-test\video_test\LB7iciNMKHyxn5N4.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

#emtion
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
print("[INFO] loading facial landmark predictor...")
#eye
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('eye_blink_detection/shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
data=[]
evaluation_interval = 30
last_evaluation_time = 0

#head_pose
head_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
print('Loading snapshot.')
# Load snapshot
saved_state_dict = torch.load(r'G:\education_quality\model\hopenet_robust_alpha1.pkl', map_location=torch.device('cpu'))
head_model.load_state_dict(saved_state_dict)
print('Loading data.')
transformations = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
print('Ready to test network.')
head_model.eval()
total = 0
idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor)
frame_cnt = 0
WEIGHT_PATH = "Student_Logger-Gaze_Tracking-master/4layer/Weights-9788--0.92333--0.96371.hdf5"
att_model = load_model('Student_Logger-Gaze_Tracking-master/4layer/Model.h5')
att_model.load_weights(WEIGHT_PATH)
while True:
    ret, frame = cap.read()
    frame_cnt = frame_cnt+1
    if frame_cnt!=5:
        continue
    if not ret:
        print("无法读取帧")
        break
    try:
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(results, list):
            results = results[0]
        dominant_emotion = results["dominant_emotion"]
        emotion_probabilities = results["emotion"]
        emotions = []
        max_emotion = max(results["emotion"], key=results["emotion"].get)
        max_probability = results["emotion"][max_emotion]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        is_att = False
        for face in faces:
            landmarks = predictor(gray, face)
            CC, angle = get_eye_points(frame, landmarks)
            x, y, w, he = get_face_points(face)
            x0, y0, x1, y1 = get_head_pose(gray, landmarks)
            a, b, c, d, e, f, g, h, i, j, k, l = main_pupil(CC)
            m = np.array([a, b, c, d, e, f, g, h, i, j, k, l, angle, x, y, w, he, x0, y0, x1, y1])
            m = tf.convert_to_tensor(np.expand_dims(m, 0), dtype=tf.float32)
            n = att_model.predict(m)
            if (n > 0.5):
                is_att = True

        eye_frame = imutils.resize(frame, width=450)
        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        rects = detector(eye_gray, 0)
        flag = False
        for rect in rects:
            shape = predictor(eye_gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    flag = True
                COUNTER = 0
                print(f"时间: {current_time_sec}秒, 情绪: {emotions}, 总眨眼次数: {TOTAL}，是否眨眼: {flag}")
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #head
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = detector(gray)
        yaw_predicteds = []
        pitch_predicteds = []
        roll_predicteds = []
        for face in rects:
            x_min, y_min, x_max, y_max = face.left(), face.top(), face.right(), face.bottom()
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            img = frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            img = transformations(img)
            img = img.view(1, img.size(0), img.size(1), img.size(2))
            img = Variable(img)
            yaw, pitch, roll = head_model(img)
            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
            yaw_predicteds.append(yaw_predicted)
            pitch_predicteds.append(pitch_predicted)
            roll_predicteds.append(roll_predicted)

        data.append({
            "time": current_time_sec,
            "max_emotion": max_emotion,
            "blinks": TOTAL,
            "is_blink":flag,
            'yaw_predicteds':yaw_predicteds,
            "pitch_predicteds":pitch_predicteds,
            "roll_predicteds":roll_predicteds,
            "is_att":is_att
        })
        cv2.imshow("Frame", frame)
        print(current_time_sec)
        if current_time_sec - last_evaluation_time >= evaluation_interval:
            print(evaluate_attention(data))
            last_evaluation_time = current_time_sec
            data = []
        frame_cnt = 0
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except Exception as e:
        print("情绪检测失败，错误详情如下：")
        traceback.print_exc()

cap.release()
cv2.destroyAllWindows()




