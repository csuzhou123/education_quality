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
    print(yaw_predicteds)
    total_yaw_movement = sum([abs(yaw[0]) for yaw in yaw_predicteds])
    total_pitch_movement = sum([abs(pitch[0]) for pitch in pitch_predicteds])
    total_roll_movement = sum([abs(roll[0]) for roll in roll_predicteds])

    # 计算平均幅度
    avg_yaw_amplitude = total_yaw_movement / len(yaw_predicteds)
    avg_pitch_amplitude = total_pitch_movement / len(pitch_predicteds)
    avg_roll_amplitude = total_roll_movement / len(roll_predicteds)

    # 计算频率
    yaw_frequency = len(yaw_predicteds) / duration
    pitch_frequency = len(pitch_predicteds) / duration
    roll_frequency = len(roll_predicteds) / duration

    return {
        'yaw_amplitude': avg_yaw_amplitude,
        'pitch_amplitude': avg_pitch_amplitude,
        'roll_amplitude': avg_roll_amplitude,
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

    head_movement_data = compute_head_movement_amplitude_and_frequency(
        [convert_tensors_to_floats(entry["yaw_predicteds"]) for entry in data],
        [convert_tensors_to_floats(entry["pitch_predicteds"]) for entry in data],
        [convert_tensors_to_floats(entry["roll_predicteds"]) for entry in data],
        duration=len(data)
    )
    return {
        'attention_level': attention_level,
        'emotion_counts': emotion_counts,
        'total_blinks': total_blinks,
        'total_head_movements': total_head_movements,
        'blink_frequency': blink_frequency,
        'head_movement_data': head_movement_data
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
            "roll_predicteds":roll_predicteds
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




