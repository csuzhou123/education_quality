from PIL import ImageDraw
# from PIL.Image import Image
# from deepface import DeepFace
# DeepFace.stream(db_path = "database")
# result = DeepFace.analyze(
#   img_path = "database/hlk.JPG",
#   actions = ['emotion'],
# )
#
# im = Image.open("images/jiangwen/0000.jpg")
# # 坐标位置
# x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
# draw = ImageDraw.Draw(im)
# # 画框
# draw.rectangle((x, y, x + w, y + h), outline="red", width=3)
# print("表情：{}".format(result["dominant_emotion"]))
# imshow([im], ["jiangwen"])


from deepface import DeepFace
import cv2
import traceback

# 打开摄像头
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    try:
        # 调用 DeepFace 进行情绪检测
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # 检查返回结果的类型
        if isinstance(results, list):
            results = results[0]

        dominant_emotion = results["dominant_emotion"]
        emotion_probabilities = results["emotion"]

        # 获取脸部位置信息
        region = results['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # 在图像上框出脸部
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 在脸部框旁边显示情绪
        text = f"Emotion: {dominant_emotion}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 打印概率大于 0.7 的情绪
        for emotion, probability in emotion_probabilities.items():
            if probability > 0.7:
                print(f"{emotion}: {probability:.2f}")

        cv2.imshow("Frame", frame)
    except Exception as e:
        print("情绪检测失败，错误详情如下：")
        traceback.print_exc()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



