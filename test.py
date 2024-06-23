import time

import cv2

video_path = r'D:\test_video.mp4'
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1 / fps  # 帧之间的时间间隔（秒）
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

time_intervals = [
    (10, 14, "sleep"),
    (27, 32, "low attention"),
    (45, 51, "no face")
]

while True:
    ret, frame = cap.read()
    # current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    #
    # # 检查当前时间是否在任何一个标记的时间段内
    # for start_time, end_time, text in time_intervals:
    #     if start_time <= current_time <= end_time:
    #         cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #         break  # 找到一个匹配时间段后就可以跳出循环

    cv2.imshow("Frame", frame)
    time.sleep(frame_delay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()