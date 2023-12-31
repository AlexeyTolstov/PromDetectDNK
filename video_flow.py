import cv2
import time

import ultralytics
from ultralytics import YOLO
from enum import Enum

ultralytics.checks()


class TypeOfObjects(Enum):
    Truck = 0
    Proflist = 1
    Person = 2
    Crane = 3


class Object:
    def __init__(self, type: TypeOfObjects, start_point: (int, int), end_point: (int, int)):
        self.type: TypeOfObjects = type
        self.start_point: (int, int) = tuple(map(int, start_point))
        self.end_point: (int, int) = tuple(map(int, end_point))

        self.x1: int = int(self.start_point[0])
        self.y1: int = int(self.start_point[1])

        self.x2: int = int(self.end_point[0])
        self.y2: int = int(self.end_point[1])

        self.center_top: (int, int) = ((self.x1 + self.x2) // 2, self.y1)
        self.center_bottom: (int, int) = ((self.x1 + self.x2) // 2, self.y2)

        self.width: int = self.start_point[0] - self.end_point[0]
        self.height: int = self.start_point[1] - self.end_point[1]

        if self.type == TypeOfObjects.Truck:
            self.title = "Truck"
        elif self.type == TypeOfObjects.Proflist:
            self.title = "Proflist"
        elif self.type == TypeOfObjects.Person:
            self.title = "Person"
        elif self.type == TypeOfObjects.Crane:
            self.title = "Crane"


def detect(frame, model):
    st_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, conf=0.2)

    img = results[0].orig_img
    classes = results[0].names
    boxes_cls = results[0].boxes.cls
    boxes_xy = results[0].boxes.xyxy

    lst_obj = []
    crane = None
    for i in range(len(boxes_cls)):
        obj = Object(TypeOfObjects(int(boxes_cls[i])),
                                    boxes_xy[i][:2].tolist(),
                                    boxes_xy[i][2:].tolist())

        if not crane and obj.type == TypeOfObjects.Crane:
            crane = obj

        lst_obj.append(obj)


    for obj in lst_obj:
        if obj.type == TypeOfObjects.Proflist and crane is None:
            continue
        elif obj.type == TypeOfObjects.Proflist:
            if ((obj.center_top[0] - crane.center_bottom[0] > 100 or obj.center_top[0] - crane.center_bottom[0] < 0)\
                    and abs(obj.center_top[1] - crane.center_bottom[1]) > 200) or \
                    not obj.width / crane.width >= 1:
                continue

        cv2.rectangle(img, obj.start_point, obj.end_point, (0, 255, 0), 2)
        cv2.putText(img, obj.title, obj.start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    fps = 1 / (time.time() - st_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    return img


cap = cv2.VideoCapture("Video/Cam 0 fragment_1.mp4")
model = YOLO('model 31 12.pt')

isCreateVideo = True

if isCreateVideo:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_.mp4', fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = cv2.cvtColor(detect(frame, model), cv2.COLOR_BGR2RGB)
    cv2.imshow('video', processed_frame)
    if isCreateVideo:
        out.write(processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if isCreateVideo: out.release()
cv2.destroyAllWindows()