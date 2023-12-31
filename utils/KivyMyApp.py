from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import time
import cv2

from ultralytics import YOLO
import ultralytics
ultralytics.checks()

FPS = 30


class MyApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.capture = cv2.VideoCapture(0)

        self.image = Image()
        self.layout.add_widget(self.image)

        Clock.schedule_interval(self.update, 1.0 / FPS)

        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            st_time = time.time()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, conf=0.5)
            
            img = results[0].orig_img
            classes = results[0].names
            boxes_cls = results[0].boxes.cls
            boxes_xy = results[0].boxes.xyxy

            for i in range(len(boxes_cls)):
                cv2.rectangle(
                    img,
                    list(map(int, boxes_xy[i][:2].tolist())),
                    list(map(int, boxes_xy[i][2:].tolist())),
                    (0, 255, 0),
                    2
                )

                xn, yn = map(int, boxes_xy[i][:2].tolist())
                cv2.putText(img, 
                        classes[int(boxes_cls[i])],
                        (xn, yn),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
                
            fps = 1 / (time.time() - st_time)
            cv2.putText(img,
                            f"FPS: {fps:.2f}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            5)
            self.image.texture = self.frame_to_texture(img)

    def frame_to_texture(self, frame):
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        return texture


if __name__ == '__main__':
    model = YOLO('best.pt')
    MyApp().run()
