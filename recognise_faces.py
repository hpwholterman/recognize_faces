import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np


class FaceRecognition(object):
    _win_title = 'Face recognition: press "f"'
    _default_name = 'Unknown'
    _factor = 3  # 1 for best recognition, 4 and up for better performance

    def __init__(self, camera_idx=0):
        self._faces_encodings = list()
        self._faces_names = list()
        self._video_capture = None
        self._camera_idx = camera_idx
        if self._load_faces() <= 0:
            raise Exception('No faces loaded!')
        if not self._get_capture():
            print(f'Invalid camera index entered: {self._camera_idx}\nTesting more indexes ->')
            self.test_cameras()
            raise Exception('No camera')

    def _load_faces(self):
        print('Loading faces...')
        image_folder = './images/'
        for f in os.listdir(image_folder):
            if not f.lower().endswith('.jpg'):
                continue
            face_name = f.replace('.jpg', '')
            print(f'-> {face_name}')
            try:
                face = face_recognition.load_image_file(file=f'{image_folder}{f}')
                self._faces_encodings.append(face_recognition.face_encodings(face_image=face)[0])
                self._faces_names.append(face_name)
            except Exception as e:
                print(f'Probably not a face: {f}... Error: {e}')
        print('Done')
        return len(self._faces_names)

    def _get_capture(self):
        self._video_capture = cv2.VideoCapture(self._camera_idx)
        return self._video_capture.isOpened()

    def _end_capture(self):
        self._video_capture.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        # Scale down for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=1/self._factor, fy=1/self._factor)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = list()
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self._faces_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self._faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self._faces_names[best_match_index]
            else:
                name = self._default_name
            face_names.append(name)
        if face_names:
            print(f'{datetime.now()} -> {",".join(face_names)}')
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= self._factor
            right *= self._factor
            bottom *= self._factor
            left *= self._factor
            # Draw a box around the face
            cv2.rectangle(img=frame,
                          pt1=(left, top),
                          pt2=(right, bottom),
                          color=(0, 0, 255),
                          thickness=2)
            # Draw a label with a name below the face
            cv2.rectangle(img=frame,
                          pt1=(left, bottom - 35),
                          pt2=(right, bottom),
                          color=(0, 0, 255),
                          thickness=cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img=frame,
                        text=name,
                        org=(left + 6, bottom - 6),
                        fontFace=font,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=1)

    def _recognise(self):
        cv2.namedWindow(self._win_title, cv2.WINDOW_NORMAL)
        start_processing = False
        while True:
            ret, frame = self._video_capture.read()
            if start_processing:
                self._process_frame(frame)
                cv2.imshow(self._win_title, frame)
            else:
                cv2.imshow(self._win_title, frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            elif key == ord('f'):
                start_processing = not start_processing

    def _gen_stream(self):
        while True:
            ret, frame = self._video_capture.read()
            self._process_frame(frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    @staticmethod
    def test_cameras():
        for idx in range(10):
            vid_cap = cv2.VideoCapture(idx)
            if vid_cap.isOpened():
                print(f'Camera found: {idx}')
                vid_cap.release()

    def start(self):
        self._recognise()

    def stream(self):
        for fr in self._gen_stream():
            yield fr
        self._end_capture()


if __name__ == '__main__':
    test = FaceRecognition(camera_idx=0)
    test.start()

