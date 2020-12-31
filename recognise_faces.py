import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np


class FaceRecognition(object):
    _win_title = 'Face recognition: press "f"'
    _default_name = 'Unknown'
    _factor = 3  # 1 for best recognition, 4 and up for better performance

    def __init__(self):
        self._faces_encodings = list()
        self._faces_names = list()
        self._video_capture = None

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

    def _get_capture(self):
        cv2.namedWindow(self._win_title, cv2.WINDOW_NORMAL)
        self._video_capture = cv2.VideoCapture(0)

    def _end_capture(self):
        self._video_capture.release()
        cv2.destroyAllWindows()

    # @profile
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

    def start(self):
        self._load_faces()
        self._get_capture()
        self._recognise()
        self._end_capture()


if __name__ == '__main__':
    test = FaceRecognition()
    test.start()
