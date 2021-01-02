from flask import Flask, Response
from recognise_faces import FaceRecognition

app = Flask(__name__)
fr = FaceRecognition(camera_idx=0)


@app.route('/')
def index():
    return "Hello"


@app.route('/camera')
def video_feed():
    return Response(fr.stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
