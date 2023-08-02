from flask import Flask, render_template, Response
from camera import VideoCamera


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
# activate vedio camera feed
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# outputs prediction in it back to the web interface
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# web interface running on the localhost
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
