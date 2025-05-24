
from flask import Flask, render_template, request
from detect_emotion import predict_emotion
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    if request.method == 'POST':
        img = request.files['image']
        img_path = os.path.join('static/images', img.filename)
        img.save(img_path)
        emotion = predict_emotion(img_path)
    return render_template('index.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
