
from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model("vehicle_counter_model.h5")

HTML = '''
<h2>🚗 Vehicle Detection App</h2>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file">
<input type="submit">
</form>
{% if result %}
<h3>{{ result }}</h3>
{% endif %}
'''

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))
    return img

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        file.save("temp.jpg")

        img = preprocess("temp.jpg")
        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            result = "🚗 Vehicle Detected"
        else:
            result = "❌ No Vehicle"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
