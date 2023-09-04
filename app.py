from flask import Flask, render_template, url_for, request, redirect, send_from_directory
import os
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'x-ray_images')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
filename=''

model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")
categories_parts = ["Elbow", "Hand", "Shoulder"]
#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']

def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    # chose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_pneumonia', methods=['GET', 'POST'])
def check_pneumonia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            model=load_model('weights/our_model.h5') #Loading our model
            img=image.load_img(filepath,target_size=(224,224))
            imagee=image.img_to_array(img) #Converting the X-Ray into pixels
            imagee=np.expand_dims(imagee, axis=0)
            img_data=preprocess_input(imagee)
            prediction=model.predict(img_data)
            if prediction[0][0]>prediction[0][1]:  #Printing the prediction of model.
                return 'Person is safe.'
            else:
                return 'Person is affected with Pneumonia.'
    
    return render_template('check_pneumonia.html')

@app.route('/check_fracture', methods=['GET', 'POST'])
def check_fracture():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            body_part=predict(filepath)
            response=predict(filepath,body_part)
            
            return f"Scanned X-Ray of {body_part} and it is observed to be {response} "
    
    return render_template('check_fracture.html')

@app.route('/success')
def success():
    return "Success"

@app.route('/x-ray/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
