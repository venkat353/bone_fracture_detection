from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''# Define the ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

# Create the train_generator
train_generator = train_datagen.flow_from_directory('path/to/train_directory',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')'''


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor= 'accuracy', patience=3)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//validation_generator.batch_size,
    callbacks=[early_stopping]
)
# # load model
# from tensorflow.keras.models import load_model
# #
# model = load_model('C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\Model.h5')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples//test_generator.batch_size)
print('Test accuracy:', test_acc)

model.save('C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\Model.h5')

test_generator.reset()
y_pred = model.predict(test_generator) #, steps=test_generator.samples//test_generator.batch_size)
y_pred = np.round(y_pred)


from sklearn.metrics import classification_report
# Generate classification report
print(classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys()))




# Plot accuracy and loss over epochs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()





# from keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import load_model
# import numpy as np
# import os
#
# os.environ['KERAS_BACKEND'] = 'tensorflow'
#
#
# def bone_image(path):
#     img = image.load_img(path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     print(x)
#
#     model = load_model('C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\Model.h5')
#     preds = model.predict(x)
#
#     return preds




# from flask import Flask, render_template, request, session, redirect, url_for
# from werkzeug.utils import secure_filename
# import urllib.request
# import os
# import bone_fracture_detector
#
# app = Flask(__name__, static_folder='static', template_folder='templates')
# UPLOAD_FOLDER = 'static/uploads/'
# RESULT_FOLDER = 'static/results/'
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER
#
# @app.route('/index')
# def home():
#     return render_template('Index.html')
#
# @app.route('/result')
# def image():
#     return render_template('Result.html')
#
# @app.route('/index', methods=['POST'])
# def upload_image():
#     file = request.files['value1']
#     filename = secure_filename(file.filename)
#     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#     with open('filenames.txt', 'a') as f:
#         f.write(filename + '\n')
#     return render_template('Result.html', filename=filename)
#
# @app.route('/image/display/<filename>')
# def display_image(filename):
#     session['filename'] = filename
#     print(filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)
#
# @app.route('/image/detect/')
# def detect():
#     filename = session.get('filename', None)
#     with open('filenames.txt', 'r') as f:
#         filenames = f.readlines()
#         if filenames:
#             filename = filenames[-1].strip()
#     path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     print(path)
#     prediction = bone_fracture_detector.bone_image(path)
#     print(filename, prediction)
#     return render_template('Result.html', filename=filename, prediction=prediction)
#
# if __name__ == '__main__':
#     app.run(debug=True)

