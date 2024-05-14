from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request
import os
import bone_fracture_detector

app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def image():
    return render_template('Result.html')

@app.route('/image', methods=['POST','GET'])
def upload_image():
    file = request.files['value1']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    with open('filenames.txt', 'a') as f:
        f.write(filename + '\n')
    return render_template('Result.html', filename=filename)

@app.route('/image/display/<filename>')
def display_image(filename):
    session['filename'] = filename
    print(filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/image/detect/')
def detect():
    filename = session.get('filename', None)
    with open('filenames.txt', 'r') as f:
        filenames = f.readlines()
        if filenames:
            filename = filenames[-1].strip()
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print("path", path)
    prediction = bone_fracture_detector.bone_image(path)
    print(filename, prediction)
    print(prediction)
    return render_template('Result.html', filename=filename, prediction=prediction )

if __name__ == '__main__':
    app.run(debug=True)








#
# from flask import Flask, render_template, request, session, redirect, url_for
# from werkzeug.utils import secure_filename
# import os
# import bone_fracture_detector
#
# app = Flask(__name__, static_folder='static', template_folder='templates')
# UPLOAD_FOLDER = 'C://Users//venka//PycharmProjects//pythonProject2//static//index.html',
# RESULT_FOLDER = 'C://Users//venka//PycharmProjects//pythonProject2//static//result.html',
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
#     return redirect(url_for('static', filename='index/' + filename))
#
# @app.route('/image/detect/')
# def detect():
#     filename = session.get('filename', None)
#     with open('filenames.txt', 'r') as f:
#         filenames = f.readlines()
#         if filenames:
#             filename = filenames[-1].strip()
#     path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     prediction = bone_fracture_detector.bone_image(path)
#     return render_template('Result.html', filename=filename, prediction=prediction)
#
# if __name__ == '__main__':
#     app.run(debug=True)
