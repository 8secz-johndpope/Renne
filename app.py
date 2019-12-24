# -*- coding: utf-8 -*-
# Source1: https://gist.github.com/greyli/a643aaac06ea8c23769c0c3d9ccaae79#file-upload-py
import os
import base64
import time
import joblib
import cv2
import numpy as np
from flask import Flask, request, url_for, send_from_directory, render_template, redirect, session
from werkzeug import secure_filename
from openpose_pytorch import body
from pose2seg import segout
from edge_connect import edgec


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = str(hash(time.time()))
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/upload'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

# 页面1：上传图片->选择人物蒙版
@app.route('/upload', methods=['GET', 'POST'])
def select():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            session['hash'] = str(hash(time.time()))

            session['name'] = session['hash'] + '.' + \
                secure_filename(file.filename).rsplit('.', 1)[1]
            session['path'] = os.path.join(
                app.config['UPLOAD_FOLDER'], session['name'])
            file.save(session['path'])
            file_url = url_for('uploaded_file', filename=session['name'])

            img = cv2.imread(session['path'])
            ratio = np.min((1.0, 800.0 / np.max(img.shape[:2])))
            img = cv2.resize(img, None, fx=ratio, fy=ratio)
            (height, width) = img.shape[:2]
            cv2.imwrite(session['path'].split('.')[0] + '_resize.jpg', img)
            poses = body.detect_pose(img)
            masks = segout.pose_seg(img, poses)
            joblib.dump((masks), session['path'].split(
                '.')[0] + '.msk', compress=3)
            mask_poly = [segout.mask2poly(mask) for mask in masks]
            return render_template('select.html', poly_list=mask_poly, image=file_url, width=width, height=height)
    return redirect(url_for('upload_file'))

# 页面2：选择人物蒙版->编辑蒙版
@app.route('/modify', methods=['GET', 'POST'])
def modify_image():
    check_result = request.form.copy().to_dict()
    masks = joblib.load(session['path'].split('.')[0] + '.msk')
    mask = segout.mask_generate(masks, check_result)
    (height, width) = mask.shape
    maskcode = cv2.imencode('.png', cv2.merge(
        [mask * 0 for i in range(2)] + [mask * 255 for i in range(2)]))[1].tostring()
    mask_url = 'data:image/png;base64,' + str(base64.b64encode(maskcode))[2:-1]

    file_url = url_for('uploaded_file', filename=session['name'])

    return render_template('modify.html', mask=mask_url, image=file_url, width=width, height=height)

# 页面3：编辑蒙版->显示结果
@app.route('/result', methods=['GET', 'POST'])
def generate():
    mask = request.form.get('maskgen')
    maskdata = base64.b64decode(mask.split(',', 1)[1])
    img_array = np.fromstring(maskdata, np.uint8)
    # Edge-Connect处理
    inpainted = edgec.inpaint(cv2.imread(
        session['path'].split('.')[0] + '_resize.jpg'), cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(session['path'].split('.')[0] + '_result.jpg', inpainted)
    res_url = url_for(
        'uploaded_file', filename=session['name'].rsplit('.', 1)[0] + '_result.jpg')
    return render_template('result.html', result=res_url)

# 还没用到
@app.route('/exit', methods=['GET', 'POST'])
def clear_file():
    os.remove(os.path.split(session['path'])[0])

# 页面0：上传图片
@app.route('/')
def upload_file():
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
