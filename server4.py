from flask import Flask, render_template, request
from flask import Markup, jsonify

import uuid
from datetime import datetime

import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
import glob
from tqdm import tqdm
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

#YOLO PARAMS
yolo_params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

#Classes
classes = load_classes(yolo_params["class_path"])

detectron = YOLOv3Predictor(params=yolo_params)

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "storages/images"

# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                print(image.filename)
                print(app.config['UPLOAD_FOLDER'])
                now = datetime.now()
                timestamp = now.strftime('%Y%m%dT%H%M%S') + ('%02d' % (now.microsecond / 10000))
                filename = timestamp + '_' + str(uuid.uuid4().hex) + '_' + image.filename
                source = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print("Save = ", source)
                image.save(source)

                img = cv2.imread(source)
                detections = detectron.get_detections(img)

                rs = []
                
                if len(detections) != 0 :
                    detections.sort(reverse=False ,key = lambda x:x[4])
                    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))           
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                            data = (int(cls_pred), float(cls_conf), x1, y1, x2, y2, classes[int(cls_pred)])
                            rs.append(data)

                # Trả về kết quả
                return jsonify({'status': 200, 'data': rs, 'message': 'success'})

            else:
                # Nếu không có file thì yêu cầu tải file
                return jsonify({'status': 500, 'message': 'an error occus'})

         except Exception as ex:
            print(ex)
            # Nếu lỗi thì thông báo
            return jsonify({'status': 400, 'message': 'an error occus'})

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)

