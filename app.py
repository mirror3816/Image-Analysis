from flask import Flask, render_template, request
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv4 model and weights
yolo_net = cv2.dnn.readNet('yolo_weights/yolov4.weights', 'yolo_weights/yolov4.cfg')
layer_names = yolo_net.getLayerNames()
output_layers = yolo_net.getUnconnectedOutLayersNames()

def analyze_image(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # COCO class names 파일을 불러옵니다.
    with open('coco.names', 'r') as f:
        coco_names = f.read().strip().split('\n')

    image = cv2.imread(image_path)
    height, width, channels = image.shape

    for i in range(len(boxes)):
        if i in indexes:
            class_id = class_ids[i]
            label = coco_names[class_id]
            confidence = confidences[i]
            box = boxes[i]
            
            # 객체 주위에 사각형을 그립니다.
            color = (0, 255, 0)  # 사각형 색상 (여기서는 녹색)
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
            
            # 클래스 이름과 신뢰도를 표시합니다.
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(image, label_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 분석 결과를 이미지 파일에 저장하거나 웹 페이지에 표시할 수 있습니다.
    result_image_path = 'static/result_image.jpg'  # 결과 이미지 파일 경로
    cv2.imwrite(result_image_path, image)  # 결과 이미지 파일로 저장

    return result_image_path  # 결과 이미지 파일 경로를 반환합니다.

@app.route('/')
def home():
    return render_template('index.html', image_path=None, result=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        result = analyze_image(image_path)

        return render_template('index.html', image_path=result, result=result)

if __name__ == '__main__':
    app.run(debug=True)