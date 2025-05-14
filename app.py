from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from deployment import Detection, draw_boxes
import cv2

app = Flask(__name__)

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Danh sách class của YOLOv8
CLASSES_YOLO = ['cửa bị hỏng', 'cửa sổ bị hỏng', 'đèn pha bị hỏng', 'gương bị hỏng',
'vết lõm', 'mui xe bị hỏng', 'cản xe bị hỏng', 'kính chắn gió bị hỏng']

# Load mô hình
detection = Detection(model_path='best.pt', classes=CLASSES_YOLO)

@app.route('/')
def index():    
    return render_template('index.html')  # Giao diện chọn ảnh

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Chạy mô hình YOLOv8
        results, output = detection(file_path)

        # Kiểm tra nếu không có kết quả
        if len(results) == 0:
            return render_template('result.html', 
                                   image_url=None, 
                                   output={'boxes': [], 'classes': [], 'confidences': []},
                                   message="Không phát hiện đối tượng nào trong ảnh.")

        # Lưu ảnh với bounding boxes
        image_with_boxes = results[0].plot()
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
        cv2.imwrite(output_image_path, image_with_boxes)

        # Render template với kết quả
        return render_template('result.html', 
                               image_url=f"/uploads/result_{filename}", 
                               output=output,
                               zip=zip)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)