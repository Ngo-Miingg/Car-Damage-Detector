import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO
from PIL import Image

class Detection:
    def __init__(self, model_path: str, classes: list):
        self.model = YOLO(model_path)  # Load mô hình từ best.pt
        self.classes = classes

    def __call__(self, image_path: str, conf_threshold: float = 0.3):
        # Đọc ảnh
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Chạy mô hình YOLOv8
        results = self.model(image)[0]

        boxes, confidences, class_ids = [], [], []
        
        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            conf = result.conf[0].item()
            class_id = int(result.cls[0].item())

            if conf > conf_threshold:
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(round(conf * 100, 2))
                class_ids.append(self.classes[class_id])

        return results, {
            'boxes': boxes,
            'confidences': confidences,
            'classes': class_ids
        }

def draw_boxes(image_path, results):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể load ảnh từ '{image_path}'. Kiểm tra lại đường dẫn.")

    res_plotted = results[0].plot()  # Vẽ kết quả phát hiện lên ảnh
    cv2.imshow("Kết quả dự đoán", res_plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Mở hộp thoại chọn file ảnh
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    img_pth = filedialog.askopenfilename(title="Chọn ảnh để dự đoán", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])

    if not img_pth:
        print("Không có ảnh nào được chọn!")
        exit()

    # Danh sách class của YOLOv8
    CLASSES_YOLO = ['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 
                    'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']

    # Load mô hình
    detection = Detection(model_path='best.pt', classes=CLASSES_YOLO)

    # Chạy mô hình và hiển thị kết quả
    results, output = detection(img_pth)
    print(output)
    draw_boxes(img_pth, results)
