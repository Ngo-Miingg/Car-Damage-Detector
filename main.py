from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
results = model.train(data="/hdd2/minhnv/Damaged-Car-parts-prediction-using-YOLOv8-main/config.yaml", epochs=1153)  # train the model
