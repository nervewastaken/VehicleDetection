from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO('yolov8n.pt')  # you can choose other YOLOv8 models such as yolov8s.pt, yolov8m.pt, etc.

    # Train the model
    model.train(data='C:/PyProj/newyolo/main/improvedataset/data.yaml', epochs=50, imgsz=640, batch=16, project='runs2/train', name='exp')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
