from ultralytics import YOLO

if __name__ == "__main__":
    
    model = YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/ultralytics/cfg/models/11/test1.yaml", task="detect").load('C:/Users/User/Desktop/Yan/University/Yolov11/Yolo_code/yolo11n.pt')  # Load model and make predictions
    # Train the model
    results = model.train(
        data = "C:/Users/User/Desktop/Yan/University/Annotated_dataset/data.yaml",  # path to your data configuration file
        epochs=330,                      # number of epochs
        imgsz=640,                      # image size
        batch=16,                       # batch size
        device='0',                      # cuda device (use '0' for GPU, 'cpu' for CPU)
        #save=True,
        patience=50

    )

    # #Validating the model
    # model=YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/runs/detect/train45/weights/best.pt")

    # metrics = model.val()  # assumes `model` has been loaded
    # print(metrics.box.map)  # mAP50-95
    # print(metrics.box.map50)  # mAP50
    # print(metrics.box.map75)  # mAP75
    # print(metrics.box.maps)  # list of mAP50-95 for each category

    # #Predicting the model
    # model=YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/runs/detect/train45/weights/best.pt")
    # # Predict on all images in the test folder
    # results = model.predict(
    #     source="C:/Users/User/Desktop/Yan/University/Annotated_dataset/test/images",  # Replace with your test folder path
    #     save=True,  # Save results
    #     conf=0.25,  # Confidence threshold
    #     save_txt=True,  # Save results in txt format
    #     save_conf=True,  # Save confidence scores
    #     visualize=False
    # )

    # Process results if you need to work with the detections programmatically
    # for result in results:
    #     boxes = result.boxes  # Get bounding boxes
    #     for box in boxes:
    #         print(f"Detection: {box.cls}")  # Class ID
    #         print(f"Confidence: {box.conf}")  # Confidence score
    #         print(f"Coordinates: {box.xyxy}")  # Box coordinates


