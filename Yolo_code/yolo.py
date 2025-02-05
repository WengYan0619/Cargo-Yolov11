from ultralytics import YOLO
from ultralytics import RTDETR
import csv

if __name__ == "__main__":
    #model = RTDETR("rtdetr-resnet101.yaml").load('C:/Users/User/Desktop/Yan/University/Yolov11/Yolo_code/rtdetr-l.pt') 
    model = YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/ultralytics/cfg/models/11/yolo-cbam2.yaml", task="detect").load('C:/Users/User/Desktop/Yan/University/Yolov11/Yolo_code/yolo11n.pt')  # Load model and make predictions
    # Train the model
    results = model.train(
        data = "C:/Users/User/Desktop/Yan/University/Annotated_dataset/data.yaml",  # path to your data configuration file
        epochs=330,                      # number of epochs
        imgsz=1088,                      # image size
        batch=16,                       # batch size
        device='0',                      # cuda device (use '0' for GPU, 'cpu' for CPU)
        #save=True,
        patience=50

    )

    # #Validating the model

    # file_path = "C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/runs/detect/train61/weights/best.pt"
    # #model=YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/runs/detect/Nano(250_epoch)/weights/best.pt")
    # file_name=file_path.split("/")[-3]
    # output_csv_path = f"{file_path.rsplit('/', 2)[0]}/{file_name}_metrics.csv"
    # model=YOLO(file_path)

    # metrics = model.val(imgsz=1088)  # assumes `model` has been loaded

    # # Metrics to save
    # data = [
    #     ["mAP50-95", metrics.box.map],
    #     ["mAP50", metrics.box.map50],
    #     ["mAP75", metrics.box.map75],
    #     ["mAPs per category"]
    # ]
    # # Adding mAPs for each category to the data
    # data.extend(["Category " + str(i+1), mAP] for i, mAP in enumerate(metrics.box.maps))

    # # Save to CSV
    # with open(output_csv_path, "w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)
    
    # print(f"Metrics written to {output_csv_path}")
    # # print(metrics.box.map)  # mAP50-95
    # # print(metrics.box.map50)  # mAP50
    # # print(metrics.box.map75)  # mAP75
    # # print(metrics.box.maps)  # list of mAP50-95 for each category



    #Predicting the model
    # model=YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/runs/detect/train39/weights/best.pt")
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

    # #Testing if yaml file is valid
    # model = YOLO("C:/Users/User/Desktop/Yan/University/Yolov11/ultralytics/ultralytics/cfg/models/11/yolo-new-1.yaml", task="detect").load('C:/Users/User/Desktop/Yan/University/Yolov11/Yolo_code/yolo11n.pt')  # Load model and make predictions
    # model.info(detailed=True)  # Get detailed model information
    # try:
    #     model.profile(imgsz=[640,640])
    # except Exception as e:
    #     print(e)
    #     pass
    # model.fuse()