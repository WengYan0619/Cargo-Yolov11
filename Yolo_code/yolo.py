

from ultralytics import YOLO

if __name__ == "__main__":
    
    model = YOLO("yolo11n.pt", task="detect")  # Load model and make predictions
    # Train the model
    results = model.train(
        data = "C:/Users/User/Downloads/Uno/data.yaml",  # path to your data configuration file
        epochs=1,                      # number of epochs
        imgsz=640,                      # image size
        batch=16,                       # batch size
        device='0',                      # cuda device (use '0' for GPU, 'cpu' for CPU)
        visualize=True,                 # plot results
    )

# out_file = model.export(
#     format="torchscript",
#     imgsz=640,         # (int | list) input images size for exported model
#     batch=1,           # (int) batch size for exported model
#     keras=False,       # (bool) use Keras
#     optimize=False,    # (bool) TorchScript: optimize for mobile
#     half=False,        # (bool) ONNX/TF/TensorRT: FP16 quantization
#     int8=False,        # (bool) CoreML/TF/TensorRT/OpenVino INT8 quantization
#     dynamic=False,     # (bool) ONNX/TF/TensorRT: dynamic axes
#     simplify=False,    # (bool) ONNX: simplify model using `onnxslim`
#     opset=None,        # (int, optional) ONNX: opset version
#     workspace=4,       # (int) TensorRT: workspace size (GiB)
#     nms=False,         # (bool) CoreML: add NMS
# )
# # reference https://docs.ultralytics.com/modes/export
# from ultralytics import YOLO
# import os
# print(os.path.exists('C:/Users/User/Downloads/images/valid/images'))
